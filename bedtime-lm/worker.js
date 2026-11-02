/* bedtime-LM inference worker: downloads (and caches) the model, runs it with onnxruntime-web,
 * and streams generated text back. Mirrors the notebook's inference_loop: a 128-token window,
 * a KV cache that slides once it is full, and RoPE positions that keep growing.
 */
"use strict";
// Scripts are loaded from `init`, using an absolute base URL, so this file also works when it
// is started from a blob: URL (needed when the bundle is hosted on a different origin than the page).

const SEQ = 128, L = 6, H = 6, HD = 64;          // trained context, layers, heads, head dim
const MARK = "<|endoftext|>";                   // the TinyStories story separator
const CACHE_NAME = "bedtime-lm-model-v1";

let sess = null, tok = null, busy = false, cancelFlag = false;

// sess.run() settles through microtasks only, which would starve incoming messages (Stop, busy
// checks) until generation ends. Hop through a MessageChannel once per token to let them in;
// unlike setTimeout(0) it is not clamped to 4 ms.
const hop = new MessageChannel();
const yieldToEvents = () => new Promise((res) => { hop.port1.onmessage = () => res(); hop.port2.postMessage(0); });

const post = (type, payload) => self.postMessage(Object.assign({ type }, payload));

// ---- model download with progress, stored in the Cache API so later visits are instant ----
async function loadModelBytes(url, expectedBytes) {
  let cache = null;
  try {
    cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (hit) {
      const buf = await hit.arrayBuffer();
      post("progress", { stage: "cached", loaded: buf.byteLength, total: buf.byteLength });
      return { buf, fromCache: true, cache };
    }
  } catch (_) { cache = null; }                 // Cache API unavailable (private mode, http, ...)

  const resp = await fetch(url);
  if (!resp.ok) throw new Error("model download failed (HTTP " + resp.status + ")");
  const total = Number(resp.headers.get("content-length")) || expectedBytes || 0;
  const reader = resp.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    post("progress", { stage: "download", loaded, total: Math.max(total, loaded) });
  }
  const bytes = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { bytes.set(c, off); off += c.length; }
  return { buf: bytes.buffer, fromCache: false, cache, bytes };
}

async function init(cfg) {
  const base = cfg.base || new URL(".", self.location.href).href;
  importScripts(new URL("vendor/ort/ort.wasm.min.js", base).href, new URL("tokenizer.js", base).href);
  ort.env.wasm.wasmPaths = new URL(cfg.wasmPath || "vendor/ort/", base).href;
  ort.env.wasm.numThreads = 1;                  // no special server headers needed; threads gave no speedup
  const tj = await (await fetch(cfg.tokenizerUrl)).json();
  tok = new BedtimeTokenizer(tj);

  let { buf, fromCache, cache, bytes } = await loadModelBytes(cfg.modelUrl, cfg.modelBytes);
  post("progress", { stage: "compile" });
  const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
  try {
    sess = await ort.InferenceSession.create(buf, opts);
  } catch (e) {
    if (!fromCache) throw e;
    // a bad cached copy: drop it and download again
    try { await cache.delete(cfg.modelUrl); } catch (_) {}
    ({ buf, cache, bytes } = await loadModelBytes(cfg.modelUrl, cfg.modelBytes));
    sess = await ort.InferenceSession.create(buf, opts);
  }
  if (cache && bytes) {                         // only cache a copy that actually loaded
    try { await cache.put(cfg.modelUrl, new Response(bytes, { headers: { "content-type": "application/octet-stream" } })); } catch (_) {}
  }
  post("ready", { vocab: tok.vocabSize });
}

// ---- one forward pass (prefill or decode step) ----
async function step(ids, pk, pv, pos) {
  const r = await sess.run({
    input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, (v) => BigInt(v)), [1, ids.length]),
    past_k: pk,
    past_v: pv,
    position: new ort.Tensor("int64", BigInt64Array.of(BigInt(pos)), [1]),
  });
  return [r.logits.data, r.present_k, r.present_v];
}

const emptyCache = () => new ort.Tensor("float32", new Float32Array(0), [L, 1, H, 0, HD]);

// drop the `drop` oldest positions from a [L,1,H,P,HD] cache
function slide(t, drop) {
  const P = t.dims[3], P2 = P - drop, d = t.data;
  const out = new Float32Array(L * H * P2 * HD);
  for (let lh = 0; lh < L * H; lh++) {
    out.set(d.subarray(lh * P * HD + drop * HD, (lh + 1) * P * HD), lh * P2 * HD);
  }
  return new ort.Tensor("float32", out, [L, 1, H, P2, HD]);
}

// ---- sampling ----
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sample(logits, temperature, topK, rand) {
  const V = logits.length;
  if (!(temperature > 0)) {                     // temperature <= 0: greedy (used for testing)
    let b = 0;
    for (let i = 1; i < V; i++) if (logits[i] > logits[b]) b = i;
    return b;
  }
  let idx = null;
  if (topK > 0 && topK < V) {
    idx = Array.from({ length: V }, (_, i) => i).sort((a, b) => logits[b] - logits[a]).slice(0, topK);
  }
  const n = idx ? idx.length : V;
  let mx = -Infinity;
  for (let j = 0; j < n; j++) { const v = logits[idx ? idx[j] : j]; if (v > mx) mx = v; }
  const w = new Float64Array(n);
  let sum = 0;
  for (let j = 0; j < n; j++) { w[j] = Math.exp((logits[idx ? idx[j] : j] - mx) / temperature); sum += w[j]; }
  let r = rand() * sum;
  for (let j = 0; j < n; j++) { r -= w[j]; if (r <= 0) return idx ? idx[j] : j; }
  return idx ? idx[n - 1] : n - 1;
}

// hold back text that might still change: an unfinished UTF-8 character or the start of <|endoftext|>
function stable(s) {
  let end = s.length;
  while (end > 0 && s[end - 1] === "�") end--;
  for (let k = Math.min(MARK.length - 1, end); k > 0; k--) {
    if (s.slice(end - k, end) === MARK.slice(0, k)) { end -= k; break; }
  }
  return s.slice(0, end);
}

async function generate(m) {
  const id = m.id;
  const maxNew = Math.max(1, Math.min(400, m.maxNewTokens == null ? 100 : m.maxNewTokens));
  const temperature = m.temperature == null ? 0.7 : m.temperature;
  const topK = m.topK || 0;
  const rand = m.seed == null ? Math.random : mulberry32(m.seed >>> 0);

  const ids = tok.encode(m.prompt);
  if (!ids.length) throw new Error("Type a few words to start the story.");
  const window = ids.slice(-SEQ);                // like the notebook, only the last 128 tokens matter
  const t0 = performance.now();
  cancelFlag = false;

  let [lg, pk, pv] = await step(window, emptyCache(), emptyCache(), 0);
  let pos = window.length;
  const gen = [];
  let emitted = "", stopped = "length", final = null;

  for (let i = 0; i < maxNew; i++) {
    await yieldToEvents();
    if (cancelFlag) { stopped = "cancelled"; break; }
    const t = sample(lg, temperature, topK, rand);
    if (tok.specialIds.has(t)) { stopped = "eos"; break; }
    gen.push(t);
    const full = tok.decode(gen);
    const cut = full.indexOf(MARK);
    const visible = cut >= 0 ? full.slice(0, cut) : stable(full);
    if (visible.length > emitted.length && visible.startsWith(emitted)) {
      post("token", { id, text: visible.slice(emitted.length) });
      emitted = visible;
    }
    if (cut >= 0) { stopped = "endoftext"; final = visible; break; }
    if (i === maxNew - 1) break;
    if (pk.dims[3] >= SEQ) {                     // window is full: forget the oldest position
      const drop = pk.dims[3] - (SEQ - 1);
      pk = slide(pk, drop); pv = slide(pv, drop);
    }
    [lg, pk, pv] = await step([t], pk, pv, pos);
    pos++;
  }

  if (final === null) final = tok.decode(gen);
  if (final.length > emitted.length && final.startsWith(emitted)) {
    post("token", { id, text: final.slice(emitted.length) });
  }
  post("done", { id, text: final, tokenIds: gen, stopped, ms: Math.round(performance.now() - t0), promptTokens: window.length });
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "init") {
      await init(m);
    } else if (m.type === "generate") {
      if (!sess) throw new Error("model is not loaded yet");
      if (busy) throw new Error("already generating");
      busy = true;
      try { await generate(m); } finally { busy = false; }
    } else if (m.type === "cancel") {
      cancelFlag = true;
    }
  } catch (err) {
    post("error", { id: m.id, message: String((err && err.message) || err) });
  }
};
