/* bedtime-LM in the browser: a small wrapper around worker.js.
 *
 *   const lm = new BedtimeLM();                       // files are found next to this script
 *   await lm.load(({stage, loaded, total}) => ...);   // first use downloads the model, then it is cached
 *   const r = await lm.generate("Once upon a time", { maxNewTokens: 100, temperature: 0.7,
 *                                                     onToken: (text) => ..., signal });
 *   // r = { text, tokenIds, stopped: "length" | "endoftext" | "eos" | "cancelled", ms, promptTokens }
 *
 * The bundle (this file, worker.js, tokenizer.*, model/, vendor/) can live on any static host that sends
 * CORS headers (GitHub Pages, jsDelivr, a Hugging Face repo...). Pass { baseUrl: "https://host/path/" } when this
 * file is not loaded from there (for example when it is pasted into a page).
 */
(function (root) {
  "use strict";
  const SCRIPT_URL = (typeof document !== "undefined" && document.currentScript && document.currentScript.src) || "";

  class BedtimeLM {
    static supported() {
      return typeof WebAssembly === "object" && typeof Worker === "function" && typeof BigInt64Array === "function";
    }

    // options: baseUrl, workerUrl, modelUrl, tokenizerUrl, wasmPath, modelBytes (for the progress bar)
    constructor(opts = {}) {
      const base = opts.baseUrl || (SCRIPT_URL ? new URL(".", SCRIPT_URL).href : "");
      if (!base) throw new Error("BedtimeLM needs { baseUrl } when it is not loaded from a <script src>");
      const here = (p) => new URL(p, base).href;
      this.opts = Object.assign({
        baseUrl: base,
        workerUrl: here("worker.js"),
        modelUrl: here("model/bedtime_kv_int8.onnx?v=1"),   // bump ?v= when you ship a new model
        tokenizerUrl: here("tokenizer.json"),
        wasmPath: here("vendor/ort/"),
        modelBytes: 26810383,
      }, opts);
      this.worker = null;
      this.ready = null;
      this.nextId = 1;
      this.jobs = new Map();
      this.onProgress = null;
    }

    // A Worker must be same-origin; for a bundle on another host, start it from a blob made from its source.
    async _startWorker() {
      const url = this.opts.workerUrl;
      if (new URL(url, location.href).origin === location.origin) return new Worker(url);
      const src = await (await fetch(url)).text();
      return new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
    }

    load(onProgress) {
      if (onProgress) this.onProgress = onProgress;
      if (this.ready) return this.ready;
      this.ready = new Promise(async (resolve, reject) => {
        let w;
        try { w = await this._startWorker(); } catch (e) { reject(e); return; }
        this.worker = w;
        w.onerror = (e) => {
          const err = new Error(e.message || "worker failed");
          reject(err);
          for (const j of this.jobs.values()) j.reject(err);
          this.jobs.clear();
        };
        w.onmessage = (e) => {
          const m = e.data;
          if (m.type === "progress") { if (this.onProgress) this.onProgress(m); }
          else if (m.type === "ready") resolve(m);
          else if (m.type === "token") { const j = this.jobs.get(m.id); if (j && j.onToken) j.onToken(m.text); }
          else if (m.type === "done") { const j = this.jobs.get(m.id); if (j) { this.jobs.delete(m.id); j.resolve(m); } }
          else if (m.type === "error") {
            const err = new Error(m.message);
            const j = this.jobs.get(m.id);
            if (j) { this.jobs.delete(m.id); j.reject(err); } else reject(err);
          }
        };
        w.postMessage(Object.assign({ type: "init", base: this.opts.baseUrl }, this.opts));
      });
      this.ready.catch(() => { this.ready = null; if (this.worker) { this.worker.terminate(); this.worker = null; } });
      return this.ready;
    }

    async generate(prompt, o = {}) {
      await this.load();
      if (o.signal && o.signal.aborted) return { text: "", tokenIds: [], stopped: "cancelled", ms: 0, promptTokens: 0 };
      const id = this.nextId++;
      return new Promise((resolve, reject) => {
        this.jobs.set(id, { resolve, reject, onToken: o.onToken });
        if (o.signal) o.signal.addEventListener("abort", () => this.cancel(), { once: true });
        this.worker.postMessage({
          type: "generate", id, prompt,
          maxNewTokens: o.maxNewTokens, temperature: o.temperature, topK: o.topK, seed: o.seed,
        });
      });
    }

    cancel() { if (this.worker) this.worker.postMessage({ type: "cancel" }); }

    dispose() { if (this.worker) this.worker.terminate(); this.worker = null; this.ready = null; this.jobs.clear(); }
  }

  root.BedtimeLM = BedtimeLM;
})(self);
