/* Byte-level BPE tokenizer for bedtime-LM. Reads the Hugging Face `tokenizer.json` the model was
 * trained with and reproduces `tokenizers` behaviour: GPT-2 style pre-tokenization, byte-to-unicode
 * mapping, BPE merges by rank, and the added tokens "bos"/"eos" (matched anywhere in the text).
 * Works in a Worker (importScripts) and in Node (require) so it can be tested against Python.
 */
(function (root) {
  "use strict";

  // GPT-2's reversible byte <-> printable-unicode table.
  function bytesToUnicode() {
    const bs = [];
    for (let i = 33; i <= 126; i++) bs.push(i);
    for (let i = 161; i <= 172; i++) bs.push(i);
    for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice();
    let n = 0;
    for (let b = 0; b < 256; b++) {
      if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
    }
    const enc = new Array(256);
    const dec = new Map();
    for (let i = 0; i < bs.length; i++) { enc[bs[i]] = String.fromCharCode(cs[i]); dec.set(String.fromCharCode(cs[i]), bs[i]); }
    return { enc, dec };
  }

  const PRETOKENIZE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

  class BedtimeTokenizer {
    constructor(json) {
      const model = json.model;
      if (model.type !== "BPE") throw new Error("expected a BPE tokenizer");
      this.vocab = model.vocab;
      this.idToToken = [];
      for (const [tok, id] of Object.entries(this.vocab)) this.idToToken[id] = tok;
      this.ranks = new Map();
      model.merges.forEach((m, i) => {
        const pair = Array.isArray(m) ? m : m.split(" ");
        this.ranks.set(pair[0] + " " + pair[1], i);
      });
      this.added = (json.added_tokens || []).map((t) => ({ content: t.content, id: t.id }));
      this.specialIds = new Set(this.added.filter((t, i) => json.added_tokens[i].special).map((t) => t.id));
      const { enc, dec } = bytesToUnicode();
      this.byteEnc = enc; this.byteDec = dec;
      this.bpeCache = new Map();
      this.utf8 = new TextEncoder();
      this.utf8dec = new TextDecoder("utf-8", { fatal: false });
      this.vocabSize = this.idToToken.length;
    }

    _bpe(word) {
      const hit = this.bpeCache.get(word);
      if (hit) return hit;
      let parts = Array.from(word);
      while (parts.length > 1) {
        let best = -1, bestRank = Infinity;
        for (let i = 0; i < parts.length - 1; i++) {
          const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
          if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
        }
        if (best < 0) break;
        parts = parts.slice(0, best).concat([parts[best] + parts[best + 1]], parts.slice(best + 2));
      }
      this.bpeCache.set(word, parts);
      return parts;
    }

    _encodePlain(text, out) {
      for (const m of text.matchAll(PRETOKENIZE)) {
        const bytes = this.utf8.encode(m[0]);
        let word = "";
        for (const b of bytes) word += this.byteEnc[b];
        for (const piece of this._bpe(word)) {
          const id = this.vocab[piece];
          if (id === undefined) throw new Error("token not in vocab: " + piece);
          out.push(id);
        }
      }
    }

    // Added tokens ("bos", "eos") are split out of the text wherever they occur, like `tokenizers` does.
    encode(text) {
      const out = [];
      let rest = text;
      for (;;) {
        let at = -1, tok = null;
        for (const t of this.added) {
          const i = rest.indexOf(t.content);
          if (i >= 0 && (at < 0 || i < at || (i === at && t.content.length > tok.content.length))) { at = i; tok = t; }
        }
        if (!tok) break;
        this._encodePlain(rest.slice(0, at), out);
        out.push(tok.id);
        rest = rest.slice(at + tok.content.length);
      }
      this._encodePlain(rest, out);
      return out;
    }

    // Special tokens are skipped, as Python's decode(skip_special_tokens=True) does.
    decode(ids) {
      const bytes = [];
      for (const id of ids) {
        if (this.specialIds.has(id)) continue;
        const tok = this.idToToken[id];
        if (tok === undefined) continue;
        for (const ch of tok) bytes.push(this.byteDec.get(ch));
      }
      return this.utf8dec.decode(Uint8Array.from(bytes));
    }
  }

  root.BedtimeTokenizer = BedtimeTokenizer;
  if (typeof module !== "undefined" && module.exports) module.exports = BedtimeTokenizer;
})(typeof self !== "undefined" ? self : globalThis);
