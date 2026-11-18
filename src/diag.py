#!/usr/bin/env python3
"""Diagnostics for the MIMIC LoRA pipeline. Prints ONLY counts / structure, never note text.
Usage:  ~/jupyterlab/.venv/bin/python ~/mimic_project/diag.py
"""
import json, os, re, sys
from collections import Counter
from pathlib import Path
import numpy as np

OUT = Path(os.path.expanduser(os.environ.get("IE_OUT_DIR", "~/mimic_project/ie_pipeline")))

def shape(o, depth=0):
    """Describe JSON structure using key names and types only."""
    if isinstance(o, dict):
        if depth >= 3: return "{...}"
        return "{" + ", ".join(f"{k}: {shape(v, depth + 1)}" for k, v in list(o.items())[:6]) + ("" if len(o) <= 6 else ", ...") + "}"
    if isinstance(o, list):
        return f"list[{len(o)}] of " + (shape(o[0], depth + 1) if o else "empty")
    return type(o).__name__

def try_json(raw):
    if raw is None: return None
    s = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    s = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.M).strip()
    try: return json.loads(s)
    except Exception:
        m = re.search(r"[\{\[].*[\}\]]", s, flags=re.S)
        if m:
            try: return json.loads(m.group(0))
            except Exception: return None
    return None

print("=" * 70, "\nPREDICTION FILES\n" + "=" * 70)
for fn in ["B1_qwen25_3b_zeroshot", "FT_qwen25_3b_lora", "B2_qwen36_zeroshot"]:
    p = OUT / "preds" / f"{fn}.jsonl"
    if not p.exists(): print(f"\n{fn}: missing"); continue
    rows = []
    for line in open(p):
        try: rows.append(json.loads(line))
        except Exception: pass
    fin = Counter(r.get("finish_reason") for r in rows)
    toks = [r.get("completion_tokens") for r in rows if isinstance(r.get("completion_tokens"), (int, float))]
    lat = [r.get("latency_s") for r in rows if isinstance(r.get("latency_s"), (int, float))]
    parsed = [try_json(r.get("raw")) for r in rows]
    shapes = Counter(shape(o) if o is not None else "NOT_JSON" for o in parsed)
    top_keys = Counter(k for o in parsed if isinstance(o, dict) for k in o.keys())
    item_keys, type_vals = Counter(), Counter()
    for o in parsed:
        items = (o.get("entities") or next((v for v in o.values() if isinstance(v, list)), None)) if isinstance(o, dict) else o if isinstance(o, list) else None
        if isinstance(items, list):
            for e in items[:50]:
                if isinstance(e, dict):
                    item_keys.update(e.keys()); type_vals[str(e.get("type"))[:20]] += 1
    print(f"\n--- {fn}: {len(rows)} rows")
    print("finish_reason:", dict(fin))
    if toks: print(f"completion tokens: median {np.median(toks):.0f} | p95 {np.percentile(toks, 95):.0f} | max {max(toks)}")
    if lat:  print(f"latency s: median {np.median(lat):.0f} | p95 {np.percentile(lat, 95):.0f}")
    print("valid JSON:", sum(o is not None for o in parsed), "/", len(rows))
    print("top-level keys:", dict(top_keys.most_common(8)))
    print("entity item keys:", dict(item_keys.most_common(10)))
    print("'type' values:", dict(type_vals.most_common(8)))
    print("most common structures:")
    for s, c in shapes.most_common(4): print(f"   {c:4d} x {s[:150]}")

print("\n" + "=" * 70, "\nTRAINING DATA LENGTH (tokens, prompt+completion)\n" + "=" * 70)
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    for sp in ["train", "val"]:
        L, Lc = [], []
        for line in open(OUT / "data" / f"sft_{sp}.jsonl"):
            d = json.loads(line)
            msgs = d["prompt"] + d["completion"]
            r = tok.apply_chat_template(msgs, tokenize=True)
            L.append(len(r["input_ids"] if hasattr(r, "keys") else r))
            Lc.append(len(tok(d["completion"][0]["content"])["input_ids"]))
        L, Lc = np.array(L), np.array(Lc)
        print(f"{sp}: n={len(L)} | total median {np.median(L):.0f}, p95 {np.percentile(L,95):.0f}, max {L.max()} "
              f"| >4096 (TRUNCATED in training): {(L > 4096).sum()} ({(L > 4096).mean():.0%})")
        print(f"      answer-only tokens: median {np.median(Lc):.0f}, p95 {np.percentile(Lc,95):.0f}, max {Lc.max()} "
              f"| >4096 (cut at inference): {(Lc > 4096).sum()}")
except Exception as e:
    print("tokenizer check skipped:", e)

print("\n" + "=" * 70, "\nTRAINING LOSS HISTORY\n" + "=" * 70)
h = OUT / "results" / "train_log_history.json"
if h.exists():
    for e in json.load(open(h)):
        if "eval_loss" in e: print(f"step {e.get('step'):>3}: eval_loss {e['eval_loss']:.4f}")
    st = json.load(open(OUT / "results" / "train_stats.json"))
    print("best checkpoint:", st.get("best_checkpoint"), "| train_loss:", st.get("train_loss"))
