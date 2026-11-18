#!/usr/bin/env python3
"""
MIMIC-III LoRA Clinical IE pipeline - resumable script version
==============================================================

Runs the same steps as MIMIC_LoRA_Clinical_IE_Pipeline_RUN_ORDER.ipynb, but:
  * every heavy step writes a checkpoint, so after a crash / reboot / server stop
    you simply run the script again and it continues where it stopped;
  * finished steps are detected and skipped automatically;
  * it waits for the vLLM server to come back after a reboot;
  * it pauses when the GPU gets too hot (the Spark shut down once under load);
  * it never prints note text (only counts and metrics) to the screen or log.

Usage (on the Spark, inside your Python environment):
    python mimic_pipeline.py --status          # show which checkpoints exist
    python mimic_pipeline.py                   # run/resume everything EXCEPT training
    python mimic_pipeline.py --train           # run/resume everything INCLUDING LoRA training
    python mimic_pipeline.py --stage eval      # run one stage only
    stages: prep, silver, labels, train, predict, eval, all (default)

Checkpoints (all under ~/mimic_project/ie_pipeline):
    data/prep_1_clean.parquet         cleaned + exact-dedup notes
    data/prep_2_neardup.json          near-duplicate search progress (resumes mid-way)
    data/prep_3_units.parquet         chunks + patient split
    data/annotation_units.parquet     the 600-chunk sample (NEVER overwritten)
    silver/teacher_constrained.jsonl  Qwen3.6 output, one line per chunk
    silver/silver_clean.parquet       cleaned silver labels
    models/qwen25_3b_lora_ie/checkpoint-*   LoRA checkpoints every few steps
    models/qwen25_3b_lora_ie/TRAINING_DONE.json   marker: training finished
    preds/*.jsonl                     predictions, one line per chunk
    results/*                         evaluation outputs (always regenerated, cheap)
"""
import argparse, os, re, sys, json, time, math, random, hashlib, platform, warnings, logging, threading, subprocess
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION (same values as the notebook)
# =====================================================================
DB_PATH      = os.path.expanduser(os.environ.get("MIMIC_DB", "~/mimic_project/mimic3.duckdb"))
VLLM_URL     = os.environ.get("VLLM_URL", "http://localhost:8001/v1")
MODEL        = "nvidia/Qwen3.6-35B-A3B-NVFP4"
STUDENT_BASE = "Qwen/Qwen2.5-3B-Instruct"
OUT_DIR      = Path(os.path.expanduser(os.environ.get("IE_OUT_DIR", "~/mimic_project/ie_pipeline")))

SEED = 42
NOTE_CATEGORIES   = ["Discharge summary", "Physician ", "Nursing"]
MIN_CHARS, MAX_CHUNK_CHARS = 200, 3000
N_ANNOTATION_UNITS = 600
DOUBLE_ANNOT_FRAC  = 0.20
SPLIT_FRAC         = {"train": 0.70, "val": 0.10, "test": 0.20}

DATA, SILVER, GOLD, MODELS, PREDS, RESULTS, FIGS, LOGS = (OUT_DIR / s for s in
    ["data", "silver", "gold", "models", "preds", "results", "figures", "logs"])
ADAPTER_DIR = MODELS / "qwen25_3b_lora_ie"          # reset in main() if --tag is given
DONE_MARK   = ADAPTER_DIR / "TRAINING_DONE.json"
FT_NAME, FT_FILE = "FT_qwen2.5-3B_LoRA", "FT_qwen25_3b_lora.jsonl"

def set_run_tag(tag):
    """--tag v2 -> separate adapter folder + prediction file, so earlier versions are kept."""
    global ADAPTER_DIR, DONE_MARK, FT_NAME, FT_FILE
    sfx = f"_{tag}" if tag else ""
    ADAPTER_DIR = MODELS / f"qwen25_3b_lora_ie{sfx}"
    DONE_MARK   = ADAPTER_DIR / "TRAINING_DONE.json"
    FT_NAME, FT_FILE = f"FT_qwen2.5-3B_LoRA{sfx}", f"FT_qwen25_3b_lora{sfx}.jsonl"

random.seed(SEED); np.random.seed(SEED)
log = logging.getLogger("pipeline")
ARGS = None   # filled in main()


# =====================================================================
# UTILITIES: logging, lock, thermal guard, vLLM wait, jsonl cache
# =====================================================================
def setup_logging():
    for d in (DATA, SILVER, GOLD, MODELS, PREDS, RESULTS, FIGS, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(LOGS / f"pipeline_{time.strftime('%Y%m%d')}.log"); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    log.handlers = [fh, sh]


class RunLock:
    """Prevents two copies of the pipeline running at the same time."""
    def __init__(self, path): self.path = Path(path)
    def __enter__(self):
        if self.path.exists():
            try:
                pid = int(self.path.read_text().strip())
                os.kill(pid, 0)                       # still alive?
                log.error(f"Another pipeline run is active (pid {pid}). Exiting.")
                sys.exit(1)
            except PermissionError:                   # exists, owned by another user
                log.error("Another pipeline run is active. Exiting."); sys.exit(1)
            except (ValueError, ProcessLookupError):
                log.warning("Removing stale lock file from a previous crashed run.")
        self.path.write_text(str(os.getpid()))
        return self
    def __exit__(self, *a):
        try: self.path.unlink()
        except FileNotFoundError: pass


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return float(out[0]) if out else None
    except Exception:
        return None


def thermal_guard():
    """If the GPU is hotter than --max-temp, pause until it cools to --max-temp minus 10."""
    if not ARGS or ARGS.max_temp <= 0: return
    t = gpu_temp()
    if t is None or t < ARGS.max_temp: return
    log.warning(f"GPU at {t:.0f} C (limit {ARGS.max_temp}). Pausing to cool down...")
    while True:
        time.sleep(60)
        t = gpu_temp()
        if t is None or t <= ARGS.max_temp - 10:
            log.info(f"GPU cooled to {t} C. Resuming."); return


def available_mem_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"): return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return None


def get_client():
    from openai import OpenAI
    return OpenAI(base_url=VLLM_URL, api_key="EMPTY", timeout=1800)


def wait_for_vllm(max_wait_min):
    """After a reboot the vLLM container needs a few minutes to load the model."""
    client = get_client(); t0 = time.time()
    while True:
        try:
            if MODEL in [m.id for m in client.models.list().data]:
                log.info("vLLM server is up and the model is loaded."); return True
            log.warning("vLLM is up but the expected model is not listed yet.")
        except Exception:
            pass
        if (time.time() - t0) / 60 > max_wait_min:
            log.error(f"vLLM not available after {max_wait_min} min. Check: docker ps / docker start <your-vllm-container>")
            return False
        log.info("Waiting for vLLM server (retry in 30 s)...")
        time.sleep(30)


def load_jsonl_cache(path, only_success=True):
    """Read a jsonl checkpoint, skipping half-written lines from a crash."""
    out, bad = {}, 0
    if Path(path).exists():
        for line in open(path):
            try: d = json.loads(line)
            except json.JSONDecodeError: bad += 1; continue
            if only_success and d.get("extraction") is None: continue
            out[d["unit_id"]] = d
    if bad: log.warning(f"{Path(path).name}: skipped {bad} broken line(s)")
    return out


# =====================================================================
# SCHEMA, GUIDELINE, PARSER, EVIDENCE CHECK (final notebook versions incl. Fix A)
# =====================================================================
class Entity(BaseModel):
    text: str
    type: Literal["PROBLEM", "MEDICATION", "TEST", "RESULT"]
    evidence: str
    negated: bool = False
    certainty: Literal["certain", "uncertain", "hypothetical"] = "certain"
    temporality: Literal["current", "historical", "future"] = "current"
    related_test: Optional[str] = None
    value: Optional[str] = None

class Extraction(BaseModel):
    entities: List[Entity] = Field(default_factory=list)

JSON_SCHEMA = Extraction.model_json_schema()

GUIDELINE = '''You are a clinical information extraction system. Extract every clinically relevant entity from the ICU note excerpt.

Entity types:
- PROBLEM: diagnoses, diseases, symptoms, signs, abnormal findings, complications.
- MEDICATION: drugs, IV fluids, blood products, drug classes when given/considered.
- TEST: laboratory tests, imaging, procedures performed for diagnosis, vital-sign measurements.
- RESULT: a reported test result. Put the test name in related_test and the value with units in value.

Attributes:
- negated: true if the entity is explicitly absent/denied/ruled out ("no", "denies", "negative for", "without").
- certainty: "uncertain" for possible/probable/suspected/likely/?/r/o/concern for; "hypothetical" for conditional or instructional mentions ("if", "should", "return if"); else "certain".
- temporality: "historical" for past history / prior episodes (h/o, s/p, PMH section); "future" for planned items; else "current".

Rules:
1. "evidence" MUST be copied verbatim from the note (shortest phrase that supports the entity, ≤ 12 words). Never paraphrase.
2. Do not infer entities that are not stated. If nothing is present return {"entities": []}.
3. Ignore de-identification tokens such as [DATE], [NAME].
4. Expand nothing; keep abbreviations as written in "text".
Return ONLY valid JSON matching the schema.
5. List each distinct entity only once. Do not repeat entries.
6. Output compact single-line JSON with no indentation or line breaks. Omit related_test and value unless type is RESULT.
7. Never use '...' or skip words inside evidence; copy one continuous phrase.'''

def build_messages(note_text):
    return [{"role": "system", "content": GUIDELINE},
            {"role": "user", "content": f"NOTE EXCERPT:\n<<<\n{note_text}\n>>>\n\nReturn the JSON."}]

TYPE_MAP = {"problem":"PROBLEM","diagnosis":"PROBLEM","symptom":"PROBLEM","condition":"PROBLEM",
            "finding":"PROBLEM","disease":"PROBLEM","medication":"MEDICATION","drug":"MEDICATION",
            "med":"MEDICATION","medicine":"MEDICATION","treatment":"MEDICATION","test":"TEST",
            "lab":"TEST","procedure":"TEST","imaging":"TEST","investigation":"TEST",
            "result":"RESULT","value":"RESULT","lab_result":"RESULT","test_result":"RESULT"}
CERT_MAP = {"certain":"certain","definite":"certain","uncertain":"uncertain","possible":"uncertain",
            "probable":"uncertain","hypothetical":"hypothetical","conditional":"hypothetical"}
TEMP_MAP = {"current":"current","present":"current","historical":"historical","past":"historical",
            "history":"historical","future":"future","planned":"future"}

def _fix_entity(e):
    if not isinstance(e, dict): return None
    e = {str(k).lower(): v for k, v in e.items()}
    text = e.get("text") or e.get("entity") or e.get("name") or e.get("mention")
    typ  = TYPE_MAP.get(str(e.get("type") or e.get("category") or e.get("label") or "").strip().lower())
    if not text or not typ:                       # e.g. {"PROBLEM": "pneumonia", ...}
        for k, v in e.items():
            if k in ("problem", "medication", "test", "result") and isinstance(v, str) and v.strip():
                text, typ = v, k.upper(); break
    if not text or not typ: return None
    neg = e.get("negated", e.get("negation", False))
    if isinstance(neg, str): neg = neg.strip().lower() in ("true", "yes", "negated", "absent", "1")
    return dict(text=str(text), type=typ, evidence=str(e.get("evidence") or e.get("span") or text),
                negated=bool(neg),
                certainty=CERT_MAP.get(str(e.get("certainty", "certain")).lower(), "certain"),
                temporality=TEMP_MAP.get(str(e.get("temporality", "current")).lower(), "current"),
                related_test=e.get("related_test"),
                value=None if e.get("value") is None else str(e.get("value")))

_DEC = json.JSONDecoder()
def _salvage(s):
    """Recover the complete entity objects from a cut-off / broken JSON answer."""
    i = s.find("[")
    if i < 0: return []
    out, pos = [], i + 1
    while True:
        j = s.find("{", pos)
        if j < 0: break
        try:
            obj, end = _DEC.raw_decode(s, j); out.append(obj); pos = end
        except json.JSONDecodeError:
            break
    return out

def _to_extraction(items):
    ents = []
    for e in items:
        fe = _fix_entity(e)
        if fe:
            try: ents.append(Entity.model_validate(fe))
            except Exception: pass
    return Extraction(entities=ents)

def parse_json(raw):
    """Returns (Extraction|None, valid_json, schema_valid). valid_json / schema_valid stay STRICT;
    entities are still recovered from near-miss or cut-off answers for scoring."""
    if raw is None: return None, False, False
    s = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    s = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.M).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"[\{\[].*[\}\]]", s, flags=re.S)
        try: obj = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError: obj = None
        if obj is None:                                   # invalid / truncated JSON
            return _to_extraction(_salvage(s)), False, False
    if isinstance(obj, dict) and isinstance(obj.get("entities"), list):   # strict = exact schema
        try:
            return Extraction.model_validate(obj), True, True
        except Exception:
            pass
    items = obj if isinstance(obj, list) else (
        next((v for v in obj.values() if isinstance(v, list)), []) if isinstance(obj, dict) else [])
    return _to_extraction(items), True, False

def _norm(s): return re.sub(r"\s+", " ", (s or "").lower()).strip()

def is_supported(ent, source, thr=90):
    from rapidfuzz import fuzz
    src, tx = _norm(source), _norm(ent.get("text"))
    frags = [f for f in (_norm(x) for x in re.split(r"\.{2,}|…", ent.get("evidence") or "")) if len(f) >= 3]
    if not frags or not tx: return False
    ev_ok = all(f in src or fuzz.partial_ratio(f, src) >= thr for f in frags)
    tx_ok = tx in src or any(tx in f for f in frags) or fuzz.partial_ratio(tx, src) >= thr
    return ev_ok and tx_ok

def entities_of(x):
    return x["entities"] if isinstance(x, dict) and isinstance(x.get("entities"), (list, np.ndarray)) else []


# =====================================================================
# STAGE 1: PREPROCESSING (only if annotation_units.parquet is missing)
# =====================================================================
DEID_MAP = [
    (r"\[\*\*[^\]]*?(Date|Year|Month|Day)[^\]]*?\*\*\]", "[DATE]"),
    (r"\[\*\*\d{4}-\d{1,2}-\d{1,2}\*\*\]",                "[DATE]"),
    (r"\[\*\*\d{1,2}-\d{1,2}\*\*\]",                      "[DATE]"),
    (r"\[\*\*[^\]]*?(Name|Initials)[^\]]*?\*\*\]",        "[NAME]"),
    (r"\[\*\*[^\]]*?(Hospital|Location|Ward|Address|State|Country)[^\]]*?\*\*\]", "[LOCATION]"),
    (r"\[\*\*[^\]]*?(Telephone|Phone|Fax|Pager)[^\]]*?\*\*\]", "[PHONE]"),
    (r"\[\*\*[^\]]*?\*\*\]",                              "[REDACTED]"),
]
def normalise(text):
    t = text.replace("\r", "")
    for pat, rep in DEID_MAP: t = re.sub(pat, rep, t, flags=re.I)
    t = re.sub(r"[ \t]+", " ", t); t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

SECTION_RE = re.compile(r"\n(?=[A-Z][A-Za-z /&]{2,40}:)")
def _segments(text, max_chars):
    for sec in (p.strip() for p in SECTION_RE.split(text)):
        if not sec: continue
        if len(sec) <= max_chars: yield sec; continue
        for para in (p.strip() for p in re.split(r"\n\s*\n", sec)):
            if not para: continue
            if len(para) <= max_chars: yield para; continue
            for sent in re.split(r"(?<=[.;])\s+|\n", para):
                for k in range(0, len(sent), max_chars):
                    if sent[k:k + max_chars].strip(): yield sent[k:k + max_chars].strip()

def chunk_note(text, max_chars=MAX_CHUNK_CHARS):
    chunks, buf = [], ""
    for seg in _segments(text, max_chars):
        if buf and len(buf) + len(seg) + 2 > max_chars: chunks.append(buf); buf = seg
        else: buf += ("\n\n" if buf else "") + seg
    if buf: chunks.append(buf)
    return [c for c in chunks if len(c) >= 100]

def shingles(s, k=5):
    w = re.findall(r"\w+", s.lower())
    return {hash(" ".join(w[i:i+k])) for i in range(max(len(w) - k + 1, 1))}

def stage_prep():
    ann_path = DATA / "annotation_units.parquet"
    if ann_path.exists():
        ann = pd.read_parquet(ann_path)
        log.info(f"[prep] SKIP - annotation sample exists ({len(ann)} chunks). Never overwritten.")
        return ann

    import duckdb
    from tqdm import tqdm
    log.info("[prep] annotation sample not found - running preprocessing with checkpoints")
    con = duckdb.connect(DB_PATH, read_only=True)
    tables = con.execute("SELECT table_schema, table_name FROM information_schema.tables").df()
    m = tables[tables.table_name.str.lower() == "noteevents"].iloc[0]
    TBL = f'"{m.table_schema}"."{m.table_name}"'
    COL = {c.lower(): c for c in con.execute(f"DESCRIBE {TBL}").df().column_name}

    # ---- 1a: load + clean + exact dedup + quality ----
    p1 = DATA / "prep_1_clean.parquet"
    if p1.exists():
        notes = pd.read_parquet(p1); log.info(f"[prep] loaded checkpoint {p1.name} ({len(notes):,} notes)")
    else:
        cats = ", ".join("'" + c.replace("'", "''") + "'" for c in NOTE_CATEGORIES)
        notes = con.execute(f'''
            SELECT {COL['row_id']} AS row_id, {COL['subject_id']} AS subject_id, {COL['hadm_id']} AS hadm_id,
                   {COL['chartdate']} AS chartdate, {COL['category']} AS category, {COL['text']} AS text
            FROM {TBL} WHERE {COL['category']} IN ({cats})
              AND ({COL['iserror']} IS NULL OR {COL['iserror']} <> 1)''').df()
        notes["category"] = notes["category"].str.strip()
        notes["clean"] = notes["text"].map(normalise); notes = notes.drop(columns="text")
        notes["sha1"] = notes["clean"].map(lambda s: hashlib.sha1(s.lower().encode()).hexdigest())
        notes = notes.drop_duplicates("sha1")
        alpha = notes["clean"].map(lambda s: sum(c.isalpha() for c in s) / max(len(s), 1))
        notes = notes[(notes["clean"].str.len() >= MIN_CHARS) & (alpha >= 0.5)].reset_index(drop=True)
        notes.to_parquet(p1); log.info(f"[prep] saved {p1.name} ({len(notes):,} notes)")

    # ---- 1b: near-duplicate search, resumable every 2,000 patients ----
    p2 = DATA / "prep_2_neardup.json"
    state = json.loads(p2.read_text()) if p2.exists() else {"done": [], "drop": [], "finished": False}
    if not state["finished"]:
        done, drop = set(state["done"]), set(state["drop"])
        groups = notes.sort_values("chartdate").groupby("subject_id")
        todo = [sid for sid in groups.groups if sid not in done]
        log.info(f"[prep] near-dup: {len(done):,} patients done, {len(todo):,} to go")
        for i, sid in enumerate(tqdm(todo, desc="near-dup"), 1):
            g = groups.get_group(sid).head(200); seen = []
            for rid, txt in zip(g.row_id, g.clean):
                sh = shingles(txt)
                if any(len(sh & s) / max(len(sh | s), 1) >= 0.8 for s in seen): drop.add(int(rid))
                else: seen.append(sh)
            done.add(int(sid))
            if i % 2000 == 0:
                p2.write_text(json.dumps({"done": list(done), "drop": list(drop), "finished": False}))
        state = {"done": list(done), "drop": list(drop), "finished": True}
        p2.write_text(json.dumps(state))
    notes = notes[~notes.row_id.isin(set(state["drop"]))].reset_index(drop=True)
    log.info(f"[prep] after near-dup removal: {len(notes):,} notes")

    # ---- 1c: chunk + patient split ----
    p3 = DATA / "prep_3_units.parquet"
    if p3.exists():
        units = pd.read_parquet(p3)
    else:
        rows = []
        for r in tqdm(notes.itertuples(), total=len(notes), desc="chunking"):
            for i, ch in enumerate(chunk_note(r.clean)):
                rows.append(dict(unit_id=f"{r.row_id}_{i}", row_id=r.row_id, subject_id=r.subject_id,
                                 hadm_id=r.hadm_id, category=r.category, chunk_idx=i, text=ch))
        units = pd.DataFrame(rows)
        patients = units.subject_id.unique(); rng = np.random.default_rng(SEED); rng.shuffle(patients)
        n = len(patients)
        split_of = {p: ("train" if i < 0.70 * n else "val" if i < 0.85 * n else "test") for i, p in enumerate(patients)}
        units["split"] = units.subject_id.map(split_of)
        assert (units.groupby("subject_id").split.nunique() == 1).all(), "patient leakage!"
        units.to_parquet(p3)
    log.info(f"[prep] {len(units):,} chunks")

    # ---- 1d: sample annotation subset ----
    out = []
    for sp, frac in SPLIT_FRAC.items():
        pool = (units[units.split == sp].sample(frac=1, random_state=SEED)
                .drop_duplicates("row_id").groupby("subject_id").head(2))
        per_cat = int(N_ANNOTATION_UNITS * frac) // len(NOTE_CATEGORIES)
        for cat in pool.category.unique(): out.append(pool[pool.category == cat].head(per_cat))
    ann = pd.concat(out, ignore_index=True)
    ann.to_parquet(ann_path); log.info(f"[prep] saved {ann_path.name} ({len(ann)} chunks)")
    return ann


# =====================================================================
# vLLM EXTRACTION (resumable)
# =====================================================================
def _call(client, text, constrained, max_tokens):
    kw = dict(model=MODEL, messages=build_messages(text), temperature=0.0, max_tokens=max_tokens, seed=SEED,
              extra_body={"chat_template_kwargs": {"enable_thinking": False}, "repetition_penalty": 1.05})
    if constrained:
        kw["response_format"] = {"type": "json_schema", "json_schema": {"name": "extraction", "schema": JSON_SCHEMA}}
    t0 = time.perf_counter(); r = client.chat.completions.create(**kw)
    return r, time.perf_counter() - t0

def _split_half(text):
    mid = len(text) // 2; cut = text.rfind("\n", 0, mid)
    if cut < len(text) * 0.25: cut = mid
    return text[:cut], text[cut:]

def llm_extract(client, text, constrained=True, max_tokens=8192, retries=2, depth=0):
    for attempt in range(retries + 1):
        try:
            r, lat = _call(client, text, constrained, max_tokens)
            raw, fin = r.choices[0].message.content, r.choices[0].finish_reason
            if fin == "length" and depth < 2 and len(text) > 400:
                a, b = _split_half(text)
                ra = llm_extract(client, a, constrained, max_tokens, retries, depth + 1)
                rb = llm_extract(client, b, constrained, max_tokens, retries, depth + 1)
                merged, seen = [], set()
                for e in entities_of(ra["extraction"]) + entities_of(rb["extraction"]):
                    k = (e["type"], e["text"].lower().strip())
                    if k not in seen: seen.add(k); merged.append(e)
                return dict(raw=None, extraction={"entities": merged},
                            valid_json=ra["valid_json"] and rb["valid_json"],
                            valid_schema=ra["valid_schema"] and rb["valid_schema"],
                            latency_s=lat + (ra["latency_s"] or 0) + (rb["latency_s"] or 0),
                            prompt_tokens=None,
                            completion_tokens=(ra["completion_tokens"] or 0) + (rb["completion_tokens"] or 0),
                            finish_reason="split", n_splits=1 + ra["n_splits"] + rb["n_splits"])
            ext, ok_json, ok_schema = parse_json(raw)
            return dict(raw=raw, extraction=ext.model_dump() if ext else None, valid_json=ok_json,
                        valid_schema=ok_schema, latency_s=lat, prompt_tokens=r.usage.prompt_tokens,
                        completion_tokens=r.usage.completion_tokens, finish_reason=fin, n_splits=0)
        except Exception as e:
            if attempt == retries:
                return dict(raw=None, extraction=None, valid_json=False, valid_schema=False, latency_s=None,
                            prompt_tokens=None, completion_tokens=None, finish_reason="error",
                            n_splits=0, error=str(e)[:300])
            time.sleep(2 ** attempt)

def run_batch(df, out_path, workers):
    """Parallel vLLM extraction; each finished chunk is appended + flushed immediately."""
    from tqdm import tqdm
    out_path = Path(out_path)
    done = load_jsonl_cache(out_path)
    todo = df[~df.unit_id.isin(done)]
    log.info(f"[{out_path.stem}] {len(done)} done, {len(todo)} to go")
    if len(todo):
        if not wait_for_vllm(ARGS.vllm_wait): return None
        client, lock = get_client(), threading.Lock()
        with open(out_path, "a") as f, ThreadPoolExecutor(workers) as ex:
            futs = {}
            for u, t in zip(todo.unit_id, todo.text):
                futs[ex.submit(llm_extract, client, t)] = u
            for n, fut in enumerate(tqdm(as_completed(futs), total=len(futs), desc=out_path.stem), 1):
                d = fut.result(); d["unit_id"] = futs[fut]
                with lock:
                    f.write(json.dumps(d) + "\n"); f.flush(); os.fsync(f.fileno())
                done[d["unit_id"]] = d
                if n % 10 == 0: thermal_guard()
    out = pd.DataFrame([done[u] for u in df.unit_id if u in done])
    failed = len(df) - len(out)
    log.info(f"[{out_path.stem}] finished {len(out)}/{len(df)} | still missing/failed: {failed}")
    return out


# =====================================================================
# STAGE 2: SILVER LABELS
# =====================================================================
def stage_silver(ann):
    raw = run_batch(ann, SILVER / "teacher_constrained.jsonl", ARGS.workers)
    if raw is None or len(raw) < len(ann):
        log.error("[silver] incomplete - re-run the script later to continue."); return None
    raw = raw.merge(ann[["unit_id", "text"]], on="unit_id")
    raw["silver"] = [{"entities": [{k: v for k, v in e.items() if k != "supported"}
                                   for e in entities_of(x) if is_supported(e, t)]}
                     for x, t in zip(raw.extraction, raw.text)]
    raw.drop(columns=["text"]).to_parquet(SILVER / "silver_clean.parquet")
    n_ent = raw.silver.map(lambda x: len(x["entities"])).mean()
    log.info(f"[silver] saved silver_clean.parquet | schema-valid {raw.valid_schema.mean():.3f} "
             f"| mean supported entities/chunk {n_ent:.1f}")
    return {u: s for u, s in zip(raw.unit_id, raw.silver)}


# =====================================================================
# STAGE 3: LABELS, ANNOTATOR EXPORT, SFT DATASET
# =====================================================================
def load_label_file(path):
    out = {}
    for line in open(path):
        try: d = json.loads(line)
        except json.JSONDecodeError: continue
        ents = []
        for e in d.get("entities", []):
            try: ents.append(Entity.model_validate(e).model_dump())
            except Exception:
                fe = _fix_entity(e)
                if fe: ents.append(Entity.model_validate(fe).model_dump())
        out[d["unit_id"]] = {"entities": ents}
    return out

def stage_labels(ann, silver):
    gold_path = GOLD / "gold_annotations.jsonl"
    gold = load_label_file(gold_path) if gold_path.exists() else {}
    labels = {}
    for r in ann.itertuples():
        if r.unit_id in gold: labels[r.unit_id] = (gold[r.unit_id], "gold")
        elif r.split in ("train", "val"): labels[r.unit_id] = (silver[r.unit_id], "silver")
    src = Counter((r.split, labels.get(r.unit_id, (None, "missing"))[1]) for r in ann.itertuples())
    log.info(f"[labels] gold units: {len(gold)} | sources: {dict(src)}")

    # annotator files: only created if missing (never overwrite files annotators may be using)
    pre = GOLD / "to_annotate_prefilled.jsonl"
    if not pre.exists() or ARGS.reexport:
        rng = np.random.default_rng(SEED)
        eval_ids = ann[ann.split.isin(["val", "test"])].unit_id.tolist()
        dbl = set(rng.choice(eval_ids, size=int(len(eval_ids) * DOUBLE_ANNOT_FRAC), replace=False))
        def export(path, prefill, only=None):
            with open(path, "w") as f:
                for r in ann.itertuples():
                    if only is not None and r.unit_id not in only: continue
                    f.write(json.dumps({"unit_id": r.unit_id, "split": r.split, "category": r.category,
                                        "text": r.text,
                                        "entities": silver[r.unit_id]["entities"] if prefill else []},
                                       ensure_ascii=False, default=list) + "\n")
        export(pre, True)
        export(GOLD / "double_annot_A_blank.jsonl", False, dbl)
        export(GOLD / "double_annot_B_blank.jsonl", False, dbl)
        log.info(f"[labels] annotator files written ({len(dbl)} double-annotation units)")
    else:
        log.info("[labels] annotator files already exist - not overwritten (use --reexport to force)")

    # SFT dataset (cheap, always rebuilt so it reflects the latest labels)
    for sp in ("train", "val"):
        with open(DATA / f"sft_{sp}.jsonl", "w") as f:
            for r in ann[ann.split == sp].itertuples():
                lab = labels.get(r.unit_id)
                if lab is None: continue
                f.write(json.dumps({"unit_id": r.unit_id, "prompt": build_messages(r.text),
                                    "completion": [{"role": "assistant",
                                                    "content": json.dumps(lab[0], ensure_ascii=False, default=list)}]},
                                   ensure_ascii=False) + "\n")
    log.info("[labels] SFT datasets written (data/sft_train.jsonl, data/sft_val.jsonl)")
    return gold, labels


# =====================================================================
# STAGE 4: LoRA TRAINING (checkpoints every few steps, auto-resume)
# =====================================================================
def stage_train():
    if DONE_MARK.exists():
        log.info(f"[train] SKIP - training already finished ({DONE_MARK.name} present)"); return True
    if not ARGS.train:
        log.info("[train] not requested (add --train to run LoRA training)"); return DONE_MARK.exists()

    mem = available_mem_gb()
    if mem is not None and mem < ARGS.min_free_gb and not ARGS.force:
        log.error(f"[train] only {mem:.0f} GB memory available (< {ARGS.min_free_gb} GB). "
                  "Free memory or use --force."); return False

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback, TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    class ThermalCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if state.global_step % 5 == 0: thermal_guard()
        def on_save(self, args, state, control, **kw):
            os.sync()                      # force checkpoint files onto disk (survives a power cut)

    train_ds = load_dataset("json", data_files=str(DATA / "sft_train.jsonl"), split="train")
    val_ds   = load_dataset("json", data_files=str(DATA / "sft_val.jsonl"), split="train")
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    # newest checkpoint that is NOT corrupted (a power cut can leave half-written files)
    def _valid(c):
        try:
            f = c / "adapter_model.safetensors"
            with open(f, "rb") as fh:                       # safetensors = 8-byte length + JSON header + data
                n = int.from_bytes(fh.read(8), "little")
                hdr = json.loads(fh.read(n))
            need = max((v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"), default=0)
            if f.stat().st_size < 8 + n + need: return False   # data truncated
            json.load(open(c / "trainer_state.json"))
            return True
        except Exception:
            return False
    last_ckpt = None
    for c in sorted([p for p in ADAPTER_DIR.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()],
                    key=lambda p: int(p.name.split("-")[-1]), reverse=True):
        if _valid(c): last_ckpt = str(c); break
        bad = ADAPTER_DIR / f"corrupt_{c.name}"
        c.rename(bad); log.warning(f"[train] {c.name} is corrupted (power cut?) - moved to {bad.name}")
    log.info(f"[train] {ADAPTER_DIR.name} | max_length {ARGS.max_length} | batch {ARGS.batch_size or 'auto'} | {len(train_ds)} train / {len(val_ds)} val examples | "
             f"{'RESUMING from ' + Path(last_ckpt).name if last_ckpt else 'starting fresh'}")

    tok = AutoTokenizer.from_pretrained(STUDENT_BASE)
    model = AutoModelForCausalLM.from_pretrained(STUDENT_BASE, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map="auto")
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    bs = ARGS.batch_size or (4 if ARGS.max_length <= 4096 else 2)
    accum, epochs = max(1, 16 // bs), 3                                     # effective batch stays 16
    total_steps = math.ceil(len(train_ds) / (bs * accum)) * epochs
    cfg = SFTConfig(
        output_dir=str(ADAPTER_DIR), num_train_epochs=epochs,
        per_device_train_batch_size=bs, per_device_eval_batch_size=bs, gradient_accumulation_steps=accum,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=max(1, int(0.03 * total_steps)),
        bf16=True, gradient_checkpointing=True, max_length=ARGS.max_length,
        logging_steps=5,
        eval_strategy="steps", eval_steps=ARGS.save_steps,
        save_strategy="steps", save_steps=ARGS.save_steps, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", seed=SEED,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tok, peft_config=lora_cfg,
                         callbacks=[EarlyStoppingCallback(early_stopping_patience=3), ThermalCallback()])
    torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
    out = trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(str(ADAPTER_DIR)); tok.save_pretrained(str(ADAPTER_DIR))

    stats = {"train_time_min_this_session": (time.perf_counter() - t0) / 60,
             "peak_gpu_mem_GB": torch.cuda.max_memory_allocated() / 1e9,
             "resumed_from": Path(last_ckpt).name if last_ckpt else None,
             "best_checkpoint": trainer.state.best_model_checkpoint, **out.metrics}
    json.dump(stats, open(RESULTS / "train_stats.json", "w"), indent=2, default=str)
    json.dump(trainer.state.log_history, open(RESULTS / "train_log_history.json", "w"), indent=1)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        h = pd.DataFrame(trainer.state.log_history); fig, ax = plt.subplots(figsize=(7, 4))
        if "loss" in h: h.dropna(subset=["loss"]).plot(x="step", y="loss", ax=ax, label="train")
        if "eval_loss" in h: h.dropna(subset=["eval_loss"]).plot(x="step", y="eval_loss", ax=ax, marker="o", label="val")
        ax.set_title("LoRA SFT loss"); fig.tight_layout(); fig.savefig(FIGS / "loss.png", dpi=200); plt.close(fig)
    except Exception as e:
        log.warning(f"[train] loss plot failed: {e}")
    DONE_MARK.write_text(json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), **stats}, default=str))
    log.info(f"[train] DONE in {stats['train_time_min_this_session']:.1f} min | peak mem {stats['peak_gpu_mem_GB']:.1f} GB")
    del trainer, model; torch.cuda.empty_cache()
    return True


# =====================================================================
# STAGE 5: PREDICTIONS ON THE TEST SET (resumable)
# =====================================================================
class HFExtractor:
    def __init__(self, adapter=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(STUDENT_BASE)
        m = AutoModelForCausalLM.from_pretrained(STUDENT_BASE, dtype=torch.bfloat16, device_map="auto")
        if adapter:
            m = PeftModel.from_pretrained(m, str(adapter)).merge_and_unload()
        self.model = m.eval()

    def __call__(self, text, max_new_tokens=None):
        max_new_tokens = max_new_tokens or ARGS.gen_tokens
        torch = self.torch
        with torch.inference_mode():
            enc = self.tok.apply_chat_template(build_messages(text), add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True).to(self.model.device)
            n_in = enc["input_ids"].shape[1]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
            torch.cuda.synchronize(); lat = time.perf_counter() - t0
        new = out[0, n_in:]
        raw = self.tok.decode(new, skip_special_tokens=True)
        ext, ok_json, ok_schema = parse_json(raw)
        return dict(raw=raw, extraction=ext.model_dump() if ext else None, valid_json=ok_json,
                    valid_schema=ok_schema, latency_s=lat, completion_tokens=int(len(new)),
                    finish_reason="length" if len(new) >= max_new_tokens else "stop")

def run_hf(name, adapter, test_units, out_path):
    from tqdm import tqdm
    done = load_jsonl_cache(out_path, only_success=False)
    todo = test_units[~test_units.unit_id.isin(done)]
    log.info(f"[{name}] {len(done)} done, {len(todo)} to go")
    if len(todo) == 0: return
    import torch
    ex = HFExtractor(adapter); torch.cuda.reset_peak_memory_stats()
    with open(out_path, "a") as f:
        for n, r in enumerate(tqdm(todo.itertuples(), total=len(todo), desc=name), 1):
            d = ex(r.text); d["unit_id"] = r.unit_id
            f.write(json.dumps(d) + "\n"); f.flush(); os.fsync(f.fileno())
            if n % 5 == 0: thermal_guard()
    peak = torch.cuda.max_memory_allocated() / 1e9
    mem_file = RESULTS / "peak_mem.json"
    mem = json.loads(mem_file.read_text()) if mem_file.exists() else {}
    mem[name] = round(peak, 2); mem_file.write_text(json.dumps(mem, indent=1))
    del ex; torch.cuda.empty_cache()

def stage_predict(test_units):
    # B2: reuse the silver run for test chunks (same model + prompt), then fill any gaps
    b2_path = PREDS / "B2_qwen36_zeroshot.jsonl"
    if not b2_path.exists():
        silver_cache = load_jsonl_cache(SILVER / "teacher_constrained.jsonl")
        with open(b2_path, "w") as f:
            for u in test_units.unit_id:
                if u in silver_cache: f.write(json.dumps(silver_cache[u]) + "\n")
        log.info("[B2] seeded from silver run")
    run_batch(test_units, b2_path, ARGS.workers)

    run_hf("B1_qwen2.5-3B_zeroshot", None, test_units, PREDS / "B1_qwen25_3b_zeroshot.jsonl")
    if DONE_MARK.exists():
        run_hf(FT_NAME, ADAPTER_DIR, test_units, PREDS / FT_FILE)
    else:
        log.info("[FT] skipped - LoRA training not finished yet")


# =====================================================================
# STAGE 6: EVALUATION (cheap; always regenerated)
# =====================================================================
TYPES = ["PROBLEM", "MEDICATION", "TEST", "RESULT"]
ATTRS = ["negated", "certainty", "temporality"]

def norm_text(s): return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", (s or "").lower())).strip()

def match_entities(gold_ents, pred_ents, mode="relaxed", thr=0.5):
    def toks(s): return set(norm_text(s).split())
    pairs, used = [], set()
    for gi, g in enumerate(gold_ents):
        best, best_j = None, -1
        for pj, p in enumerate(pred_ents):
            if pj in used or p["type"] != g["type"]: continue
            if mode == "strict": sc = 1.0 if norm_text(p["text"]) == norm_text(g["text"]) else 0.0
            else:
                a, b = toks(g["text"]), toks(p["text"]); sc = len(a & b) / max(len(a | b), 1)
            if sc >= (1.0 if mode == "strict" else thr) and (best is None or sc > best): best, best_j = sc, pj
        if best_j >= 0: used.add(best_j); pairs.append((gi, best_j))
    matched = {a for a, _ in pairs}
    return pairs, [i for i in range(len(gold_ents)) if i not in matched], [j for j in range(len(pred_ents)) if j not in used]

def unit_counts(ref, pred, mode):
    c = {t: [0, 0, 0] for t in TYPES}; ah = {a: [0, 0] for a in ATTRS}
    pairs, mg, mp = match_entities(ref, pred, mode)
    for i, j in pairs:
        c[ref[i]["type"]][0] += 1
        for a in ATTRS: ah[a][0] += int(ref[i][a] == pred[j].get(a)); ah[a][1] += 1
    for i in mg: c[ref[i]["type"]][2] += 1
    for j in mp:
        if pred[j]["type"] in c: c[pred[j]["type"]][1] += 1
    return c, ah

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

def aggregate(pu):
    tot = {t: np.sum([u[0][t] for u in pu], axis=0) for t in TYPES}
    TP, FP, FN = np.sum(list(tot.values()), axis=0); p, r, f = prf(TP, FP, FN)
    f1s = [prf(*tot[t])[2] for t in TYPES if tot[t].sum() > 0]
    attrs = {a: sum(u[1][a][0] for u in pu) / max(sum(u[1][a][1] for u in pu), 1) for a in ATTRS}
    return dict(P=p, R=r, micro_F1=f, macro_F1=float(np.mean(f1s)) if f1s else 0.0,
                **{f"acc_{a}": v for a, v in attrs.items()}), tot

def bootstrap_ci(pu, B, seed=SEED):
    rng = np.random.default_rng(seed); n = len(pu)
    vals = [aggregate([pu[i] for i in rng.integers(0, n, n)])[0]["micro_F1"] for _ in range(B)]
    return np.percentile(vals, [2.5, 97.5])

def paired_bootstrap(pa, pb, B, seed=SEED):
    rng = np.random.default_rng(seed); n = len(pa); wins = 0; deltas = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        fa = aggregate([pa[i] for i in idx])[0]["micro_F1"]; fb = aggregate([pb[i] for i in idx])[0]["micro_F1"]
        deltas.append(fa - fb); wins += fa > fb
    return dict(delta_mean=round(float(np.mean(deltas)), 4),
                CI95=[round(float(x), 4) for x in np.percentile(deltas, [2.5, 97.5])], p_value=round(1 - wins / B, 4))

def load_preds(path, test_units):
    cache = load_jsonl_cache(path, only_success=False)
    if not all(u in cache for u in test_units.unit_id): return None
    rows = []
    for u in test_units.unit_id:
        d = dict(cache[u])
        if d.get("raw") is not None:                          # Fix C: forgiving re-parse
            ext, ok_json, ok_schema = parse_json(d["raw"])
            d.update(extraction=ext.model_dump() if ext else None, valid_json=ok_json, valid_schema=ok_schema)
        rows.append(d)
    return pd.DataFrame(rows)

def rule_baseline(ann, labels, test_units):
    NEG  = r"\b(no|not|denies|denied|without|negative for|absence of|free of|ruled out|r/o'd|resolved)\b"
    UNC  = r"\b(possible|possibly|probable|likely|suspected|suspect|concern for|cannot rule out|r/o|rule out|questionable|\?|may have|consistent with)\b"
    HYP  = r"\b(if|should|return if|in case|prn)\b"
    HIST = r"\b(h/o|history of|hx of|s/p|status post|prior|previous|past|pmh)\b"
    FUT  = r"\b(will|plan to|planned|to start|to be|consider|schedule[d]?)\b"
    lc = defaultdict(Counter)
    for r in ann[ann.split == "train"].itertuples():
        for e in labels.get(r.unit_id, ({"entities": []},))[0]["entities"]:
            t = norm_text(e["text"])
            if 2 < len(t) < 60: lc[t][e["type"]] += 1
    LEX = {t: c.most_common(1)[0][0] for t, c in lc.items() if sum(c.values()) >= 2}
    LEX_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, LEX), key=len, reverse=True)) + r")\b", re.I) if LEX else None
    rows = []
    for u, text in zip(test_units.unit_id, test_units.text):
        t0 = time.perf_counter(); ents, seen = [], set()
        for sent in re.split(r"(?<=[.;\n])\s+", text):
            if not LEX_RE: break
            for m in LEX_RE.finditer(sent):
                key = norm_text(m.group(0)); typ = LEX.get(key)
                if not typ or key in seen: continue
                seen.add(key); pre = sent[max(0, m.start() - 60):m.start()].lower()
                ents.append(dict(text=m.group(0), type=typ, evidence=sent.strip()[:200],
                                 negated=bool(re.search(NEG, pre)),
                                 certainty="hypothetical" if re.search(HYP, pre) else "uncertain" if re.search(UNC, pre) else "certain",
                                 temporality="historical" if re.search(HIST, pre) else "future" if re.search(FUT, pre) else "current",
                                 related_test=None, value=None))
        rows.append(dict(unit_id=u, extraction={"entities": ents}, valid_json=True, valid_schema=True,
                         latency_s=time.perf_counter() - t0))
    log.info(f"[eval] rule lexicon size: {len(LEX):,}")
    return pd.DataFrame(rows)

def stage_eval(ann, gold, labels, silver):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    test_units = ann[(ann.split == "test") & (ann.unit_id.isin(gold))].copy()
    if len(test_units) == 0:
        log.warning("[eval] NO GOLD TEST LABELS YET - scoring against SILVER labels (debug only, do NOT report)")
        test_units = ann[ann.split == "test"].copy(); REF = {u: silver[u] for u in test_units.unit_id}
        ref_kind = "silver_DEBUG_ONLY"
    else:
        REF = {u: gold[u] for u in test_units.unit_id}; ref_kind = "gold"
    log.info(f"[eval] {len(test_units)} test units | reference = {ref_kind}")
    text_of = dict(zip(test_units.unit_id, test_units.text))
    mem = json.loads((RESULTS / "peak_mem.json").read_text()) if (RESULTS / "peak_mem.json").exists() else {}

    preds = {"B0_rules": rule_baseline(ann, labels, test_units)}
    systems = [("B2_qwen3.6-35B_zeroshot", "B2_qwen36_zeroshot.jsonl"),
               ("B1_qwen2.5-3B_zeroshot", "B1_qwen25_3b_zeroshot.jsonl")]
    for p in sorted(PREDS.glob("FT_qwen25_3b_lora*.jsonl")):          # every FT version (v1, v2, ...)
        sfx = p.stem[len("FT_qwen25_3b_lora"):]
        systems.append((f"FT_qwen2.5-3B_LoRA{sfx}", p.name))
    for name, fn in systems:
        df = load_preds(PREDS / fn, test_units)
        if df is None: log.info(f"[eval] {name}: predictions incomplete/missing - left out")
        else: preds[name] = df

    results, per_type, pu_store = [], [], {}
    for name, df in preds.items():
        for mode in ("relaxed", "strict"):
            pu, n_pred, n_sup = [], 0, 0
            for r in df.itertuples():
                pred = entities_of(r.extraction)
                for e in pred: n_pred += 1; n_sup += is_supported(e, text_of[r.unit_id])
                pu.append(unit_counts(REF[r.unit_id]["entities"], pred, mode))
            m, tot = aggregate(pu); lo, hi = bootstrap_ci(pu, ARGS.bootstrap)
            lat = pd.to_numeric(df.latency_s, errors="coerce").dropna()
            results.append(dict(system=name, match=mode, reference=ref_kind, **{k: round(float(v), 3) for k, v in m.items()},
                                micro_F1_CI_low=round(lo, 3), micro_F1_CI_high=round(hi, 3),
                                json_valid=round(df.valid_json.mean(), 3), schema_valid=round(df.valid_schema.mean(), 3),
                                unsupported_rate=round(1 - n_sup / max(n_pred, 1), 3), n_pred=n_pred,
                                latency_mean_s=round(lat.mean(), 3) if len(lat) else None,
                                latency_p95_s=round(lat.quantile(0.95), 3) if len(lat) else None,
                                peak_gpu_mem_GB=0.0 if name == "B0_rules" else mem.get(name)))
            if mode == "relaxed":
                pu_store[name] = pu
                for t in TYPES:
                    p, rr, f = prf(*tot[t])
                    per_type.append(dict(system=name, type=t, P=round(p, 3), R=round(rr, 3), F1=round(f, 3),
                                         support=int(tot[t][0] + tot[t][2])))
    res = pd.DataFrame(results); res.to_csv(RESULTS / "main_results.csv", index=False)
    pd.DataFrame(per_type).to_csv(RESULTS / "per_type_results.csv", index=False)

    # by note category
    cat_of = dict(zip(test_units.unit_id, test_units.category)); rows = []
    for name, df in preds.items():
        for cat, g in df.assign(category=df.unit_id.map(cat_of)).groupby("category"):
            pu = [unit_counts(REF[r.unit_id]["entities"], entities_of(r.extraction), "relaxed") for r in g.itertuples()]
            rows.append(dict(system=name, category=cat, n=len(g), micro_F1=round(aggregate(pu)[0]["micro_F1"], 3)))
    pd.DataFrame(rows).to_csv(RESULTS / "by_category.csv", index=False)

    # paired significance
    sig = {}
    ft_names = [n for n in pu_store if n.startswith("FT_")]
    for ft in ft_names:
        for base in ["B1_qwen2.5-3B_zeroshot", "B0_rules", "B2_qwen3.6-35B_zeroshot"] + [o for o in ft_names if o < ft]:
            if base in pu_store:
                sig[f"{ft}_vs_{base}"] = paired_bootstrap(pu_store[ft], pu_store[base], ARGS.bootstrap * 2)
    json.dump(sig, open(RESULTS / "significance.json", "w"), indent=2)

    # error analysis (saved to CSV only - contains entity text, keep local)
    focus = FT_NAME if FT_NAME in preds else next((n for n in reversed(list(preds)) if n.startswith("FT_")),
            "B2_qwen3.6-35B_zeroshot" if "B2_qwen3.6-35B_zeroshot" in preds else "B0_rules")
    err = []
    for r in preds[focus].itertuples():
        ref, pred = REF[r.unit_id]["entities"], entities_of(r.extraction)
        pairs, mg, mp = match_entities(ref, pred)
        for i in mg: err.append(dict(unit_id=r.unit_id, error="FN", type=ref[i]["type"], text=ref[i]["text"], detail=""))
        for j in mp:
            conf = [g["type"] for g in ref if norm_text(g["text"]) == norm_text(pred[j]["text"])]
            err.append(dict(unit_id=r.unit_id, error="TYPE_CONFUSION" if conf else "FP", type=pred[j]["type"],
                            text=pred[j]["text"], detail=f"gold={conf[0]}" if conf else ""))
        for i, j in pairs:
            for a in ATTRS:
                if ref[i][a] != pred[j].get(a):
                    err.append(dict(unit_id=r.unit_id, error=f"ATTR_{a}", type=ref[i]["type"], text=ref[i]["text"],
                                    detail=f"gold={ref[i][a]} pred={pred[j].get(a)}"))
    errors = pd.DataFrame(err, columns=["unit_id", "error", "type", "text", "detail"])
    errors.to_csv(RESULTS / f"errors_{focus}.csv", index=False)
    errors.groupby(["error", "type"]).size().unstack(fill_value=0).to_csv(RESULTS / f"error_summary_{focus}.csv")

    # figure
    rel = res[res.match == "relaxed"].set_index("system")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(rel.index, rel.micro_F1, xerr=[rel.micro_F1 - rel.micro_F1_CI_low, rel.micro_F1_CI_high - rel.micro_F1],
            color="#4C72B0", capsize=4)
    ax.set_xlim(0, 1); ax.set_xlabel("Micro-F1 (relaxed) with 95% bootstrap CI")
    ax.set_title(f"Clinical IE - held-out test patients (reference: {ref_kind})")
    fig.tight_layout(); fig.savefig(FIGS / "micro_f1.png", dpi=200); plt.close(fig)

    # reproducibility
    repro = dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), platform=platform.platform(),
                 python=platform.python_version(), teacher_model=MODEL, student_base=STUDENT_BASE, seed=SEED,
                 reference=ref_kind, n_test_units=len(test_units), gold_test_units=int(sum(u in gold for u in test_units.unit_id)),
                 test_unit_hash=hashlib.sha1(",".join(sorted(test_units.unit_id)).encode()).hexdigest(),
                 split_patients_in_sample={sp: int(ann[ann.split == sp].subject_id.nunique()) for sp in SPLIT_FRAC})
    for mod in ("torch", "transformers", "peft", "trl", "duckdb", "sklearn", "vllm"):
        try: repro[mod] = __import__(mod).__version__
        except Exception: pass
    json.dump(repro, open(RESULTS / "reproducibility.json", "w"), indent=2)

    # console summary: metrics only, no note text
    cols = ["system", "P", "R", "micro_F1", "micro_F1_CI_low", "micro_F1_CI_high", "acc_negated",
            "acc_certainty", "acc_temporality", "schema_valid", "unsupported_rate"]
    log.info("[eval] RESULTS (relaxed match):\n" + res[res.match == "relaxed"][cols].to_string(index=False))
    if sig: log.info("[eval] significance: " + json.dumps(sig))
    log.info(f"[eval] all outputs written to {RESULTS} and {FIGS}")


# =====================================================================
# STATUS + MAIN
# =====================================================================
def count_lines(p):
    return len(load_jsonl_cache(p, only_success=False)) if Path(p).exists() else 0

def show_status():
    def row(label, ok, extra=""): print(f"  {'✅' if ok else '⬜'} {label:42s} {extra}")
    print("\nCheckpoint status -", OUT_DIR)
    ann_p = DATA / "annotation_units.parquet"; n_ann = len(pd.read_parquet(ann_p)) if ann_p.exists() else 0
    row("prep: annotation sample", ann_p.exists(), f"{n_ann} chunks")
    s = count_lines(SILVER / "teacher_constrained.jsonl"); row("silver: Qwen3.6 extractions", n_ann and s >= n_ann, f"{s}/{n_ann}")
    row("silver: cleaned labels", (SILVER / "silver_clean.parquet").exists())
    row("labels: SFT train/val files", (DATA / "sft_train.jsonl").exists())
    row("gold: annotator corrections", (GOLD / "gold_annotations.jsonl").exists())
    ck = sorted(ADAPTER_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])) if ADAPTER_DIR.exists() else []
    row(f"train: LoRA finished ({ADAPTER_DIR.name})", DONE_MARK.exists(), f"latest checkpoint: {ck[-1].name if ck else 'none'}")
    n_test = int((pd.read_parquet(ann_p).split == "test").sum()) if ann_p.exists() else 0
    for fn in ("B2_qwen36_zeroshot", "B1_qwen25_3b_zeroshot", FT_FILE[:-6]):
        c = count_lines(PREDS / f"{fn}.jsonl"); row(f"predict: {fn}", n_test and c >= n_test, f"{c}/{n_test}")
    row("eval: main_results.csv", (RESULTS / "main_results.csv").exists())
    m = available_mem_gb(); t = gpu_temp()
    print(f"\n  memory available: {m:.0f} GB" if m else "", f"| GPU temp: {t:.0f} C" if t else "")

def main():
    global ARGS
    ap = argparse.ArgumentParser(description="Resumable MIMIC-III LoRA clinical IE pipeline")
    ap.add_argument("--stage", default="all", choices=["all", "prep", "silver", "labels", "train", "predict", "eval"])
    ap.add_argument("--train", action="store_true", help="include LoRA training")
    ap.add_argument("--status", action="store_true", help="show checkpoint status and exit")
    ap.add_argument("--workers", type=int, default=6, help="parallel vLLM requests (default 6, gentler on the Spark)")
    ap.add_argument("--vllm-wait", type=int, default=20, help="minutes to wait for vLLM after a reboot")
    ap.add_argument("--max-temp", type=int, default=87, help="pause if GPU hotter than this (C); 0 = off")
    ap.add_argument("--min-free-gb", type=int, default=30, help="minimum free memory before training")
    ap.add_argument("--save-steps", type=int, default=10, help="LoRA checkpoint every N optimizer steps")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for CIs")
    ap.add_argument("--reexport", action="store_true", help="overwrite annotator template files")
    ap.add_argument("--force", action="store_true", help="ignore the free-memory check")
    ap.add_argument("--tag", default="", help="version tag, e.g. v2 -> separate adapter + FT predictions")
    ap.add_argument("--max-length", type=int, default=4096, help="max training sequence length (tokens)")
    ap.add_argument("--batch-size", type=int, default=0, help="per-step batch size (0 = auto); lower = less memory")
    ap.add_argument("--gen-tokens", type=int, default=4096, help="max tokens the small models may generate")
    ARGS = ap.parse_args()
    set_run_tag(ARGS.tag)
    setup_logging()
    if ARGS.status: show_status(); return

    with RunLock(OUT_DIR / "pipeline.lock"):
        st = ARGS.stage
        log.info(f"===== pipeline start | stage={st} | train={ARGS.train} | tag={ARGS.tag or 'v1'} "
                 f"| max_length={ARGS.max_length} | gen_tokens={ARGS.gen_tokens} =====")
        ann = stage_prep()
        if st == "prep": return

        silver = stage_silver(ann)
        if silver is None: log.error("Stopping: silver labels incomplete."); return
        if st == "silver": return

        gold, labels = stage_labels(ann, silver)
        if st == "labels": return

        if st in ("all", "train"):
            if ARGS.stage == "train": ARGS.train = True
            stage_train()
            if st == "train": return

        if st in ("all", "predict"):
            test_units = ann[ann.split == "test"]
            stage_predict(test_units)
            if st == "predict": return

        stage_eval(ann, gold, labels, silver)
        log.info("===== pipeline finished =====")

if __name__ == "__main__":
    main()
