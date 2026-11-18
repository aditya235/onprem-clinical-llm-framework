#!/usr/bin/env python3
"""
Annotation <-> Excel converter for the MIMIC-III clinical IE study.
Runs ON THE SPARK (the files contain note text and must not leave institutional systems).

EXPORT  (JSONL -> Excel workbooks for annotators)
    python annot_excel.py export
    -> ~/mimic_project/ie_pipeline/gold/excel/
         1_TEST_to_correct.xlsx, 2_VAL_to_correct.xlsx,
         double_annot_A_blank.xlsx, double_annot_B_blank.xlsx

IMPORT  (returned Excel -> JSONL the pipeline reads), with validation report
    python annot_excel.py import gold/excel/1_TEST_to_correct_AS.xlsx gold/excel/2_VAL_to_correct_AS.xlsx --out gold_annotations.jsonl
    python annot_excel.py import gold/excel/double_annot_A_blank_XX.xlsx --out double_annot_A.jsonl
    python annot_excel.py import gold/excel/double_annot_B_blank_YY.xlsx --out double_annot_B.jsonl
    (--out is written into the gold/ folder; the report never prints note text)
"""
import argparse, json, os, re, sys, textwrap, time
from pathlib import Path

try:
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.worksheet.hyperlink import Hyperlink
except ImportError:
    sys.exit("openpyxl missing. Install it with:  ~/jupyterlab/.venv/bin/pip install openpyxl")

GOLD = Path(os.path.expanduser(os.environ.get("IE_OUT_DIR", "~/mimic_project/ie_pipeline"))) / "gold"
XL = GOLD / "excel"

TYPES = ["PROBLEM", "MEDICATION", "TEST", "RESULT"]
ACTIONS = ["KEEP", "FIX", "DELETE", "ADD"]
NEG = ["TRUE", "FALSE"]
CERT = ["certain", "uncertain", "hypothetical"]
TEMP = ["current", "historical", "future"]
COLS = ["unit_id", "open_note", "ACTION", "text", "type", "evidence", "negated",
        "certainty", "temporality", "related_test", "value", "annotator_comment"]
WIDTH = dict(unit_id=13, open_note=11, ACTION=10, text=28, type=13, evidence=45, negated=9,
             certainty=13, temporality=13, related_test=14, value=12, annotator_comment=30)

EXPORTS = [  # source jsonl, workbook name, prefilled?, empty rows per chunk
    ("1_TEST_to_correct.jsonl",     "1_TEST_to_correct.xlsx",     True,  10),
    ("2_VAL_to_correct.jsonl",      "2_VAL_to_correct.xlsx",      True,  10),
    ("double_annot_A_blank.jsonl",  "double_annot_A_blank.xlsx",  False, 45),
    ("double_annot_B_blank.jsonl",  "double_annot_B_blank.xlsx",  False, 45),
]

INSTRUCTIONS = [
    "MIMIC-III clinical information extraction - annotation workbook",
    "",
    "Read the Annotation Guideline document before starting.",
    "Work only on institutional systems. Do not copy this file elsewhere (PhysioNet DUA).",
    "",
    "HOW TO WORK",
    "1. Go to the 'Entities' sheet. Rows are grouped by unit_id (one chunk of a note).",
    "2. Click 'Open note' to read that chunk in the 'Notes' sheet (Ctrl+[ or the back arrow to return).",
    "3. For each row choose ACTION:  KEEP = correct,  FIX = you edited it,  DELETE = wrong / not in note,  ADD = new entity.",
    "4. Add missing entities in the empty rows of the same unit_id (ACTION = ADD).",
    "5. evidence = exact phrase copied from the note (no paraphrase, no '...').",
    "6. Use the dropdowns for ACTION, type, negated, certainty, temporality.",
    "7. RESULT rows: also fill related_test (e.g. Hct) and value (e.g. 23.4).",
    "",
    "DO NOT rename sheets, delete columns, change headers, or change unit_id values.",
    "Save with your initials added to the file name, e.g. 1_TEST_to_correct_AS.xlsx",
]


def read_jsonl(p):
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


# ------------------------------------------------------------------ EXPORT
def export_one(src, dst, prefilled, n_empty):
    recs = read_jsonl(src)
    wb = Workbook()
    ws_i = wb.active; ws_i.title = "Instructions"
    for i, line in enumerate(INSTRUCTIONS, 1):
        c = ws_i.cell(row=i, column=1, value=line)
        if i == 1 or line in ("HOW TO WORK",): c.font = Font(bold=True, size=12 if i == 1 else 11)
    ws_i.column_dimensions["A"].width = 120

    # ---- Notes: one row per (wrapped) line, unit_id on every row
    ws_n = wb.create_sheet("Notes")
    ws_n.append(["unit_id", "split", "category", "note_text"])
    for c in ws_n[1]: c.font = Font(bold=True)
    ws_n.column_dimensions["A"].width = 13; ws_n.column_dimensions["B"].width = 7
    ws_n.column_dimensions["C"].width = 17; ws_n.column_dimensions["D"].width = 160
    ws_n.freeze_panes = "A2"
    first_row, shade = {}, PatternFill("solid", fgColor="EEF3F8")
    for k, r in enumerate(recs):
        first_row[r["unit_id"]] = ws_n.max_row + 1
        lines = []
        for para in str(r["text"]).split("\n"):
            lines += textwrap.wrap(para, 150) or [""]
        for j, ln in enumerate(lines):
            ws_n.append([r["unit_id"], r.get("split", "") if j == 0 else "", r.get("category", "") if j == 0 else "", ln])
            if k % 2:
                for c in ws_n[ws_n.max_row]: c.fill = shade
        ws_n.append([])                                   # spacer between chunks

    # ---- Entities
    ws = wb.create_sheet("Entities")
    ws.append(COLS)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="305496")
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for i, col in enumerate(COLS, 1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = WIDTH[col]
    ws.freeze_panes = "D2"
    for k, r in enumerate(recs):
        u = r["unit_id"]
        rows = []
        if prefilled:
            for e in r.get("entities", []):
                rows.append(["KEEP", e.get("text"), e.get("type"), e.get("evidence"),
                             "TRUE" if e.get("negated") else "FALSE", e.get("certainty", "certain"),
                             e.get("temporality", "current"), e.get("related_test"), e.get("value"), None])
        rows += [[None] * 10 for _ in range(n_empty)]
        for vals in rows:
            ws.append([u, "Open note"] + vals)
            row = ws.max_row
            link = ws.cell(row=row, column=2)
            link.hyperlink = Hyperlink(ref=link.coordinate, location=f"'Notes'!A{first_row[u]}", tooltip="Open this chunk's note"); link.font = Font(color="0563C1", underline="single")
            if k % 2:
                for c in ws[row]: c.fill = shade
            for col in (4, 6, 12):
                ws.cell(row=row, column=col).alignment = Alignment(wrap_text=True, vertical="top")
    last = ws.max_row + 2000
    for col, opts in [("C", ACTIONS), ("E", TYPES), ("G", NEG), ("H", CERT), ("I", TEMP)]:
        dv = DataValidation(type="list", formula1='"' + ",".join(opts) + '"', allow_blank=True,
                            showErrorMessage=True, errorTitle="Invalid value",
                            error="Please choose a value from the dropdown list.")
        ws.add_data_validation(dv); dv.add(f"{col}2:{col}{last}")
    ws.auto_filter.ref = f"A1:L{ws.max_row}"

    meta = wb.create_sheet("meta"); meta.sheet_state = "hidden"
    meta.append(["source", src.name]); meta.append(["created", time.strftime("%Y-%m-%d %H:%M")])
    meta.append(["prefilled", str(prefilled)])
    wb.active = wb.sheetnames.index("Entities")
    wb.save(dst)
    n_ent = sum(len(r.get("entities", [])) for r in recs) if prefilled else 0
    print(f"  {dst.name:32s} {len(recs):4d} chunks | {n_ent:5d} draft entities | {n_empty} empty rows per chunk")


def cmd_export(args):
    XL.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to {XL}")
    for src, dst, pre, n_empty in EXPORTS:
        s = GOLD / src
        if not s.exists(): print(f"  skip {src} (not found)"); continue
        d = XL / dst
        if d.exists() and not args.force:
            print(f"  skip {dst} (exists - use --force to overwrite; annotators may be using it)"); continue
        export_one(s, d, pre, n_empty)


# ------------------------------------------------------------------ IMPORT
def note_texts():
    """unit_id -> note text, from the master file (for the evidence check)."""
    out = {}
    for f in ["to_annotate_prefilled.jsonl", "double_annot_A_blank.jsonl"]:
        p = GOLD / f
        if p.exists():
            for r in read_jsonl(p): out[r["unit_id"]] = r["text"]
    return out


def as_bool(v):
    if isinstance(v, bool): return v
    s = str(v).strip().upper()
    if s in ("TRUE", "T", "YES", "Y", "1"): return True
    if s in ("FALSE", "F", "NO", "N", "0", "", "NONE"): return False
    return None


def cmd_import(args):
    texts = note_texts()
    units, issues, stats = {}, [], dict(rows=0, deleted=0, kept=0, empty=0)
    for xf in args.files:
        xf = Path(xf).expanduser()
        wb = load_workbook(xf, data_only=True)
        if "Entities" not in wb.sheetnames:
            sys.exit(f"{xf.name}: no 'Entities' sheet - was it renamed?")
        ws = wb["Entities"]
        header = [str(c.value).strip() if c.value is not None else "" for c in ws[1]]
        missing = [c for c in COLS if c not in header]
        if missing: sys.exit(f"{xf.name}: missing columns {missing} - headers must not be changed")
        idx = {c: header.index(c) for c in COLS}
        # every chunk in the workbook is kept, even with zero entities
        if "Notes" in wb.sheetnames:
            for (u,) in wb["Notes"].iter_rows(min_row=2, max_col=1, values_only=True):
                if u: units.setdefault(str(u).strip(), [])
        for rn, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
            g = lambda c: row[idx[c]] if idx[c] < len(row) else None
            u = str(g("unit_id") or "").strip()
            action = str(g("ACTION") or "").strip().upper()
            text, typ = str(g("text") or "").strip(), str(g("type") or "").strip().upper()
            if not u and not text: continue
            stats["rows"] += 1
            if action == "DELETE": stats["deleted"] += 1; continue
            if not text and not typ: stats["empty"] += 1; continue
            loc = f"{xf.name} row {rn}"
            if not u: issues.append((loc, "missing unit_id")); continue
            if texts and u not in texts: issues.append((loc, "unknown unit_id (changed?)")); continue
            if not text: issues.append((loc, "type set but text empty")); continue
            if typ not in TYPES: issues.append((loc, f"invalid type '{typ[:15]}'")); continue
            neg = as_bool(g("negated"))
            if neg is None: issues.append((loc, "negated must be TRUE/FALSE - set to FALSE")); neg = False
            cert = str(g("certainty") or "certain").strip().lower()
            if cert not in CERT: issues.append((loc, f"invalid certainty - set to certain")); cert = "certain"
            temp = str(g("temporality") or "current").strip().lower()
            if temp not in TEMP: issues.append((loc, f"invalid temporality - set to current")); temp = "current"
            ev = str(g("evidence") or "").strip()
            if not ev: issues.append((loc, "evidence empty - text used as evidence")); ev = text
            elif u in texts and norm(ev) not in norm(texts[u]):
                issues.append((loc, "evidence not found exactly in note (check copy)"))
            ent = dict(text=text, type=typ, evidence=ev, negated=neg, certainty=cert, temporality=temp,
                       related_test=(str(g("related_test")).strip() if g("related_test") else None),
                       value=(str(g("value")).strip() if g("value") is not None and str(g("value")).strip() else None))
            if typ == "RESULT" and not ent["value"]:
                issues.append((loc, "RESULT without value"))
            units.setdefault(u, []).append(ent); stats["kept"] += 1

    out = GOLD / args.out
    if out.exists() and not args.force:
        sys.exit(f"{out.name} already exists - use --force to overwrite")
    with open(out, "w", encoding="utf-8") as f:
        for u, ents in units.items():
            f.write(json.dumps({"unit_id": u, "entities": ents}, ensure_ascii=False) + "\n")

    rep = GOLD / (Path(args.out).stem + "_import_report.txt")
    with open(rep, "w") as f:
        f.write(f"Imported: {', '.join(Path(x).name for x in args.files)}\n")
        f.write(f"chunks: {len(units)} | entities kept: {stats['kept']} | deleted: {stats['deleted']} "
                f"| empty rows skipped: {stats['empty']}\n")
        f.write(f"chunks with zero entities: {sum(1 for v in units.values() if not v)}\n\n")
        f.write(f"ISSUES ({len(issues)}):\n")
        for loc, msg in issues: f.write(f"  {loc}: {msg}\n")
    print(open(rep).read() if len(issues) <= 60 else
          open(rep).read().split("ISSUES")[0] + f"ISSUES: {len(issues)} (see {rep})")
    print(f"Written: {out}\nReport : {rep}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--force", action="store_true")
    i = sub.add_parser("import"); i.add_argument("files", nargs="+")
    i.add_argument("--out", required=True, help="e.g. gold_annotations.jsonl"); i.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cmd_export(a) if a.cmd == "export" else cmd_import(a)


if __name__ == "__main__":
    main()
