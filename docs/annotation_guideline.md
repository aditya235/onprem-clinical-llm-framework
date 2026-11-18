# Annotation guideline: clinical information extraction

Your labels are the "answer key" every model is scored against, so accuracy and consistency matter more than speed. Each note excerpt ("chunk", up to about 3,000 characters) gets every clinical entity, its type and three attributes.

| Task | What you do |
| --- | --- |
| Correct files (`1_TEST_to_correct`, `2_VAL_to_correct`) | **Correct** the AI draft: keep, fix, delete, add |
| Blank files (`double_annot_A_blank`, `double_annot_B_blank`) | **Annotate from scratch**; annotators A and B work independently on the same chunks |

## Data-use rules (confirm before starting)

- [ ] I have my own credentialed access to the dataset and have signed its data-use agreement (for a provider's own records: I am authorised under institutional confidentiality).
- [ ] I open and edit files only on institutional systems.
- [ ] I do not copy files to personal devices, email, messaging apps, cloud drives or online AI tools.
- [ ] I do not try to identify any patient.
- [ ] I delete local working copies when finished.

## Entity types

| Type | Includes | Examples | Do not mark |
| --- | --- | --- | --- |
| **PROBLEM** | Diagnoses, diseases, symptoms, signs, abnormal findings, complications | pneumonia, chest pain, hypotension, AKI | Normal findings stated as normal, unless negated or relevant |
| **MEDICATION** | Drugs, drug classes given or considered, IV fluids, blood products, oxygen | metoprolol, heparin gtt, NS 1L, 2 units PRBC | Allergies listed only as history (mark as PROBLEM) |
| **TEST** | Lab tests, imaging, diagnostic procedures, vital-sign measurements | CBC, CXR, CT abdomen, echo, BP, HR | Treatments and surgeries (mark the indication as PROBLEM) |
| **RESULT** | A reported value of a test | WBC 14.2, Hct 23.4, EF 35% | A test name without a value (that is TEST) |

RESULT rows: whole phrase in `text`, test name in `related_test`, value with units in `value` (e.g. "Hct 23.4" / "Hct" / "23.4"). Keep abbreviations as written.

## Attributes

| Attribute | Value | Use when | Cue words |
| --- | --- | --- | --- |
| negated | TRUE | Explicitly absent, denied or ruled out | no, not, denies, negative for, without, ruled out, resolved |
| negated | FALSE | Default: present or mentioned positively | |
| certainty | certain | Default: stated as fact | |
| certainty | uncertain | Possible, suspected or being considered | possible, probable, likely, suspected, ?, r/o, concern for, cannot exclude |
| certainty | hypothetical | Conditional or instructional | if, should, return if, in case of, prn |
| temporality | current | Default: this admission or now | |
| temporality | historical | Past history or previous episode | h/o, history of, s/p, prior, PMH section |
| temporality | future | Planned or to be started | will, plan to, to start, scheduled |

The attributes are independent: "No h/o MI" = MI, negated TRUE, certain, historical.

## Evidence and general rules

1. **Evidence = exact copy** of the shortest continuous phrase (about 12 words or fewer) showing the entity. No paraphrase, no "...".
2. Mark only what is written; do not infer diagnoses from results.
3. One row per entity per chunk, unless the attributes differ.
4. Ignore privacy tags such as [DATE], [NAME].
5. Lists: "PMH: DM2, HTN, CKD" = three historical PROBLEM rows.
6. Family history is not the patient's problem: do not mark it.
7. Unsure? Make your best choice and explain in `annotator_comment`.

| Situation | Decision |
| --- | --- |
| "no evidence of PE on CTA" | PROBLEM "PE" negated; TEST "CTA" |
| "?PNA" | PROBLEM "PNA" uncertain |
| "d/c metoprolol" | MEDICATION "metoprolol" current |
| "if SBP < 90 give fluids" | MEDICATION "fluids" hypothetical |

## Using the Excel workbook

Sheets: **Instructions**, **Notes** (chunk text) and **Entities** (one row per entity). Click **Open note** to read a chunk, then review all rows with its `unit_id`.

| Column | What to enter |
| --- | --- |
| unit_id | Pre-filled; never change |
| ACTION | KEEP, FIX, DELETE or ADD |
| text, type, evidence | As above |
| negated, certainty, temporality | Dropdowns |
| related_test, value | RESULT rows only |
| annotator_comment | Optional |

Add entities in the empty rows of the same chunk (ACTION = ADD). Do not rename sheets, delete columns or change headers. Save with your initials in the file name.

## Workflow

1. Read this guideline and confirm the data-use checklist.
2. Practice round: annotate 5 chunks and get feedback.
3. Complete your file (about 5–10 minutes per chunk).
4. Double annotation: work alone; do not discuss chunks or look at AI drafts.
5. Return the workbook to the shared institutional folder.
