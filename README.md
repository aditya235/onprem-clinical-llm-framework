# On-Premise Clinical LLM Framework

Code and templates accompanying the paper **"From Governance to Deployment: A Framework for On-Premise Fine-Tuning of Clinical Language Models in Healthcare Organisations"** (in preparation for *Health Information Science and Systems*).

The repository shows how a healthcare provider can adapt an open-weight large language model (LLM) to its own clinical notes **without any patient data leaving its premises**: a large local "teacher" model annotates notes, a small local "student" model is fine-tuned with LoRA, and every step is governed, checkpointed and auditable.

> **No data, labels, model outputs or trained adapters are included.** MIMIC-III and the n2c2 corpora may only be obtained by credentialed users under their own data-use agreements. See [DATA_USE.md](DATA_USE.md).

## The framework

| Stage | Purpose |
| --- | --- |
| 0 Governance set-up | Oversight committee, roles, use case and risk class |
| 1 Data governance | Legal basis (e.g. India's DPDP Act), data-use agreements, de-identification, no transfer off premises |
| 2 Platform and model | On-premise compute sizing; open-weight model chosen by licence, size, language and security |
| 3 Ground truth | Annotation guideline, trained annotators, teacher pre-annotation, double annotation |
| 4 Model adaptation | LoRA fine-tuning, versioned adapters, checkpoints, reproducibility record |
| 5 Validation and assurance | Accuracy, faithfulness to the source note, output validity, acceptance thresholds (go / no-go) |
| 6 Deployment and operations | Capacity and resilience planning, human review, incident handling |
| 7 Monitoring and retirement | Drift monitoring, revalidation, audit trail, retirement |

## Demonstration task

Structured extraction of **problems, medications, tests and results**, each with **negation, certainty and temporality**, from MIMIC-III ICU notes into a fixed JSON schema, on a single NVIDIA DGX Spark (128 GB unified memory).

| Role | Model |
| --- | --- |
| Teacher (served locally with vLLM) | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| Student (LoRA fine-tuned) | `Qwen/Qwen2.5-3B-Instruct` (check its licence for your use) |
| Baselines | Rule-based lexicon + NegEx-style cues; untuned student |

## Repository contents

| Path | What it is |
| --- | --- |
| `notebooks/MIMIC_LoRA_Clinical_IE_Pipeline.ipynb` | Step-by-step notebook (outputs cleared). Cells are marked RUN / SKIP / DO NOT RUN for restarting after a crash. |
| `src/mimic_pipeline.py` | Resumable command-line pipeline: preprocessing, teacher annotation, training, prediction, evaluation |
| `src/annot_excel.py` | Converts annotation files to Excel workbooks for clinicians and back, with validation |
| `src/diag.py` | Diagnostics (counts and structure only, never note text) |
| `scripts/run_pipeline.sh` | Starts or resumes the pipeline in the background |
| `docs/annotation_guideline.md` | Annotation guideline for clinical annotators |

## Quick start

```bash
pip install -r requirements.txt

# 1. Serve the teacher model locally (example)
#    vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --port 8001 ...

# 2. Point the pipeline at your MIMIC-III DuckDB file
export MIMIC_DB=~/mimic_project/mimic3.duckdb

# 3. Check status, then run (resumes automatically after any interruption)
python src/mimic_pipeline.py --status
bash scripts/run_pipeline.sh                       # includes LoRA training
python src/mimic_pipeline.py --stage eval          # evaluation only
```

Useful options: `--tag v2` (separate model version), `--max-length 8192`, `--gen-tokens 8192`, `--batch-size 1` (lower memory), `--workers 6` (parallel teacher requests), `--max-temp 87` (pause when the GPU is hot).

## Resilience features (from real incidents)

- Every teacher and student output is written and flushed to disk per note chunk; reruns skip finished work.
- LoRA checkpoints every 10 steps, with an integrity check that skips checkpoints damaged by a power cut.
- Free-memory check before training; batch size and sequence length are configurable for shared servers.
- A run lock prevents two copies running at once; logs never print note text.

## Citation

Saxena A, et al. From governance to deployment: a framework for on-premise fine-tuning of clinical language models in healthcare organisations. *In preparation*, 2026.

Please also cite MIMIC-III (Johnson et al., *Sci Data* 2016) and, where used, the n2c2 2010 (Uzuner et al., *JAMIA* 2011) and 2018 (Henry et al., *JAMIA* 2020) shared tasks.

## Licence

Code: MIT (see [LICENSE](LICENSE)). Models and datasets are subject to their own licences and data-use agreements.

This is a research prototype and must not be used for clinical decision-making.
