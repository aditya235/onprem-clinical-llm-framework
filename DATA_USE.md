# Data use and privacy

This repository contains **code and templates only**.

## What is not included, and why

| Not included | Reason |
| --- | --- |
| MIMIC-III notes or tables | PhysioNet Credentialed Health Data Use Agreement forbids redistribution |
| Any annotation files (silver or gold labels, workbooks) | They contain note text or text derived from it |
| Model predictions, error analyses, notebook outputs | They contain note text or entity text quoted from notes |
| Trained LoRA adapters | Trained on data covered by the data-use agreement |
| n2c2 corpora | Available only through the DBMI Data Portal under its own agreement |

## Getting the data

- **MIMIC-III:** complete the required training and sign the data-use agreement on PhysioNet, then download MIMIC-III and load it into a local DuckDB file.
- **n2c2 2010 and 2018 Track 2 (optional, for gold-standard evaluation):** request access through the DBMI Data Portal. Each user must download the data individually.

## Rules for contributors

1. **Clear all notebook outputs before committing** (Jupyter: Edit → Clear All Outputs). Outputs can contain note text.
2. Never commit files from the pipeline output folder (`ie_pipeline/`); `.gitignore` excludes it, but web uploads ignore `.gitignore`, so check what you upload.
3. Never paste note text, model outputs or entity tables into issues, pull requests or external AI tools. Share error messages and counts only.
4. Run all processing on institutional, on-premise systems.
