# ChurnLens publication checklist

## Intended files

- `app.py`, `src/prediction.py`: application and inference validation.
- `tests/test_prediction.py`, `tests/test_evaluation.py`: UI, contract, inference, and evaluation checks.
- `requirements.txt`, `requirements-lock-windows.txt`, `.gitignore`, `.github/workflows/tests.yml`, `.streamlit/config.toml`.
- `README.md`, this checklist, `data/README.md`, `LICENSES/IBM-Apache-2.0.txt`.
- A project code license remains to be chosen; no MIT draft is included in the current files.
- Both notebooks, with execution results retained and local paths removed.
- `models/best_churn_pipeline.joblib`, `models/model_metadata.json`, `models/artifact.sha256`.
- `reports/phase2_cv_scores.csv`, `reports/figures/phase2_test_evaluation.png`.
- `data/raw/Telco-Customer-Churn.txt` and the original root-level copy (identical; both retained deliberately).
- Existing `.gitkeep` files may be included to retain empty folders.

## Exclusions

Do not stage `.venv-publication/`, `.phase2_packages/`, caches, secrets, or logs.
`data/processed/Untitled.ipynb` is a local scratch notebook; it is ignored and preserved on disk.
The local repository is connected to `https://github.com/SWAPNA2301/churn_lens.git`.

## Approval items

- Confirm the copyright-holder name and license before adopting MIT terms.
- Review staged files before subsequent commits and pushes.
- Verify Linux CI after publication; current clean-environment verification is Windows.
- Keep dataset attribution and its Apache-2.0 license with redistributed data.

The model and UI are preserved. No deployment is configured. The user authorized publishing the remaining project changes on 2026-11-08.

## Verification on 2026-11-08

- Fresh Windows Python 3.13 environment created without system packages; installed requirements successfully.
- `pytest -q`: **117 passed** in the fresh environment.
- `pip check`: no broken requirements. XGBoost import path confirmed inside the fresh environment, not `.phase2_packages`.
- Saved probabilities and holdout metrics reproduced; model checksum unchanged.
- Both notebook schemas validated; local username and absolute output paths removed.
- Git-ignore pattern checks passed for environments, caches, secrets, and the local scratch notebook.
- Pattern-based secret scan found no obvious credentials among intended text files. This is not a comprehensive security audit.
- Twelve NumPy 2.5/joblib deprecation warnings occur on artifact loading; predictions and tests pass. Retest serialization when upgrading either dependency.
