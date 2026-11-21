# Diagnostic fibers for LVEF

Code-only companion for diagnostic-fiber models and the LVEF clinical implementation, plus synthetic software tests. Manuscript and result materials are separate.

## Contents

- `src/medical_ai/`: clinical implementation and required supporting modules.
- `src/diagnostic_fibers/`: earlier synthetic reference.
- `config/`: method settings.
- `tests/`: existing artificial-fixture tests.
- `docs/cardiac/`: frozen clinical protocol required by the original source-freeze routine.

## Software checks

Python 3.12 with the recorded environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Tests use artificial arrays; small optimization steps inside them are not clinical model runs or a rerun of comparative research experiments. Dependency pins record a validated environment and may not be available on every package index/platform.

## Credentialed data and reproducibility

The clinical analysis uses credentialed MIMIC-IV/ECG. **No database files, patient data, identifiers, individual predictions, embeddings, weights, private donor arrays or MIMIC-derived result files are included.** Data access must be obtained independently: https://physionet.org/about/faqs/ . Source/schema names and generic private-data paths do not provide database access.

Critical executed clinical source was located and matched to recorded execution hashes during repository preparation. Supporting utility provenance is more limited. Full clinical reproducibility remains partial: the exact frozen cohort/features, patient-fold arrays, input plan, private constraints/checkpoints and preservation receipts are not distributed. The original clinical_inputs() expects a separately prepared results/cardiac_stage_c_v2/experiment_plan.json and hash-matched private bundles. This minimal repository cannot recreate those artifacts or clinical results automatically.

Original stage syntax, only in a separately prepared compatible authorized workspace:

```bash
PYTHONPATH=src python -m medical_ai.cardiac.clinical_neural_revision_v2 prepare
PYTHONPATH=src python -m medical_ai.cardiac.clinical_neural_revision_v2 cache
PYTHONPATH=src python -m medical_ai.cardiac.clinical_neural_revision_v2 fit --seed 20261114
```

These commands are not part of the software test procedure. Original private prerequisites and the other protocol stages/seeds are required for clinical execution. No clinical figure reproduction is advertised because its aggregate inputs are deliberately excluded.

## Scope

Earlier synthetic and later clinical experiments have different budgets and assumptions. Synthetic tower identities, donor-record coherence and software tests are not independent clinical validation or evidence of prospective test-acquisition benefit. Scientific algorithms are unchanged. This package contains no manuscripts, result tables, figures or personal submission notes. No software license grant is specified; data-access terms remain separate.
