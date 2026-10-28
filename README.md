# Fabrication After Tool Failure: research workflow repository

Companion to [arXiv:2609.14758](https://arxiv.org/abs/2609.14758). This repository preserves the complete supplied Paper 1 notebook set, with research-support scripts and bibliographic support artifacts. It is a sanitized research-workflow snapshot; external benchmark records/configurations remain required to reproduce final tables.

## Contents and workflow

| Notebook | Role |
|---|---|
| tool_failure_benchmark_workflow (1), (2) | Canonical domains, payload injection, prompt conditions, repeated runs, annotation and rejudging |
| deploy_scaffold (4) | Frozen replay grid and foreign scaffold comparison |
| defences_dev (1) | Candidate mitigations and intended development/holdout protocol |
| multimodel_sweep_1 (1) | Cross-model replay and log-probability feasibility |
| frameworks_run_(1) | Pinned prompt transfer, single-turn/multiturn and declared-tool controls |
| final_run (1) | Final harness comparisons, scope restriction and revised judgments |

The two canonical notebook copies contain identical concatenated cell-source hashes; one canonical notebook is published with the duplicate recorded in notebooks/ALIASES.md. `support/` includes echo quantification, method-fact extraction, model probing and bibliography/citation utilities plus their reviewed CSV inputs/reports. Those supporting artifacts are not automatically results of the final published experiment.

## Mechanism and external inputs

Freeze benchmark configuration, model/prompt versions and tool declarations. Force the first tool call, inject an assigned unusable payload, allow bounded continuation, and retain transcripts and item identities. Use the declaration-aware four-label rubric, validate annotations, restrict applicable analyses to in-scope questions and compare matched prompt arms. Scripts expect external `benchmark_config` and result JSONL under Google Drive; provenance tables identify their roles but do not fabricate missing data.

Controls are intentionally uneven across repeats: the full repeat-zero grid has 1,024 items; explicit TIMEOUT/HTTP_500 controls occur only once, while six ambiguous conditions repeat. Thus 2,560 rows per baseline arm is the intended design, not 3,072. Final scope-restricted tables use a different reporting population.

## Known research-snapshot caveats

- `defences_dev` cell 2 sets PHASE to TEST but cell 6 still filters DEV_DOMAINS and constructs DEV_IDS. Do not use that snapshot as evidence of a correct held-out execution; reconcile the final code and run artifacts before reproducing the paper's held-out result. The preparation deliberately preserves this discrepancy rather than silently repairing research logic.
- `final_run` contains several successive setup and table cells. Its 688-item in-scope restriction addresses whether a tool could supply the requested field; it must not be confused with the complete 1,024-item benchmark.
- Nonanswers, retries, API errors and judge errors have version-specific treatment. Preserve intended denominator rules and harness identity when comparing tables.
- Model/access probing and bibliography utilities can make network/API requests. `dump_methods_facts.py` inspects assignments from external notebooks; only run it on trusted research inputs.

Offline echo self-tests can be requested after inspecting the script:

```bash
python support/echo_quantification.py --selftest
```

The self-test was not executed as part of preparation.

## Credentials and public preparation

All notebook outputs, execution counts and cell/notebook metadata were cleared. Four embedded service-account dictionaries in the Paper 2 source were replaced with environment loading. Credential identity diagnostics were removed. Colab secret reads for the service account now load the `GCP_SA_JSON` environment variable. Original spreadsheet identifiers are replaced with `RESEARCH_SHEET_URL`; automatic sheet publishing defaults to false. Literal cloud project settings use `GCP_PROJECT_ID`. Do not commit real values, credential JSON or generated raw datasets.

Set `GCP_SA_JSON` to the complete service-account JSON string through your runtime's private environment configuration, and `GCP_PROJECT_ID` when a notebook requests it. Do not put these values into notebook cells. The original notebooks use mounted Google Drive and historical Vertex AI model identifiers; they are Colab-oriented and require the corresponding external run/configuration files. Endpoints, permissions and SDK behavior must be checked before a paid run.

## Setup and run

Use a fresh Colab Python runtime, privately configure the environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Open the chosen notebook in Colab or Jupyter and review its configuration and local/Drive file dependencies before running its cells in the documented order. These notebooks contain successive setup/result-cell revisions; they are research records rather than a single unattended execution script. No end-to-end run or model call was performed during repository preparation. Do not use Execute All until you have resolved replacement cells, supplied the intended frozen inputs and confirmed output paths.

For an optional local notebook editor:

```bash
python -m pip install jupyterlab
python -m jupyter lab
```

The notebooks' Google Drive and `google.colab` calls require adaptation outside hosted Colab. Model generation can incur charges; no generation was performed here. Spreadsheet write cells must remain disabled unless intentionally configured.

## Provenance and integrity

`provenance.json` records original and sanitized file SHA-256 hashes and cell-specific sanitization changes. Original notebook filenames are preserved. No benchmark questions, payload rules, rubric logic, model settings, statistical procedures or research-control logic were intentionally changed. Credentials, identifying export configuration and execution metadata were the only preparation changes. `security-scan.json` records a static scan with zero detected key literals/provider tokens, zero retained notebook outputs and no syntax findings in parseable non-magic cells. This is not a runtime or scientific replication result.

No license has been inferred from local possession. Confirm collaborator agreement and applicable rights before adding a redistribution license. Paper findings should be attributed to the collaborative paper rather than solely to this repository owner.


## Browse source

[Every published file](FILE-INDEX.md) · [Snapshot hashes](publication-manifest.json)
