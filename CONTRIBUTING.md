# CONTRIBUTING.md

How to contribute to ExpertForge. This document is normative for the change
workflow. Assistants and human contributors follow the same rules.

## 1. Canonical state

`main` is the sole canonical accepted state. See
[doctrine/collaboration.md](doctrine/collaboration.md). Chat, drafts, local
changes, and unpushed commits are not canonical.

## 2. Project scope

ExpertForge is a complete language-model project: it builds, trains, evaluates,
instruments, and deploys its own dense and sparse models from random
initialization. Contributions span data, tokenizer, model, training, evaluation,
instrumentation, reference inference, doctrine, schemas, and reports. See
[doctrine/charter.md](doctrine/charter.md).

## 3. Change workflow

```text
GitHub issue
→ working branch
→ implementation or document change
→ validation
→ pull request
→ review
→ merge into main
```

Direct commits to `main` are permitted only during the one-time bootstrap
exception or for explicitly documented emergency corrections. All other work
uses the issue→branch→PR→review→merge path.

## 4. Issues first

Every substantive unit of work has a GitHub issue. The issue defines: objective;
scope; acceptance criteria; affected files or systems; dependencies; assigned
execution environment (`web-assistant`, `local-assistant`, `human`, or `joint`);
current status. Do not begin overlapping work when another open issue or PR
already owns the same files or responsibility (collaboration §4, §13).

## 5. Branch naming

```text
web/<issue-number>-<description>
local/<issue-number>-<description>
human/<issue-number>-<description>
```

Examples: `local/1-correct-founding-scope`, `web/5-experiment-schema`.

## 6. Pull-request requirements

Every substantive PR uses `.github/pull_request_template.md` and includes:

```text
Issue:
Execution environment:
Objective:
Changes:
Validation:
Known limitations:
Evidence:
Files requiring special review:
Follow-up work:
```

- **Implementation PRs** must state the exact commands used for validation.
- **Research PRs** must state hypothesis, control, fixed constraints, observed
  result, and decision (doctrine/evaluation-and-experiments.md §2).
- **Documentation PRs** must identify whether the change is normative or
  descriptive.

Normative doctrine, schema/resource-contract changes, baseline changes, and
architectural decisions require explicit review before merge (collaboration §10).

## 7. Validation

Validation must be performed in the same session in which completion is claimed,
and the actual command output must be reported — not asserted. The permanent
`.github/workflows/ci.yml` workflow repeats the portable checks on pull requests
and pushes to `main` with read-only repository permissions.

### Environment bootstrap (first checkout / CI)

```bash
uv python install 3.11
uv sync --locked
```

### Canonical full validation

Run all commands from a clean `uv sync --locked` environment:

```bash
uv sync --locked
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src tests
uv run --locked pytest
uv run --locked python -c "import expertforge"
uv run --locked expertforge-config configs/smoke.yaml > /dev/null
uv run --locked python scripts/validate_repository.py all
uv lock --check
```

### Test tiers

Fast CPU suite for routine development:

```bash
uv run --locked pytest -m "not integration and not smoke and not accelerator"
```

Portable CPU integration suite:

```bash
uv run --locked pytest -m "integration and not smoke and not accelerator"
```

The unfiltered `uv run --locked pytest` command is the complete current suite.
Pytest markers are strict:

- `integration` identifies portable cross-component, subprocess, or isolated
  filesystem tests;
- `smoke` identifies end-to-end training/recovery checks owned by Issue #14;
- `accelerator` identifies optional hardware/runtime-dependent tests, which must
  skip with an explicit reason when unavailable.

### Minimum per-PR review set

| Check | Command |
|-------|---------|
| Intended working-tree state | `git status --porcelain` |
| Staged file set matches plan | `git diff --cached --name-only` |
| Repository policy checks | `uv run --locked python scripts/validate_repository.py all` |
| Local HEAD pushed to origin | `git rev-parse HEAD` equals `origin/<branch>` |

The repository-policy command parses all issue-template YAML, performs the
high-confidence tracked-file secret scan, and audits the ExpertOS source
boundary. It is deterministic, network-free, and emits stable diagnostics.

Implementation/training PRs restate their lint/format/test commands verbatim
(the canonical set above) and respect the scientific baseline rules
([doctrine/model-lineage.md](doctrine/model-lineage.md) §7). Dependency changes
update both `pyproject.toml` and `uv.lock`; upgrades are explicit PRs.

## 8. Review policy

No contributor treats their own output as accepted merely because it was
generated or committed. Review checks: correctness; scope compliance;
compatibility with doctrine; test coverage; reproducibility; hidden assumptions;
provenance; unnecessary complexity; consistency with current baselines
(collaboration §10).

## 9. Decision records

Durable architectural or process decisions are recorded under
`doctrine/decisions/NNNN-<slug>.md` with: status, context, alternatives,
decision, evidence, consequences, reversal conditions. Chat agreement alone is
not a durable decision (collaboration §11).

## 10. Security

Never commit access tokens, passwords, private keys, cloud credentials,
unrestricted signed URLs, confidential dataset contents, or machine-specific
secrets. Supply secrets through approved local or GitHub secret-management
mechanisms only (collaboration §15).

## 11. Concurrency

Before editing, inspect open issues, active branches where visible, open PRs,
`PROJECT_STATE.md`, and recent changes to the target files. Large shared
documents are not modified independently on multiple branches without an
explicit merge strategy (collaboration §13).
