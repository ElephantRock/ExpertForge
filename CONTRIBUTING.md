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
and the actual command output must be reported — not asserted.

### Environment bootstrap (first checkout / CI)

```bash
uv python install 3.11
uv sync --locked
```

### Canonical checks (run all from a clean `uv sync --locked` environment)

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
uv run python -c "import expertforge"
```

### Minimum per-PR review set

| Check | Command |
|-------|---------|
| Intended working-tree state | `git status --porcelain` |
| Staged file set matches plan | `git diff --cached --name-only` |
| No secrets staged | `git diff --cached` (scan for tokens/keys/passwords) |
| No ExpertOS content copied | path audit on `git diff --cached` (boundary is the schema/contract) |
| Local HEAD pushed to origin | `git rev-parse HEAD` equals `origin/<branch>` |
| Issue/PR template YAML parses | `uv run --locked python -c "import yaml,glob; [yaml.safe_load(open(f, encoding='utf-8')) for f in sorted(glob.glob('.github/ISSUE_TEMPLATE/*.yml'))]"` (validates all templates; PyYAML is in the dev group) |

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
