# CONTRIBUTING.md

How to contribute to ExpertForge. This document is normative for the change
workflow. Assistants and human contributors follow the same rules.

## 1. Canonical state

`main` is the sole canonical accepted state. See
[doctrine/collaboration.md](doctrine/collaboration.md). Chat, drafts, local
changes, and unpushed commits are not canonical.

## 2. Change workflow

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
exception (doctrine §16 / decision record 0001) or for explicitly documented
emergency corrections. All other work uses the issue→branch→PR→review→merge path.

## 3. Issues first

Every substantive unit of work has a GitHub issue. The issue defines: objective;
scope; acceptance criteria; affected files or systems; dependencies; assigned
execution environment (`web-assistant`, `local-assistant`, `human`, or `joint`);
current status.

Do not begin overlapping work when another open issue or PR already owns the
same files or responsibility (see doctrine §4, §13).

## 4. Branch naming

```text
web/<issue-number>-<description>
local/<issue-number>-<description>
human/<issue-number>-<description>
```

Examples:

```text
web/1-bootstrap-doctrine
local/2-schema-registry
human/3-access-review
```

## 5. Pull-request requirements

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
  result, and decision.
- **Documentation PRs** must identify whether the change is normative or
  descriptive.

Normative doctrine, schema changes, baseline changes, and architectural
decisions require explicit review before merge (doctrine §10).

## 6. Validation

Validation must be performed in the same session in which completion is claimed,
and the actual command output must be reported — not asserted. The minimum
validation set for a documentation/schema repository like ExpertForge:

| Check | Command |
|-------|---------|
| Intended working-tree state | `git status --porcelain` |
| Staged file set matches plan | `git diff --cached --name-only` |
| No secrets staged | `git diff --cached` (scan for tokens/keys/passwords) |
| Local HEAD pushed to origin | `git rev-parse HEAD` equals `origin/main` |
| Issue/PR template YAML parses | `python -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" <file>` per template |

When executable schema tooling is added (e.g., a JSON Schema validator), the
exact command and tool version will be recorded here and referenced from
[AGENTS.md](AGENTS.md) §9. Implementation PRs restate their validation commands
verbatim.

## 7. Review policy

No contributor treats their own output as accepted merely because it was
generated or committed. Review checks: correctness; scope compliance;
compatibility with doctrine; test coverage; reproducibility; hidden assumptions;
provenance; unnecessary complexity; consistency with current baselines
(doctrine §10).

## 8. Decision records

Durable architectural or process decisions are recorded under
`doctrine/decisions/NNNN-<slug>.md` with: status, context, alternatives,
decision, evidence, consequences, reversal conditions. Chat agreement alone is
not a durable decision (doctrine §11).

## 9. Security

Never commit access tokens, passwords, private keys, cloud credentials,
unrestricted signed URLs, confidential dataset contents, or machine-specific
secrets. Supply secrets through approved local or GitHub secret-management
mechanisms only (doctrine §15).

## 10. Concurrency

Before editing, inspect open issues, active branches where visible, open PRs,
`PROJECT_STATE.md`, and recent changes to the target files. Large shared
documents are not modified independently on multiple branches without an
explicit merge strategy (doctrine §13).
