# AGENTS.md

Operating instructions for all assistants and contributors working on ExpertForge.
**Read this before modifying the repository.** More specific `AGENTS.md` files
may be added inside subdirectories for component-specific instructions.

## 1. Source of truth

The `main` branch of `ElephantRock/ExpertForge` is the sole canonical accepted
state of the project. Everything else is provisional:

- chat discussions;
- generated drafts;
- local working-tree changes;
- unpushed commits;
- assistant memory;
- decisions not represented by an issue, pull request, or committed document.

When `main` conflicts with any other statement, `main` prevails. See
[doctrine/collaboration.md](doctrine/collaboration.md).

## 2. Relationship to ExpertOS

ExpertForge and [ExpertOS](https://github.com/ElephantRock/ExpertOS) are
**separate repositories with distinct responsibilities**.

- ExpertForge owns specification, doctrine, schema, and decisions.
- ExpertOS is the external runtime/control-plane counterpart.

**Do not** move, copy, or rewrite ExpertOS source into ExpertForge. Where the two
must agree on data or behavior, define a **versioned schema** in ExpertForge and
have both sides conform to it. The boundary between the projects is the schema
contract.

## 3. Required validation

Every substantive change must be validated before it is claimed complete. Run
the validation commands and report their actual output and exit codes — do not
assert results you have not observed in the current session.

Minimum validation for a documentation/bootstrap change:

```text
1. Working tree is in the intended state:        git status --porcelain
2. The staged file set matches the plan:         git diff --cached --name-only
3. No secrets are staged:                         git diff --cached | secret-pattern check
4. The local commit is what was pushed:          git rev-parse HEAD vs origin/main
5. YAML frontmatter in templates parses:         see CONTRIBUTING.md §Validation
```

Implementation changes must additionally state the exact commands used for
linting, formatting, and testing. Research changes must state hypothesis,
control, fixed constraints, observed result, and decision.

## 4. Repository architecture

```text
README.md                 Project overview and canonical-state statement
AGENTS.md                 This file — assistant operating instructions
PROJECT_STATE.md          Concise index of current milestone, active baseline,
                          open issues/PRs, blockers, latest accepted experiment,
                          next recommended action
CONTRIBUTING.md           Branch naming, PR requirements, validation commands

doctrine/                 Normative project documents
  charter.md              Founding charter and project scope
  collaboration.md        Source-of-truth and change protocol (normative)
  model-lineage.md        Model/lineage provenance policy
  decisions/              Durable decision records (ADR-style)

.github/
  pull_request_template.md
  ISSUE_TEMPLATE/         implementation / experiment / research / decision

schemas/                  (future) Versioned shared schemas interfacing ExpertOS
```

## 5. Prohibited shortcuts

Do **not**:

- commit secrets, tokens, credentials, or confidential dataset contents;
- copy ExpertOS internals into ExpertForge instead of defining a schema;
- commit directly to `main` after bootstrap (except documented emergencies);
- claim work is complete without showing the validation output in the same session;
- silently overwrite conflicting work — preserve both and escalate per doctrine §12;
- treat chat agreement as a durable decision — record it in `doctrine/decisions/`;
- push an unvalidated commit, or push and then validate.

## 6. Branch and pull-request requirements

Branch naming (origin and purpose):

```text
web/<issue-number>-<description>
local/<issue-number>-<description>
human/<issue-number>-<description>
```

Examples: `web/1-bootstrap-doctrine`, `local/2-schema-registry`,
`human/3-access-review`.

Every substantive pull request must use the template at
`.github/pull_request_template.md` and include: issue, execution environment,
objective, changes, validation, known limitations, evidence, files requiring
special review, and follow-up work. See CONTRIBUTING.md for full requirements.

## 7. Documentation obligations

- Normative doctrine changes require explicit review before merge.
- Schema changes require explicit review before merge.
- Baseline changes require explicit review before merge.
- Every durable architectural/process decision gets a record under
  `doctrine/decisions/NNNN-<slug>.md` with: status, context, alternatives,
  decision, evidence, consequences, reversal conditions.
- `PROJECT_STATE.md` is an index, not a substitute for issue/PR records.

## 8. Experiment provenance requirements

Experiment configurations, manifests, reports, and lightweight evidence belong
in GitHub. Large artifacts (datasets, checkpoints, traces, profiler outputs) may
live outside GitHub, but the repository must record for each: durable artifact
identifier, location, content hash, generation command, parent experiment,
format version, and retention status. An external artifact without repository
provenance is not a canonical project result.

## 9. Commands

ExpertForge is primarily a documentation/schema repository at bootstrap. The
authoritative validation commands are listed in CONTRIBUTING.md §Validation and
must be re-stated verbatim in each implementation PR. When executable schema
tooling is added (e.g., a JSON Schema validator), the exact command and version
will be recorded there and referenced here.
