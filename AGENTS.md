# AGENTS.md

Operating instructions for all assistants and contributors working on ExpertForge.
**Read this before modifying the repository.** More specific `AGENTS.md` files
may be added inside subdirectories for component-specific instructions.

## 1. Project identity

ExpertForge is a **complete language-model research and engineering project**.
It builds, trains, evaluates, instruments, and deploys its own dense and sparse
models from random initialization. See [doctrine/charter.md](doctrine/charter.md).

ExpertForge is *not* a documentation/schema-only repository. It owns and
executes its own model, training, evaluation, and reference inference code.

## 2. Source of truth

The `main` branch of `ElephantRock/ExpertForge` is the sole canonical accepted
state of the project. Everything else is provisional: chat discussions;
generated drafts; local working-tree changes; unpushed commits; assistant
memory; decisions not represented by an issue, pull request, or committed
document. When `main` conflicts with any other statement, `main` prevails. See
[doctrine/collaboration.md](doctrine/collaboration.md).

## 3. Relationship to ExpertOS

ExpertForge and [ExpertOS](https://github.com/ElephantRock/ExpertOS) are
separate repositories with distinct responsibilities.

- **ExpertForge owns:** data ingestion and preparation; tokenizer training and
  evaluation; dense Transformer implementations; sparse MoE implementations;
  optimizers, schedulers, mixed precision, distributed training, checkpointing,
  and recovery; evaluation, generation, benchmarking, profiling, and experiment
  tracking; routing and expert instrumentation; controlled architectural and
  training experiments; model lineage and experiment evidence; reference
  inference/runtime code needed to understand and test ExpertForge models;
  deployment-aware objectives and hardware-cost modeling; versioned
  interoperability schemas and resource contracts.
- **ExpertOS owns:** profiling and controlling existing or ExpertForge-produced
  MoEs through its runtime/control plane; expert intervention and
  quality-sensitivity evaluation; placement, residency, movement, prefetch,
  compression, and runtime policy selection; execution policies for constrained
  hardware; runtime-side evidence and deployment evaluation.

**Do not** move, copy, or rewrite ExpertOS source into ExpertForge. Where the two
must agree on data or behavior, define a **versioned schema or resource
contract** in ExpertForge (`schemas/`) and have ExpertOS consume it. The
boundary is the contract, not source code.

**Prohibited statement:** "model execution happens in ExpertOS, not ExpertForge."
ExpertForge may and must execute its own models for training, evaluation,
generation, profiling, and reference inference.

## 4. Repository architecture (planned)

```text
doctrine/        Normative documents (charter, lineage, doctrine, decisions)
configs/         Validated, serialized, fully resolved run configurations
src/             Executable source: data, tokenizer, model, training,
                 evaluation, instrumentation, runtime/reference inference
schemas/         Versioned interoperability schemas + ExpertOS resource contract
tests/           Automated tests
experiments/     Experiment manifests and lightweight evidence
reports/         Experiment reports and decision records
scripts/         Operational and reproduction scripts

.github/
  pull_request_template.md
  ISSUE_TEMPLATE/   implementation / experiment / research / decision
```

Core model logic must not live only in notebooks. Canonical behavior is
controlled by validated, serialized, immutable, fully resolved configuration
embedded in or referenced by checkpoints and experiment records.

## 5. Required validation

Every substantive change must be validated before it is claimed complete. Run
the validation commands and report their actual output and exit codes — do not
assert results you have not observed in the current session.

Minimum validation for a doctrine/documentation change:

```text
1. Working tree is in the intended state:        git status --porcelain
2. The staged file set matches the plan:         git diff --cached --name-only
3. No secrets are staged:                         git diff --cached | secret-pattern check
4. The local commit is what was pushed:          git rev-parse HEAD vs origin
5. YAML frontmatter in templates parses:         see CONTRIBUTING.md §Validation
6. No ExpertOS source/content copied:            git diff --cached path audit
```

Implementation/training changes must additionally state the exact commands used
for linting, formatting, and testing, and must respect the scientific baseline
rules ([doctrine/model-lineage.md](doctrine/model-lineage.md) §3). Research
changes must state hypothesis, control, fixed constraints, observed result, and
decision ([doctrine/evaluation-and-experiments.md](doctrine/evaluation-and-experiments.md)).

## 6. Prohibited shortcuts

Do **not**:

- commit secrets, tokens, credentials, or confidential dataset contents;
- copy ExpertOS internals into ExpertForge instead of defining a schema/contract;
- claim work is complete without showing the validation output in the same session;
- silently overwrite conflicting work — preserve both and escalate (collaboration §12);
- treat chat agreement as a durable decision — record it in `doctrine/decisions/`;
- push an unvalidated commit, or push and then validate;
- begin a large training run before Milestone 0 is complete;
- enable multiple deployment-aware penalties together and report them as one
  "hardware-aware" result;
- combine several unablated architectural changes and attribute the result to one
  technique;
- retroactively modify a frozen baseline.

## 7. Branch and pull-request requirements

Branch naming (origin and purpose):

```text
web/<issue-number>-<description>
local/<issue-number>-<description>
human/<issue-number>-<description>
```

Examples: `local/1-correct-founding-scope`, `web/5-experiment-schema`,
`human/8-access-review`.

Every substantive pull request must use the template at
`.github/pull_request_template.md`. See [CONTRIBUTING.md](CONTRIBUTING.md).

## 8. Documentation obligations

- Normative doctrine changes require explicit review before merge.
- Schema / resource-contract changes require explicit review before merge.
- Baseline changes require explicit review before merge.
- Every durable architectural/process decision gets a record under
  `doctrine/decisions/NNNN-<slug>.md`.
- `PROJECT_STATE.md` is an index, not a substitute for issue/PR records.

## 9. Experiment provenance

Experiment configurations, manifests, reports, and lightweight evidence belong
in GitHub. Large artifacts (datasets, checkpoints, traces, profiler outputs) may
live outside GitHub, but the repository must record for each: durable artifact
identifier, location, content hash, generation command, parent experiment,
format version, and retention status. An external artifact without repository
provenance is not a canonical project result (collaboration §14).

## 10. Commands

The authoritative validation commands are listed in CONTRIBUTING.md §Validation
and must be re-stated verbatim in each implementation PR. When executable
tooling is added (schema validator, training entrypoint, test runner), the exact
command and version will be recorded there and referenced here.
