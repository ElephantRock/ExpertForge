# Collaboration and Source-of-Truth Protocol

**Status:** Normative
**Applies to:** Web assistant, local assistant, human operator, and all project contributors

## 1. Canonical State

The GitHub repository is the sole source of truth for the project.

The following are not canonical until committed to GitHub:

* chat discussions;
* generated drafts;
* local working-tree changes;
* unpushed commits;
* assistant memory;
* experiment notes outside the repository;
* decisions not represented by an issue, pull request, or committed document.

When repository state conflicts with a chat statement or local assumption, the repository state prevails.

## 2. Canonical Branch

`main` represents the accepted project state.

All assistants must begin substantive work by reading or synchronizing with the latest `main`.

No assistant may assume that its previous understanding remains current without checking the repository.

## 3. Change Workflow

Normal changes follow this sequence:

```text
GitHub issue
→ working branch
→ implementation or document change
→ tests and validation
→ pull request
→ review
→ merge into main
```

Direct changes to `main` are permitted only during initial repository bootstrap or for explicitly documented emergency corrections.

## 4. Work Ownership

Every substantive unit of work must have a GitHub issue.

The issue defines:

* objective;
* scope;
* acceptance criteria;
* affected files or systems;
* dependencies;
* assigned execution environment;
* current status.

The execution environment must be identified as one of:

```text
web-assistant
local-assistant
human
joint
```

An assistant should not begin overlapping work when another open issue or pull request already owns the same files or responsibility.

## 5. Branch Naming

Branches should identify their origin and purpose:

```text
web/<issue-number>-<description>
local/<issue-number>-<description>
human/<issue-number>-<description>
```

Examples:

```text
web/1-bootstrap-doctrine
local/2-config-loader
local/8-d0-attention
web/12-review-experiment-schema
```

## 6. Assistant Responsibilities

### Web assistant

The web assistant is primarily responsible for:

* project doctrine;
* architecture and research specifications;
* issue decomposition;
* experiment design;
* schema design;
* repository review;
* pull-request review;
* cross-component consistency;
* evidence interpretation;
* documentation;
* milestone and acceptance review.

The web assistant may make repository changes directly through GitHub branches and pull requests when the available GitHub interface supports the required operation.

### Local assistant

The local assistant is primarily responsible for:

* implementation;
* local repository operations;
* tests;
* execution;
* profiling;
* training runs;
* environment inspection;
* dependency management;
* checkpoint verification;
* experiment artifact generation;
* changes requiring access to local hardware or files.

### Human operator

The human operator retains authority over:

* credentials;
* repository creation and access;
* hardware allocation;
* financial or compute commitments;
* external data acquisition;
* licensing decisions;
* security-sensitive actions;
* final resolution of disputed architectural decisions.

These responsibilities are defaults rather than exclusive boundaries. Each issue determines its actual owner.

## 7. Handoff Protocol

Every handoff between assistants must be represented in GitHub.

A valid handoff must identify:

* issue number;
* branch or pull request;
* current commit;
* completed work;
* unresolved work;
* validation performed;
* known failures;
* next required action.

The root file `PROJECT_STATE.md` shall provide a concise index of:

* current milestone;
* active baseline;
* active issues;
* open pull requests;
* known blockers;
* latest accepted experiment;
* next recommended action.

`PROJECT_STATE.md` is an index, not a substitute for detailed issue and pull-request records.

## 8. Agent Instructions

The root `AGENTS.md` file shall contain operating instructions that both assistants must read before modifying the repository.

It must define:

* source-of-truth rules;
* required validation;
* repository architecture;
* prohibited shortcuts;
* branch and pull-request requirements;
* documentation obligations;
* experiment provenance requirements;
* commands used for formatting, linting, testing, and validation.

More specific `AGENTS.md` files may be added inside subdirectories when component-specific instructions are needed.

## 9. Pull-Request Requirements

Every substantive pull request must include:

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

Implementation pull requests must state the exact commands used for validation.

Research pull requests must state the hypothesis, control, fixed constraints, observed result, and decision.

Documentation pull requests must identify whether the change is normative or descriptive.

## 10. Review Policy

No assistant should treat its own output as accepted merely because it was generated or committed.

Review should check:

* correctness;
* scope compliance;
* compatibility with doctrine;
* test coverage;
* reproducibility;
* hidden assumptions;
* provenance;
* unnecessary complexity;
* consistency with current baselines.

Normative doctrine, schema changes, baseline changes, and architectural decisions require explicit review before merge.

## 11. Decision Records

Durable architectural or process decisions must be recorded under:

```text
doctrine/decisions/
```

Decision records must contain:

```text
status
context
alternatives
decision
evidence
consequences
reversal conditions
```

Chat agreement alone does not constitute a durable project decision.

## 12. Conflict Resolution

When the web and local assistants produce conflicting recommendations:

1. preserve both alternatives;
2. identify the disputed assumptions;
3. determine whether repository evidence resolves the conflict;
4. create an issue or decision record when it does not;
5. prefer a reversible controlled experiment over an unsupported judgment;
6. require human resolution only when evidence or experimentation cannot economically decide.

Conflicting work must not be silently overwritten.

## 13. Concurrency Control

Before editing, each assistant must inspect:

* open issues;
* active branches where visible;
* open pull requests;
* `PROJECT_STATE.md`;
* recent changes to the target files.

When concurrent editing is unavoidable, the issue must identify file ownership or integration order.

Large shared documents should not be modified independently on multiple branches without an explicit merge strategy.

## 14. Experiment State

Experiment configurations, manifests, reports, and lightweight evidence belong in GitHub.

Large artifacts such as datasets, checkpoints, traces, and profiler outputs may live outside GitHub, but the repository must contain:

* durable artifact identifier;
* location;
* content hash;
* generation command;
* parent experiment;
* format version;
* retention status.

An external artifact without repository provenance is not a canonical project result.

## 15. Security

The repository must never contain:

* access tokens;
* passwords;
* private keys;
* cloud credentials;
* unrestricted signed URLs;
* confidential dataset contents;
* machine-specific secrets.

Secrets must be supplied through approved local or GitHub secret-management mechanisms.

## 16. Initial Exception

The first repository bootstrap may be committed directly to `main`.

After the bootstrap establishes `AGENTS.md`, `CONTRIBUTING.md`, issue templates, and pull-request rules, all substantive work must use the normal issue-and-pull-request workflow.

## 17. Founding Rule

```text
Discussion proposes.
Issues authorize.
Branches implement.
Pull requests demonstrate.
Reviews challenge.
Main decides.
GitHub remembers.
```
