# ExpertForge
ExpertForge project repository.

## Purpose

ExpertForge is a distinct project from [ExpertOS](https://github.com/ElephantRock/ExpertOS).
The two repositories have separate responsibilities:

- **ExpertForge** is the specification, doctrine, schema, and decision authority.
  It holds the project's normative documents, versioned shared schemas, decision
  records, and the issue/PR workflow that governs accepted work.
- **ExpertOS** is the external runtime and control-plane counterpart. ExpertForge
  does not copy ExpertOS internals. Where the two systems must agree on data or
  behavior, they agree through **versioned schemas** defined and owned here.

ExpertForge treats ExpertOS solely as an external runtime/control-plane
counterpart. Nothing in the ExpertOS repository is moved, rewritten, or imported
verbatim into ExpertForge. The boundary is the schema contract, not source code.

## Canonical state

The `main` branch of this repository is the **sole canonical accepted state** of
the ExpertForge project. Chat messages, local working-tree changes, unpushed
commits, drafts, and assistant memory are provisional until merged into `main`.

When this repository's `main` conflicts with any other statement — chat, local
assumption, or external note — `main` prevails. See
[doctrine/collaboration.md](doctrine/collaboration.md).

## Working on ExpertForge

After the bootstrap, all substantive work follows:

```text
GitHub issue
→ working branch
→ implementation or document change
→ validation
→ pull request
→ review
→ merge into main
```

The bootstrap commit is the one-time founding exception that permits direct
commits to `main` (see doctrine, §16 / decision record 0001). Once the
foundation is on `main`, direct commits to `main` are no longer permitted except
for explicitly documented emergency corrections.

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch naming, validation, and
pull-request requirements, and [AGENTS.md](AGENTS.md) for assistant operating
instructions.

## Repository layout (post-bootstrap)

```text
README.md
AGENTS.md
PROJECT_STATE.md
CONTRIBUTING.md

doctrine/
├── charter.md
├── collaboration.md
├── model-lineage.md
└── decisions/
    └── 0001-github-source-of-truth.md

.github/
├── pull_request_template.md
└── ISSUE_TEMPLATE/
    ├── implementation.yml
    ├── experiment.yml
    ├── research.yml
    └── decision.yml
```

Shared interfaces with ExpertOS live under versioned schemas (added by future
issues), not as copies of ExpertOS source.
