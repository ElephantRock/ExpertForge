# Founding Charter

**Status:** Normative
**Applies to:** ExpertForge repository and all contributors

## 1. What ExpertForge is

ExpertForge is the **specification, doctrine, schema, and decision authority**
for the ExpertForge project. It holds normative documents, versioned shared
schemas, decision records, and the issue/pull-request workflow that governs
accepted work.

## 2. What ExpertForge is not

ExpertForge is not ExpertOS, and is not a copy or fork of ExpertOS. ExpertOS is
referenced as the **external runtime/control-plane counterpart** only. ExpertForge
does not move, rewrite, or import ExpertOS internals. See §3.

## 3. Relationship to ExpertOS

- [ExpertOS](https://github.com/ElephantRock/ExpertOS) is a separate repository
  with separate responsibilities (runtime and control plane).
- ExpertForge and ExpertOS are **distinct projects**. Nothing in ExpertOS is
  treated as ExpertForge bootstrap material.
- Where the two systems must agree on data or behavior, they agree through
  **versioned schemas** defined and owned by ExpertForge. Both sides conform to
  the schema contract; neither copies the other's source.

## 4. Canonical state

The `main` branch of `ElephantRock/ExpertForge` is the **sole canonical accepted
state** of the project. All other artifacts — chat, drafts, local working-tree
changes, unpushed commits, assistant memory — are provisional until merged into
`main`. When `main` conflicts with any other statement, `main` prevails.

## 5. Change authority

After the bootstrap exception (doctrine §16 / decision record 0001), normal work
requires: a GitHub issue; a working branch; validation; a pull request; review;
and merge into `main`. Direct commits to `main` are prohibited thereafter except
for explicitly documented emergency corrections.

## 6. Honesty obligation

Contributors and assistants report actual validation output, not asserted
results. Incomplete work is stated as incomplete. Disagreements between
assistants are preserved and resolved by evidence or experiment, not by silent
overwrite (doctrine §12).

## 7. Founding rule

```text
Discussion proposes.
Issues authorize.
Branches implement.
Pull requests demonstrate.
Reviews challenge.
Main decides.
GitHub remembers.
```
