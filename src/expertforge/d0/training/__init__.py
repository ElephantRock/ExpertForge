"""D0 production training-path semantics (Issue #54).

This package owns the frozen D0 training semantics: the learning-rate
schedule, gradient-accumulation accounting, the AdamW optimizer binding,
the precision preflight, and the accept/fail/kill threshold matrix. It is
the production counterpart of the Milestone 0 smoke gate: the smoke gate
proved the exact capture/restore methodology on a NumPy fixture; this
package applies the same discipline to the ratified D0 contract over the
D0.2 torch model.

Dependency discipline mirrors ``expertforge.d0.model``: the base install
does not require torch, and torch-dependent modules import it lazily via
:mod:`expertforge.d0.training._torch`.

Out of scope for this package (later tranches / other issues): the
training runner loop, dataset/packing integration (D0.1b), distributed
execution, generation, and any material training run (D0.5 remains
unauthorized).
"""
