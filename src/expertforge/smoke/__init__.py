"""ExpertForge Milestone 0 smoke gate (Issue #14).

This package implements the **end-to-end training, checkpoint, recovery, and
exact-restoration smoke gate** that proves the Milestone 0 substrate works
together. It is explicitly an **infrastructure gate**, not a reusable training
framework or a D0 model precursor (amendment B): the toy model is the smallest
auditable pure-NumPy autoregressive decoder, and all of training/checkpoint/
recovery/manifest publication flows through the real substrate subsystems.

Normative references:
- Design comment ``5151917939`` (ratified proposal).
- Binding amendment ``5152078541`` (amendments A–P).
- Authorization + clarifications ``5152259715``.
"""

from __future__ import annotations
