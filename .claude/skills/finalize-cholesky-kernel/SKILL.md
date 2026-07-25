---
name: finalize-cholesky-kernel
description: Finalizes the cholesky.py kernel for submission. Use when the user asks to update/finalize cholesky.py, consolidate cholesky kernels.
---

## Instructions

The default variant of a shape is its `_DEFAULT_VARIANT = <id>  # POPCORN_VARIANT` line; files without one are torch fallbacks, so skip them. Resolve the id to a name via that file's `_VARIANT_NAMES` tuple or its `DESIGN.md` variant table.

Fetch the code for **default variants** of every cholesky/bBnN/cholesky_bBnN.py kernel. Strip ALL development code that we don't need for full submission (ex. launch configurations/templates, metadata, prints etc.). Fold the cleaned up kernel into cholesky/cholesky.py, appending 'bBnN' suffix to symbols to resolve naming conflicts. The full launcher should route to this kernel only for BxNxN shape matrices.

Add comments at the top for which variant you added for each implemented shape. Route to torch.linalg.cholesky_ex for any unimplemented shapes.