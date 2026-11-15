"""Deliberate flag-only interventions for the Protocol A invariant (SPEC_trajectory "Implementation
assertions"; docs/proj_v3.md; step 5 proposal rev 3 section 5).

An edit maps a 2D validity flag (1 = invalid) to a NEW bool mask; the input is never modified. Edits are
applied inside GraspDataset.sample() on every non-Astar path, so the invariant exercises the flag-blind
input construction of R and A0 as well as M's flag channel. Eval mode only (the dataset enforces it).

centre_box48_flip: flip the flag in a fixed 48 x 48 box centred on the image grid (rows / cols
(H - 48) // 2 ... + 47). Independent of the object mask, so nothing about the object enters the edit;
exactly 48 * 48 = 2,304 flag pixels change in every scene.
"""
import numpy as np

BOX = 48


def centre_box48_flip(flag):
    f = np.asarray(flag)
    if f.ndim != 2 or f.shape[0] < BOX or f.shape[1] < BOX:
        raise ValueError(f"flag must be 2D and at least {BOX} x {BOX}, got {f.shape}")
    out = f.astype(bool, copy=True)
    i0, j0 = (f.shape[0] - BOX) // 2, (f.shape[1] - BOX) // 2
    out[i0:i0 + BOX, j0:j0 + BOX] = ~out[i0:i0 + BOX, j0:j0 + BOX]
    return out


FLAG_EDITS = {"centre_box48_flip": centre_box48_flip}
EDIT_PIXELS = {"centre_box48_flip": BOX * BOX}      # flag pixels every scene must change by
