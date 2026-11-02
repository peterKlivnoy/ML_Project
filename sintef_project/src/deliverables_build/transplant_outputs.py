"""Copy outputs from an executed notebook into a regenerated one whose code cells are identical.

    python transplant_outputs.py <executed.ipynb> <regenerated.ipynb>

Used after a generator changes only markdown (headers, notes): the outputs stay valid because every
code cell is asserted identical.
"""
import sys, nbformat
ex = nbformat.read(sys.argv[1], as_version=4); new = nbformat.read(sys.argv[2], as_version=4)
ec = [c for c in ex.cells if c.cell_type == "code"]; nc = [c for c in new.cells if c.cell_type == "code"]
assert len(ec) == len(nc)
for a, b in zip(ec, nc):
    assert a.source == b.source, "code cell differs: " + b.source[:80]
    b.outputs, b.execution_count = a.outputs, a.execution_count
for k in ("language_info",):
    if k in ex.metadata: new.metadata[k] = ex.metadata[k]
nbformat.write(new, sys.argv[2]); print("transplanted", len(nc), "code cells")
