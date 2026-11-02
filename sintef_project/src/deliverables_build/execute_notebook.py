"""Execute a notebook headlessly and save it with outputs.

    python execute_notebook.py <in.ipynb> <out.ipynb> <working dir> [kernel name]

Writes the output notebook once, at the end (also on failure), and prints each cell's output tail.
"""
import sys, time, nbformat
from nbclient import NotebookClient
src, dst, cwd = sys.argv[1], sys.argv[2], sys.argv[3]
nb = nbformat.read(src, as_version=4)
t = time.time()
client = NotebookClient(nb, timeout=None, kernel_name=sys.argv[4] if len(sys.argv) > 4 else "python3", resources={"metadata": {"path": cwd}})
try:
    client.execute()
    print(f"EXECUTED OK in {time.time()-t:.0f}s", flush=True)
finally:
    nbformat.write(nb, dst)
    for c in nb.cells:
        if c.cell_type == "code":
            for o in c.get("outputs", []):
                txt = o.get("text") or "".join(o.get("traceback", [])) or str(o.get("data", {}).get("text/plain", ""))
                print(txt[-600:], flush=True)
