"""git clean filter: store notebooks WITHOUT outputs (they are large and may contain subject images).
Setup (once per clone):  git config filter.nbstrip.clean "python scripts/strip_notebook_outputs.py" ; git config filter.nbstrip.smudge cat"""
import json, sys
nb = json.load(sys.stdin)
for c in nb.get("cells", []):
    if c.get("cell_type") == "code":
        c["outputs"] = []; c["execution_count"] = None
    c.get("metadata", {}).pop("execution", None)
json.dump(nb, sys.stdout, ensure_ascii=False, indent=1); sys.stdout.write("\n")
