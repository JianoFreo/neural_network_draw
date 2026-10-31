"""
Play every example in this folder, in order (example_1 ... example_14).

    python tests/run_all_examples.py              # a window per diagram; close it for the next
    python tests/run_all_examples.py --save-only  # no windows, PNGs go to tests/demo_output/
    python tests/run_all_examples.py 7 10         # only examples 7 and 10
    python tests/run_all_examples.py --list       # just list them

The PyTorch examples (names containing "pytorch") are skipped if torch isn't installed.
"""
import glob
import os
import re
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

args = [a for a in sys.argv[1:] if not a.startswith("--")]
SAVE_ONLY = "--save-only" in sys.argv

import matplotlib  # noqa: E402
if SAVE_ONLY:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def number(path):
    return int(re.match(r"example_(\d+)_", os.path.basename(path)).group(1))


examples = sorted(glob.glob(os.path.join(HERE, "example_*.py")), key=number)
if args:
    wanted = {int(a) for a in args}
    examples = [p for p in examples if number(p) in wanted]

if "--list" in sys.argv:
    for p in examples:
        print(os.path.basename(p))
    sys.exit()

out_dir = os.path.join(HERE, "demo_output")
if SAVE_ONLY:
    os.makedirs(out_dir, exist_ok=True)

real_show = plt.show
for path in examples:
    name = os.path.splitext(os.path.basename(path))[0]
    if "pytorch" in name and not HAVE_TORCH:
        print(f"-- skipping {name} (PyTorch not installed)")
        continue

    counter = {"n": 0}

    def show(*a, _name=name, _c=counter, **k):
        _c["n"] += 1
        if SAVE_ONLY:
            plt.savefig(os.path.join(out_dir, f"{_name}_{_c['n']}.png"), dpi=100,
                        bbox_inches="tight")
        real_show(*a, **k)

    plt.show = show
    print(f"\n=== {name} ===")
    runpy.run_path(path)["run"]()

if SAVE_ONLY:
    print(f"\nPNGs saved in {out_dir}")
