#!/usr/bin/env python3
# build_prototype_gallery.py — HTML page of prototype mean shapes & exemplars
# Fix: use string.Template so CSS braces { } don't collide with str.format()

import os
import glob
import argparse
from pathlib import Path
from string import Template

TPL = Template("""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Prototype gallery</title>
  <style>
    body { font-family: sans-serif; margin: 24px; }
    h1 { margin-bottom: 8px; }
    h2 { margin: 28px 0 8px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px; }
    img { max-width: 100%; height: auto; display: block; }
    .caption { color: #555; font-size: 12px; margin-top: 6px; }
  </style>
</head>
<body>
  <h1>ERROR prototype narratives</h1>
  $sections
</body>
</html>
""")

SECTION_TPL = Template("""
<h2>Prototype $pid</h2>
<div class="grid">
  $cards
</div>
""")

CARD_TPL = Template("""
<div class="card">
  <img src="$src" alt="$alt">
  <div class="caption">$alt</div>
</div>
""")

def make_section(pid: str, images):
    cards = "\n".join(
        CARD_TPL.safe_substitute(src=os.path.basename(p), alt=os.path.basename(p))
        for p in images
    )
    return SECTION_TPL.safe_substitute(pid=pid, cards=cards)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="outputs/narratives")
    ap.add_argument("--out", default=None, help="Optional explicit output path")
    args = ap.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder not found: {folder}")

    # Collect all narrative images: narrative_proto_{pid}_*.png
    paths = sorted(glob.glob(os.path.join(folder, "narrative_proto_*_*.png")))
    if not paths:
        raise SystemExit(f"No narrative images found in {folder} (expected narrative_proto_*_*.png).")

    groups = {}
    for p in paths:
        base = os.path.basename(p)
        # expected pattern: narrative_proto_{pid}_...
        parts = base.split("_")
        # guard against unexpected names
        pid = parts[2] if len(parts) > 2 and parts[0] == "narrative" and parts[1] == "proto" else "unknown"
        groups.setdefault(pid, []).append(p)

    # Build sections in numeric pid order when possible
    def pid_key(k):
        try:
            return int(k)
        except:
            return float("inf")

    sections = "\n".join(
        make_section(pid, imgs) for pid, imgs in sorted(groups.items(), key=lambda kv: pid_key(kv[0]))
    )

    out = args.out or os.path.join(folder, "index.html")
    Path(out).write_text(TPL.safe_substitute(sections=sections), encoding="utf-8")
    print(f"✓ wrote {out}")

if __name__ == "__main__":
    main()
