"""make_dataset_layout_flat.py — convert a FLAT, filename-split LEVIR-CD256
style download into the layout DiffRegCD's CMUCDFlowDataset expects.

Use this version when your download looks like:
    <src>/A/train_1_1.png, val_3_2.png, test_100_1.png, ...
    <src>/B/<same names>
    <src>/label/<same names>
i.e. images for ALL splits sit flat in A/ B/ label/, and the split is the
prefix of the filename up to the first underscore (train_ / val_ / test_).

(If your download instead has train/A, val/A, test/A SUBFOLDERS, use the
original make_dataset_layout.py instead.)

Output (DiffRegCD layout):
    <out>/train/t0/     post-change images (epoch B) — the reference
    <out>/srcA/         pre-change images (epoch A)   — staging only
    <out>/train/mask/   binary change masks
    <out>/list/{train,val,test}.txt

Then generate the affine-warped view + GT flow:
    python scripts/prepare_gt_flow.py --src <out>/srcA \
        --warped-out <out>/train/t1 --flow-out <out>/train/flow --seed 0

Usage:
    # everything (train+val+test) — big, only needed if you'll also train:
    python make_dataset_layout_flat.py --src LEVIR-CD256 --out LEVIR-CD256-out

    # TEST SPLIT ONLY (what you need for `-p test`) — much smaller/faster:
    python make_dataset_layout_flat.py --src LEVIR-CD256 --out LEVIR-CD256-out --splits test
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

SPLIT_RE = re.compile(r"^(train|val|test)_")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="folder containing A/ B/ label/")
    p.add_argument("--out", required=True, help="output dataroot for DiffRegCD")
    p.add_argument("--a-name", default="A", help="pre-change folder name")
    p.add_argument("--b-name", default="B", help="post-change folder name")
    p.add_argument("--label-name", default="label", help="mask folder name")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   choices=["train", "val", "test"],
                   help="which splits to materialise (default: all three; "
                        "pass '--splits test' if you only need evaluation)")
    args = p.parse_args()

    src, out = Path(args.src), Path(args.out)
    a_dir, b_dir, m_dir = src / args.a_name, src / args.b_name, src / args.label_name
    for d in ("train/t0", "train/mask", "srcA", "list"):
        (out / d).mkdir(parents=True, exist_ok=True)

    wanted = set(args.splits)
    per_split: dict[str, list[str]] = {s: [] for s in ("train", "val", "test")}
    skipped_no_pair = 0

    for m in sorted(m_dir.iterdir()):
        if m.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif"):
            continue
        match = SPLIT_RE.match(m.stem)
        if not match:
            continue                      # unrecognised prefix, ignore
        split = match.group(1)
        if split not in wanted:
            continue
        a, b = a_dir / m.name, b_dir / m.name
        if not (a.exists() and b.exists()):
            skipped_no_pair += 1
            continue
        shutil.copy2(b, out / "train" / "t0" / m.name)
        shutil.copy2(a, out / "srcA" / m.name)
        shutil.copy2(m, out / "train" / "mask" / m.name)
        per_split[split].append(m.stem)

    for split, names in per_split.items():
        (out / "list" / f"{split}.txt").write_text("\n".join(names))
        if names:
            print(f"[ok] {split}: {len(names)} samples")
        elif split in wanted:
            print(f"[warn] {split}: 0 samples found")

    if skipped_no_pair:
        print(f"[warn] {skipped_no_pair} masks skipped (missing A or B pair)")

    total = sum(len(v) for v in per_split.values())
    print(f"\nDone — {total} samples copied. Next step:\n"
          f"  python scripts/prepare_gt_flow.py --src {out}/srcA "
          f"--warped-out {out}/train/t1 --flow-out {out}/train/flow --seed 0")


if __name__ == "__main__":
    main()
