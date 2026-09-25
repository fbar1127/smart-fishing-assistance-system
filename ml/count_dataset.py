"""
IsdaYou - FishImgDataset audit and cleaning script
==================================================

Produces the numbers needed for Manuscript Sections 6.3 (Data Volume),
6.4 (Data Quality), 7.1 (Data Cleaning) and 7.5 (Dataset Splitting),
and optionally writes a cleaned copy of the dataset for training.

Usage (Windows, from Command Prompt or PowerShell):
    pip install pillow
    pip install imagehash          (optional, enables near-duplicate check)

    python count_dataset.py
    python count_dataset.py --root "C:\\Users\\acer\\OneDrive\\Desktop\\All FIles\\SoftEng\\FishImgDataset"
    python count_dataset.py --clean            (also writes FishImgDataset_clean next to the original)
    python count_dataset.py --clean --resplit  (removes duplicates across ALL splits, then makes a new
                                                stratified 70/15/15 split; use this when the audit
                                                reports duplicates across splits)

IMPORTANT: the dataset lives in OneDrive. If files show "Sync pending" or are
online-only, right-click the FishImgDataset folder -> "Always keep on this
device" and wait for the download to finish first, otherwise online-only
files may be reported as unreadable.

The original dataset is never modified. All outputs go to ./audit_output/.
"""

import argparse
import csv
import hashlib
import os
import shutil
import statistics
import sys
from collections import Counter, defaultdict

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    sys.exit("Pillow is required:  pip install pillow")

try:
    import imagehash  # optional
except ImportError:
    imagehash = None

DEFAULT_ROOT = r"C:\Users\acer\OneDrive\Desktop\All FIles\SoftEng\FishImgDataset"
SPLITS = ["train", "val", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
EXT_TO_FORMAT = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "BMP",
                 ".webp": "WEBP", ".gif": "GIF", ".tif": "TIFF", ".tiff": "TIFF"}
MIN_SIDE = 64          # images smaller than this on either side are flagged
MAX_ASPECT = 3.0       # width/height ratio beyond this is flagged
PHASH_DISTANCE = 4     # Hamming distance treated as near-duplicate


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def inspect(path):
    """Return a dict describing one file."""
    info = {"path": path, "ext": os.path.splitext(path)[1].lower(), "bytes": 0,
            "readable": False, "format": "", "mode": "", "width": 0, "height": 0,
            "phash": None, "error": ""}
    try:
        info["bytes"] = os.path.getsize(path)
        if info["bytes"] == 0:
            info["error"] = "zero-byte file"
            return info
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:           # re-open: verify() invalidates
            im.load()
            info.update(readable=True, format=im.format or "", mode=im.mode,
                        width=im.width, height=im.height)
            if imagehash is not None:
                info["phash"] = imagehash.phash(im.convert("RGB"))
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def main():
    ap = argparse.ArgumentParser(description="Audit (and optionally clean) FishImgDataset")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default="audit_output")
    ap.add_argument("--clean", action="store_true",
                    help="write a cleaned RGB-JPEG copy to <root>_clean")
    ap.add_argument("--resplit", action="store_true",
                    help="with --clean: deduplicate across all splits and create a new stratified 70/15/15 split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit(f"Dataset folder not found: {root}")
    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------ scan
    classes = sorted({c for s in SPLITS if os.path.isdir(os.path.join(root, s))
                      for c in os.listdir(os.path.join(root, s))
                      if os.path.isdir(os.path.join(root, s, c))})
    records, non_image = [], []
    for split in SPLITS:
        for cls in classes:
            d = os.path.join(root, split, cls)
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                p = os.path.join(d, name)
                if not os.path.isfile(p):
                    continue
                ext = os.path.splitext(name)[1].lower()
                if ext not in IMAGE_EXTS:
                    non_image.append((split, cls, name))
                    continue
                r = inspect(p)
                r.update(split=split, cls=cls, name=name)
                records.append(r)
        print(f"  scanned {split}")

    # ------------------------------------------------------------ flags
    flags = defaultdict(list)  # path -> list of issue strings
    for r in records:
        if not r["readable"]:
            flags[r["path"]].append("unreadable/corrupt: " + r["error"])
            continue
        expected = EXT_TO_FORMAT.get(r["ext"])
        if expected and r["format"] and r["format"] != expected:
            flags[r["path"]].append(f"extension {r['ext']} but encoded as {r['format']}")
        if r["mode"] != "RGB":
            flags[r["path"]].append(f"colour mode {r['mode']} (not RGB)")
        if min(r["width"], r["height"]) < MIN_SIDE:
            flags[r["path"]].append(f"very small ({r['width']}x{r['height']})")
        ar = max(r["width"], r["height"]) / max(1, min(r["width"], r["height"]))
        if ar > MAX_ASPECT:
            flags[r["path"]].append(f"extreme aspect ratio {ar:.1f}:1")

    # exact duplicates
    by_hash = defaultdict(list)
    for r in records:
        if r["readable"]:
            r["md5"] = md5(r["path"])
            by_hash[r["md5"]].append(r)
    dup_groups = [g for g in by_hash.values() if len(g) > 1]
    within_class = cross_class = cross_split = 0
    for g in dup_groups:
        labels = {x["cls"] for x in g}
        splits = {x["split"] for x in g}
        if len(labels) > 1:
            cross_class += 1
            for x in g:
                flags[x["path"]].append("exact duplicate with CONFLICTING labels: "
                                        + ", ".join(sorted(labels)))
        if len(splits) > 1:
            cross_split += 1
            for x in g:
                if x["split"] != "train":
                    flags[x["path"]].append("exact duplicate of an image in another split (leakage)")
        if len(labels) == 1 and len(splits) == 1:
            within_class += 1
            for x in g[1:]:
                flags[x["path"]].append("exact duplicate within same class/split")

    # near duplicates (optional, vectorised within class)
    near_pairs = 0
    near_edges = []  # (record_a, record_b)
    if imagehash is not None:
        import numpy as np
        by_cls = defaultdict(list)
        for r in records:
            if r.get("phash") is not None:
                r["phash_int"] = int(str(r["phash"]), 16)
                by_cls[r["cls"]].append(r)
        for cls, items in by_cls.items():
            h = np.array([x["phash_int"] for x in items], dtype=np.uint64)
            x = np.bitwise_xor(h[:, None], h[None, :])
            dist = np.unpackbits(x.view(np.uint8).reshape(len(h), len(h), 8), axis=2).sum(2)
            ii, jj = np.where(np.triu(dist <= PHASH_DISTANCE, k=1))
            for i, j in zip(ii, jj):
                a, b = items[i], items[j]
                if a.get("md5") == b.get("md5"):
                    continue
                near_pairs += 1
                near_edges.append((a, b))
                if a["split"] != b["split"]:
                    tgt = b if b["split"] != "train" else a
                    flags[tgt["path"]].append("near-duplicate of image in another split: "
                                              + os.path.basename((a if tgt is b else b)["path"]))

    # ------------------------------------------------------------ tables
    counts = Counter((r["split"], r["cls"]) for r in records)
    split_tot = Counter(r["split"] for r in records)
    total = sum(split_tot.values())

    with open(os.path.join(args.out, "class_counts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Index", "Class", "Train", "Validation", "Test", "Total", "Train %", "Val %", "Test %"])
        for i, c in enumerate(classes):
            t, v, te = counts[("train", c)], counts[("val", c)], counts[("test", c)]
            s = t + v + te
            pct = lambda n: f"{100 * n / s:.1f}" if s else "0.0"
            w.writerow([i, c, t, v, te, s, pct(t), pct(v), pct(te)])
        w.writerow(["", "TOTAL", split_tot["train"], split_tot["val"], split_tot["test"], total,
                    *(f"{100 * split_tot[s] / total:.1f}" if total else "0.0" for s in SPLITS)])

    with open(os.path.join(args.out, "flagged_files.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Split", "Class", "File", "Issues"])
        for r in records:
            if r["path"] in flags:
                w.writerow([r["split"], r["cls"], r["name"], " | ".join(flags[r["path"]])])
        for split, cls, name in non_image:
            w.writerow([split, cls, name, "non-image file"])

    # ------------------------------------------------------------ summary
    readable = [r for r in records if r["readable"]]
    widths = [r["width"] for r in readable]
    heights = [r["height"] for r in readable]
    train_counts = [counts[("train", c)] for c in classes]
    issue_counter = Counter()
    for issues in flags.values():
        for i in issues:
            issue_counter[i.split(":")[0].split(" (")[0].split(" with")[0]] += 1

    lines = []
    lines.append("# FishImgDataset audit summary\n")
    lines.append(f"Root: {root}\n")
    lines.append(f"- Classes: {len(classes)}")
    lines.append(f"- Image files: {total}  (train {split_tot['train']}, val {split_tot['val']}, test {split_tot['test']})")
    if total:
        lines.append("- Split proportions: " + ", ".join(
            f"{s} {100 * split_tot[s] / total:.1f}%" for s in SPLITS))
    lines.append(f"- File formats: {dict(Counter(r['ext'] for r in records))}")
    lines.append(f"- Colour modes: {dict(Counter(r['mode'] for r in readable))}")
    if readable:
        lines.append(f"- Width  px: min {min(widths)}, median {statistics.median(widths):.0f}, max {max(widths)}")
        lines.append(f"- Height px: min {min(heights)}, median {statistics.median(heights):.0f}, max {max(heights)}")
    if train_counts and min(train_counts) > 0:
        mx, mn = max(train_counts), min(train_counts)
        lines.append(f"- Train imbalance: largest class {classes[train_counts.index(mx)]} ({mx}), "
                     f"smallest {classes[train_counts.index(mn)]} ({mn}), ratio {mx / mn:.2f}:1")
    lines.append(f"- Unreadable/corrupt files: {sum(1 for r in records if not r['readable'])}")
    lines.append(f"- Non-image files in class folders: {len(non_image)}")
    lines.append(f"- Exact-duplicate groups: {len(dup_groups)} "
                 f"(same class+split {within_class}, across splits {cross_split}, conflicting labels {cross_class})")
    lines.append(f"- Near-duplicate pairs (pHash<= {PHASH_DISTANCE}): "
                 + (str(near_pairs) if imagehash else "not checked (pip install imagehash)"))
    lines.append(f"- Files with at least one flag: {len(flags) + len(non_image)}")
    lines.append("- Flag breakdown: " + ", ".join(f"{k}: {v}" for k, v in issue_counter.most_common()))
    lines.append("")
    lines.append("| # | Class | Train | Val | Test | Total |")
    lines.append("|---|---|---|---|---|---|")
    for i, c in enumerate(classes):
        t, v, te = counts[("train", c)], counts[("val", c)], counts[("test", c)]
        lines.append(f"| {i} | {c} | {t} | {v} | {te} | {t + v + te} |")
    lines.append(f"| | **Total** | {split_tot['train']} | {split_tot['val']} | {split_tot['test']} | {total} |")
    summary = "\n".join(lines)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary)
    with open(os.path.join(args.out, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(classes))
    print("\n" + summary)
    print(f"\nWrote {args.out}/summary.md, class_counts.csv, flagged_files.csv, labels.txt")

    # ------------------------------------------------------------ clean copy
    if args.clean and args.resplit:
        resplit(records, non_image, classes, flags, near_edges, root, args.out, args.seed)
    elif args.clean:
        out_root = root.rstrip("\\/") + "_clean"
        log_path = os.path.join(args.out, "cleaning_log.csv")
        drop_reasons = {}
        for r in records:
            reasons = [i for i in flags.get(r["path"], [])
                       if i.startswith(("unreadable", "exact duplicate", "near-duplicate"))]
            if reasons:
                drop_reasons[r["path"]] = reasons
        kept = dropped = converted = 0
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Split", "Class", "File", "Action", "Reason"])
            for r in records:
                if r["path"] in drop_reasons:
                    dropped += 1
                    w.writerow([r["split"], r["cls"], r["name"], "removed", " | ".join(drop_reasons[r["path"]])])
                    continue
                dst_dir = os.path.join(out_root, r["split"], r["cls"])
                os.makedirs(dst_dir, exist_ok=True)
                base = os.path.splitext(r["name"])[0]
                dst = os.path.join(dst_dir, base + ".jpg")
                n = 1
                while os.path.exists(dst):
                    dst = os.path.join(dst_dir, f"{base}_{n}.jpg"); n += 1
                if r["format"] == "JPEG" and r["mode"] == "RGB":
                    shutil.copy2(r["path"], dst)
                else:
                    with Image.open(r["path"]) as im:
                        im = im.convert("RGBA") if im.mode in ("P", "LA") else im
                        if im.mode == "RGBA":            # flatten transparency on white
                            bg = Image.new("RGB", im.size, (255, 255, 255))
                            bg.paste(im, mask=im.split()[-1])
                            im = bg
                        im.convert("RGB").save(dst, "JPEG", quality=95)
                    converted += 1
                    w.writerow([r["split"], r["cls"], r["name"], "converted to RGB JPEG",
                                f"{r['format']}/{r['mode']}"])
                kept += 1
            for split, cls, name in non_image:
                w.writerow([split, cls, name, "removed", "non-image file"])
        print(f"\nClean copy: {out_root}\n  kept {kept}, removed {dropped + len(non_image)}, "
              f"converted {converted}\n  log: {log_path}")


def save_rgb_jpeg(r, dst):
    if r["format"] == "JPEG" and r["mode"] == "RGB":
        shutil.copy2(r["path"], dst)
        return False
    with Image.open(r["path"]) as im:
        im.seek(0)
        im = im.convert("RGBA") if im.mode in ("P", "LA") else im
        if im.mode == "RGBA":
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        im.convert("RGB").save(dst, "JPEG", quality=95)
    return True


def resplit(records, non_image, classes, flags, near_edges, root, out_dir, seed,
            ratios=(0.70, 0.15, 0.15)):
    """Deduplicate across the whole dataset and write a new stratified split.

    Exact duplicates (same MD5) and, when imagehash is installed, near-duplicates
    (pHash distance <= PHASH_DISTANCE) are merged into clusters with union-find.
    Exact duplicates are reduced to one copy. Near-duplicates (e.g. resized or
    re-encoded versions) are kept, but every group of near-duplicates is placed in
    the same split, so no photo or variant of it can appear in two splits.
    """
    import random
    out_root = root.rstrip("\\/") + "_clean"
    if os.path.exists(out_root):
        sys.exit(f"{out_root} already exists - delete it first (e.g. !rm -rf \"{out_root}\")")
    log_path = os.path.join(out_dir, "cleaning_log.csv")
    log = []

    usable = []
    for r in records:
        if not r["readable"]:
            log.append([r["split"], r["cls"], r["name"], "removed", "unreadable/corrupt"])
        elif min(r["width"], r["height"]) < MIN_SIDE:
            log.append([r["split"], r["cls"], r["name"], "removed",
                        f"too small to use ({r['width']}x{r['height']})"])
        else:
            usable.append(r)
    for split, cls, name in non_image:
        log.append([split, cls, name, "removed", "non-image file"])

    # union-find over usable records
    idx = {id(r): i for i, r in enumerate(usable)}
    parent = list(range(len(usable)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_md5 = defaultdict(list)
    for i, r in enumerate(usable):
        by_md5[r["md5"]].append(i)
    # 1) exact duplicates: keep one copy; drop images stored under two species
    conflict = set()
    kept = []
    exact_removed = 0
    for members in by_md5.values():
        if len({usable[i]["cls"] for i in members}) > 1:
            conflict.update(members)
            for i in members:
                r = usable[i]
                log.append([r["split"], r["cls"], r["name"], "removed", "same image under different species"])
            continue
        members.sort(key=lambda i: (usable[i]["split"] != "train", usable[i]["name"]))
        kept.append(members[0])
        for i in members[1:]:
            r, k = usable[i], usable[members[0]]
            exact_removed += 1
            log.append([r["split"], r["cls"], r["name"], "removed",
                        f"exact duplicate of {k['split']}/{k['cls']}/{k['name']}"])
    kept_set = set(kept)
    rep_of = {}
    for members in by_md5.values():
        for i in members:
            rep_of[i] = members[0] if not conflict.intersection(members) else None
    # 2) near duplicates: group (union-find) so each group goes to one split
    for a, b in near_edges:
        ia, ib = rep_of.get(idx.get(id(a))), rep_of.get(idx.get(id(b)))
        if ia is not None and ib is not None and ia in kept_set and ib in kept_set \
                and usable[ia]["cls"] == usable[ib]["cls"]:
            union(ia, ib)
    groups = defaultdict(list)
    for i in kept:
        groups[find(i)].append(usable[i])
    groups_by_class = defaultdict(list)
    for g in groups.values():
        groups_by_class[g[0]["cls"]].append(sorted(g, key=lambda r: (r["split"], r["name"])))
    near_grouped = sum(len(g) for g in groups.values() if len(g) > 1)
    near_removed = 0

    rng = random.Random(seed)
    new_counts = {}
    converted = 0
    for cls in classes:
        glist = sorted(groups_by_class.get(cls, []), key=lambda g: (g[0]["split"], g[0]["name"]))
        rng.shuffle(glist)
        n = sum(len(g) for g in glist)
        target = {"val": round(n * ratios[1]), "test": round(n * ratios[2])}
        parts = {"train": [], "val": [], "test": []}
        for g in glist:            # greedy fill; large near-duplicate groups end up in train
            for split in ("test", "val"):
                if len(parts[split]) + len(g) <= target[split]:
                    parts[split].extend(g)
                    break
            else:
                parts["train"].extend(g)
        new_counts[cls] = {k: len(v) for k, v in parts.items()}
        for split, rs in parts.items():
            d = os.path.join(out_root, split, cls)
            os.makedirs(d, exist_ok=True)
            used = set()
            for r in rs:
                base = os.path.splitext(r["name"])[0]
                name, k = base + ".jpg", 1
                while name.lower() in used:
                    name = f"{base}_{k}.jpg"; k += 1
                used.add(name.lower())
                if save_rgb_jpeg(r, os.path.join(d, name)):
                    converted += 1
                    log.append([r["split"], r["cls"], r["name"], f"converted to RGB JPEG -> {split}",
                                f"{r['format']}/{r['mode']}"])

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Original split", "Class", "File", "Action", "Reason"])
        w.writerows(log)
    with open(os.path.join(out_dir, "new_split_counts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Index", "Class", "Train", "Validation", "Test", "Total"])
        for i, c in enumerate(classes):
            t = new_counts[c]
            w.writerow([i, c, t["train"], t["val"], t["test"], sum(t.values())])

    tot = {s: sum(v[s] for v in new_counts.values()) for s in SPLITS}
    total = sum(tot.values())
    removed_small = sum(1 for row in log if row[4].startswith("too small"))
    lines = ["", "# Cleaned and re-split dataset", "",
             f"Output: {out_root}",
             f"- Images before cleaning: {len(records)} (+{len(non_image)} non-image files)",
             f"- Removed: unreadable {sum(1 for r in records if not r['readable'])}, "
             f"too small (<{MIN_SIDE}px) {removed_small}, non-image {len(non_image)}, "
             f"exact duplicates {exact_removed}, conflicting labels {len(conflict)}",
             f"- Near-duplicate images kept but grouped into the same split: {near_grouped}",
             f"- Converted to RGB JPEG: {converted}",
             f"- Unique images kept: {total}",
             f"- New split (seed {seed}): train {tot['train']} ({100 * tot['train'] / total:.1f}%), "
             f"val {tot['val']} ({100 * tot['val'] / total:.1f}%), test {tot['test']} ({100 * tot['test'] / total:.1f}%)",
             "", "| # | Class | Train | Val | Test | Total |", "|---|---|---|---|---|---|"]
    for i, c in enumerate(classes):
        t = new_counts[c]
        lines.append(f"| {i} | {c} | {t['train']} | {t['val']} | {t['test']} | {sum(t.values())} |")
    lines.append(f"| | **Total** | {tot['train']} | {tot['val']} | {tot['test']} | {total} |")
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "resplit_summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nWrote {out_dir}/resplit_summary.md, new_split_counts.csv, cleaning_log.csv")


if __name__ == "__main__":
    main()
