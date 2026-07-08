"""Watch two unified result CSVs and write a before/after/diff markdown table.

before = unified++ (results/unified_pp_results.csv)
after  = unified++masked (results/unified_ppmasked_results.csv)
diff   = after - before   (positive => masking the student helped)

Focus metrics: CBM I-AUC (primary), P-AUC (heatmap pixel), STFPM I-AUC (heatmap image).

Usage:
    python -m main_scripts.compare_ablation           # one-shot
    python -m main_scripts.compare_ablation --watch    # regenerate on file change,
                                                        # auto-stop when 'after' is complete
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

METRICS = [("cbm_image_auc", "CBM I-AUC"),
           ("pixel_auc", "P-AUC"),
           ("stfpm_image_auc", "STFPM I-AUC")]


def fmt(x):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def fmt_delta(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    arrow = "▲" if x > 0.0005 else ("▼" if x < -0.0005 else "·")
    return f"{arrow} {x:+.3f}"


def build_table(before_csv, after_csv, out_md):
    if not (os.path.exists(before_csv) and os.path.exists(after_csv)):
        return 0, 0
    b = pd.read_csv(before_csv)
    a = pd.read_csv(after_csv)
    key = ["dataset", "category"]
    m = b.merge(a, on=key, suffixes=("_before", "_after"))
    if m.empty:
        return 0, len(a)
    m = m.sort_values(key).reset_index(drop=True)

    # header
    cols = ["dataset", "category"]
    for _, name in METRICS:
        cols += [f"{name} before", "after", "Δ"]
    header = "| " + " | ".join(cols) + " |\n"
    header += "|" + "|".join(["---"] * len(cols)) + "|\n"

    def row_for(r):
        cells = [str(r["dataset"]), str(r["category"])]
        for col, _ in METRICS:
            bv, av = r.get(f"{col}_before"), r.get(f"{col}_after")
            d = (av - bv) if pd.notna(bv) and pd.notna(av) else np.nan
            cells += [fmt(bv), fmt(av), fmt_delta(d)]
        return "| " + " | ".join(cells) + " |"

    lines = [row_for(r) for _, r in m.iterrows()]

    # per-dataset + overall mean rows
    def mean_row(sub, label):
        cells = [f"**{label}**", "**mean**"]
        for col, _ in METRICS:
            bv = sub[f"{col}_before"].mean()
            av = sub[f"{col}_after"].mean()
            d = av - bv
            cells += [f"**{fmt(bv)}**", f"**{fmt(av)}**", f"**{fmt_delta(d)}**"]
        return "| " + " | ".join(cells) + " |"

    body = []
    for ds, sub in m.groupby("dataset"):
        for _, r in sub.iterrows():
            body.append(row_for(r))
        body.append(mean_row(sub, ds))
    body.append(mean_row(m, "ALL"))

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    note = (f"# unified++ vs unified++masked\n\n"
            f"**before** = unified++ · **after** = unified++masked (student not updated "
            f"by anomalous samples) · **Δ = after − before** (positive => masking helped).\n\n"
            f"_Compared {len(m)} categories · updated {ts}_\n\n")
    with open(out_md, "w") as f:
        f.write(note + header + "\n".join(body) + "\n")
    return len(m), len(a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", default="results/unified_pp_results.csv")
    p.add_argument("--after", default="results/unified_ppmasked_results.csv")
    p.add_argument("--out", default="results/compare_pp_vs_ppmasked.md")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=int, default=15)
    args = p.parse_args()

    def mtimes():
        return tuple(os.path.getmtime(f) if os.path.exists(f) else 0
                     for f in (args.before, args.after))

    last = None
    stable = 0
    while True:
        cur = mtimes()
        if cur != last:
            n, n_after = build_table(args.before, args.after, args.out)
            print(f"[{time.strftime('%H:%M:%S')}] compared {n} cats "
                  f"(after has {n_after} rows) -> {args.out}")
            last = cur
            stable = 0
        else:
            stable += 1
        if not args.watch:
            break
        # auto-stop: after-file has reached before-file's category count and is stable
        try:
            n_before = len(pd.read_csv(args.before)) if os.path.exists(args.before) else 0
            n_after = len(pd.read_csv(args.after)) if os.path.exists(args.after) else 0
        except Exception:
            n_before, n_after = 0, 0
        if n_after >= n_before > 0 and stable >= 2:
            print(f"[{time.strftime('%H:%M:%S')}] after-run complete "
                  f"({n_after}/{n_before}) — stopping watcher.")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
