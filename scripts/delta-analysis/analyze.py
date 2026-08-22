# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Evaluate cheap sampled stats as predictors of Delta / Delta-of-Delta benefit."""

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
df = pd.read_csv("results.csv")


def section(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


section("1. Corpus")
one = df[df["sample"] == "1k"]
print(
    one.groupby("dataset").agg(
        columns=("column", "nunique"), blocks=("block", "count")
    ).to_string()
)
print(f"\ntotal (column, block) units: {len(one)}, distinct columns: "
      f"{one.groupby(['dataset', 'column']).ngroups}")

section("2. Is Delta-of-Delta ever worth it? (ground truth, full blocks)")
print(f"blocks where DoD beats Delta:  {(one.t_dod_bpv < one.t_delta_bpv).mean():.4f}")
print(f"blocks where DoD beats plain:  {(one.t_dod_bpv < one.t_plain_bpv).mean():.4f}")
print(f"blocks where Delta beats plain:{(one.t_delta_bpv < one.t_plain_bpv).mean():.4f}")
wd = one.t_delta_w - one.t_dod_w
print("\nbit-width saved by the second delta layer (delta_w - dod_w):")
print(wd.value_counts().sort_index().tail(6).to_string())
print(f"max width saving over the whole corpus: {wd.max()} bits "
      f"(a delta layer costs 1 bit/value of bases)")

section("3. Where does Delta win, and by how much?")
g = one.groupby(["dataset", "column"]).agg(
    bw=("byte_width", "first"),
    plain=("t_plain_bpv", "mean"),
    delta=("t_delta_bpv", "mean"),
    dod=("t_dod_bpv", "mean"),
    delta_w=("t_delta_w", "mean"),
    nonneg=("s_nonneg_frac", "mean"),
    zero=("s_delta_zero_frac", "mean"),
).reset_index()
g["ratio"] = g.plain / g.delta
print(g.nlargest(15, "ratio").to_string())
print(f"\ncolumns where delta wins: {(g.ratio > 1).sum()} / {len(g)}")

section("4. Sampled predictor accuracy (bit width, delta residuals)")
for tag in ["1k", "4k", "4k_wide", "8k_wide"]:
    s = df[df["sample"] == tag]
    err = s.s_delta_w - s.t_delta_w
    err_p = s.s_plain_w - s.t_plain_w
    print(
        f"{tag:8s} n={len(s):5d}  delta_w err: mean {err.mean():+.2f} "
        f"median {err.median():+.1f} p10 {err.quantile(.1):+.1f} p90 {err.quantile(.9):+.1f} "
        f"|within 1 bit| {(err.abs() <= 1).mean():.3f}   ||  "
        f"plain_w err: mean {err_p.mean():+.2f} |within 1 bit| {(err_p.abs() <= 1).mean():.3f}"
    )
print("\n(delta residuals are a LOCAL property -> sampling is nearly unbiased;")
print(" the value range is a GLOBAL property -> sampling under-estimates plain FoR width)")

section("5. Decision quality: pick Delta when predicted ratio > threshold")
one = df[df["sample"] == "1k"].copy()
one["true_best"] = np.where(one.t_delta_bpv < one.t_plain_bpv, "delta", "plain")
best = np.minimum(one.t_delta_bpv, one.t_plain_bpv)
print("regret = bytes/value paid above the better of {plain, delta}; "
      f"corpus mean best size {best.mean():.3f} B/value")
for thresh in [1.0, 1.05, 1.1, 1.25, 1.5, 2.0]:
    pred = np.where(one.s_plain_bpv / one.s_delta_bpv > thresh, "delta", "plain")
    chosen = np.where(pred == "delta", one.t_delta_bpv, one.t_plain_bpv)
    regret = chosen - best
    tp = ((pred == "delta") & (one.true_best == "delta")).sum()
    fp = ((pred == "delta") & (one.true_best == "plain")).sum()
    fn = ((pred == "plain") & (one.true_best == "delta")).sum()
    print(
        f"threshold {thresh:4.2f}: picks delta {(pred == 'delta').mean():.3f}  "
        f"precision {tp / max(tp + fp, 1):.3f} recall {tp / max(tp + fn, 1):.3f}  "
        f"mean regret {regret.mean():.4f} B/value  p99 {np.quantile(regret, .99):.3f}  "
        f"total corpus size {chosen.mean():.4f} B/value"
    )

section("6. Cheaper predictors than the 65-bucket histogram")
one = df[df["sample"] == "1k"].copy()
print("Rules evaluated on the same sample, compared against the true best choice:")


def score(name, pred):
    chosen = np.where(pred, one.t_delta_bpv, one.t_plain_bpv)
    best = np.minimum(one.t_delta_bpv, one.t_plain_bpv)
    regret = chosen - best
    truth = one.t_delta_bpv < one.t_plain_bpv
    tp = (pred & truth).sum()
    fp = (pred & ~truth).sum()
    fn = (~pred & truth).sum()
    print(
        f"  {name:52s} precision {tp / max(tp + fp, 1):.3f} recall {tp / max(tp + fn, 1):.3f} "
        f"mean regret {regret.mean():.4f}"
    )


score("histogram cost ratio > 1.25", (one.s_plain_bpv / one.s_delta_bpv > 1.25).values)
score("delta_w + 1 < plain_w (widths only)", (one.s_delta_w + 1 < one.s_plain_w).values)
score("delta_w + 1 < plain_w AND nonneg > 0.9",
      ((one.s_delta_w + 1 < one.s_plain_w) & (one.s_nonneg_frac > 0.9)).values)
score("delta zero-fraction > 0.5", (one.s_delta_zero_frac > 0.5).values)
score("nonneg fraction == 1.0 (sorted sample)", (one.s_nonneg_frac >= 0.999).values)

section("7. What the sample says about DoD (would a DoD rule ever fire?)")
for tag in ["1k", "4k"]:
    s = df[df["sample"] == tag]
    fires = (s.s_dod_bpv < s.s_delta_bpv).mean()
    print(f"{tag}: sampled DoD predicted better than sampled Delta in {fires:.4f} of blocks; "
          f"ground truth {(s.t_dod_bpv < s.t_delta_bpv).mean():.4f}")

section("8. Separating true delta structure from run-length structure")
one = df[df["sample"] == "1k"].copy()
runheavy = one.s_delta_zero_frac > 0.5
for name, sub in [("run-heavy sample (>50% zero deltas)", one[runheavy]),
                  ("genuinely varying values", one[~runheavy])]:
    print(f"\n{name}: n={len(sub)}")
    print(f"  delta beats plain in {(sub.t_delta_bpv < sub.t_plain_bpv).mean():.3f} of blocks; "
          f"median ratio {(sub.t_plain_bpv / sub.t_delta_bpv).median():.2f}")
    print(f"  sampled width within 1 bit: {((sub.s_delta_w - sub.t_delta_w).abs() <= 1).mean():.3f}")

section("9. Why the second delta layer cannot pay: noise doubling")
one = df[df["sample"] == "1k"]
print("second differencing of iid residuals doubles the span (+1 bit) and adds another")
print("1 bit/value of bases; it only wins if the delta RATE drifts across the block.")
print("\nsampled dod_w - delta_w over the corpus:")
print((one.s_dod_w - one.s_delta_w).value_counts().sort_index().to_string())
print("\nground-truth dod_bpv - delta_bpv (bytes/value):")
print((one.t_dod_bpv - one.t_delta_bpv).describe()[["mean", "min", "25%", "50%", "75%", "max"]].to_string())

section("10. Per-dataset predictor accuracy (1k sample)")
one = df[df["sample"] == "1k"].copy()
one["within1"] = (one.s_delta_w - one.t_delta_w).abs() <= 1
one["pick"] = one.s_plain_bpv / one.s_delta_bpv > 1.0
one["truth"] = one.t_delta_bpv < one.t_plain_bpv
print(one.groupby("dataset").apply(
    lambda s: pd.Series({
        "blocks": len(s),
        "delta_wins": s.truth.mean(),
        "width_within_1bit": s.within1.mean(),
        "decision_agrees": (s.pick == s.truth).mean(),
        "mean_regret_B": (np.where(s.pick, s.t_delta_bpv, s.t_plain_bpv)
                          - np.minimum(s.t_delta_bpv, s.t_plain_bpv)).mean(),
    }), include_groups=False).to_string())
