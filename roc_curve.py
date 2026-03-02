

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import model

W_SPATIAL = 0.45
W_TEXTURE = 0.30
W_MOTION  = 0.25
W_DEPTH   = 0.00
NUM_FRAMES = 16


def fuse(spatial: float, texture: float, motion: float, depth: float) -> float:
    is_image = motion == 0.0
    if is_image:
        active_weights = [w for w in [W_SPATIAL, W_TEXTURE, W_DEPTH] if w > 0.0]
        n_active = len(active_weights)
        extra = W_MOTION / n_active if n_active > 0 else 0.0
        score = (
            (W_SPATIAL + extra) * spatial +
            (W_TEXTURE + extra) * texture +
            (W_DEPTH   + extra) * depth
        )
    else:
        score = (
            W_SPATIAL * spatial +
            W_TEXTURE * texture +
            W_MOTION  * motion  +
            W_DEPTH   * depth
        )
    return float(np.clip(score, 0.0, 1.0))


def run(dataset_root: str, max_n: int, output_path: str = "roc_curve.png") -> None:
    samples = model.iter_dataset(dataset_root)[:max_n]
    print(f"Running inference on {len(samples)} samples...\n")

    scores = []
    labels = []
    errors = []

    for i, (path, label, folder) in enumerate(samples):
        try:
            spatial = model.spatial_stage(path, NUM_FRAMES)
            texture = model.texture_stage(path, NUM_FRAMES)
            motion  = model.motion_stage(path, NUM_FRAMES)
            depth   = model.depth_stage(path, 4)
            score   = fuse(spatial, texture, motion, depth)

            scores.append(score)
            labels.append(label)
            print(f"  [{i+1:>2}/{len(samples)}]  {folder:<22}  score={score:.4f}  label={label}")
        except Exception as e:
            errors.append((path, str(e)))
            print(f"  [{i+1:>2}/{len(samples)}]  ERROR: {folder} — {e}")

    if errors:
        print(f"\n{len(errors)} sample(s) failed and were skipped.")

    model.print_crop_stats()


#roc
    y_true  = np.array(labels)
    y_score = np.array(scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Youden's J statistic — maximises TPR - FPR
    # This gives the threshold that best balances sensitivity and specificity
    j_scores   = tpr - fpr
    best_idx   = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])
    best_tpr    = float(tpr[best_idx])
    best_fpr    = float(fpr[best_idx])

    print(f"\n{'='*50}")
    print(f"  ROC AUC:            {roc_auc:.4f}")
    print(f"  Optimal threshold:  {best_thresh:.4f}  (Youden's J)")
    print(f"    TPR at optimal:   {best_tpr:.4f}  (sensitivity / 1 - FRR)")
    print(f"    FPR at optimal:   {best_fpr:.4f}  (1 - specificity / FAR)")
    print(f"{'='*50}\n")
    print(f"Current thresholds in main.rs:")
    print(f"  ACCEPT_THRESHOLD = 0.55")
    print(f"  RETRY_THRESHOLD  = 0.38")
    print(f"\nIf AUC is near 0.5, the model is not separating live from spoof.")
    print(f"If AUC > 0.7, there is meaningful signal even with the stub backbone.")



    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr, tpr, color="#2563EB", lw=2,
            label=f"Fused score (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1, linestyle="--",
            label="Random classifier (AUC = 0.500)")

    # Mark optimal threshold point
    ax.scatter([best_fpr], [best_tpr], color="#DC2626", zorder=5, s=80,
               label=f"Optimal threshold = {best_thresh:.3f}")
    ax.annotate(
        f"  thresh={best_thresh:.3f}\n  TPR={best_tpr:.3f}, FPR={best_fpr:.3f}",
        xy=(best_fpr, best_tpr),
        xytext=(best_fpr + 0.08, best_tpr - 0.12),
        fontsize=8,
        color="#DC2626",
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=0.8),
    )

    accept_thresh = 0.55
    idx_accept = int(np.argmin(np.abs(thresholds - accept_thresh)))
    ax.scatter([fpr[idx_accept]], [tpr[idx_accept]], color="#16A34A",
               zorder=5, s=80, marker="s",
               label=f"Current accept threshold = {accept_thresh}")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate  (FAR — spoof accepted)", fontsize=11)
    ax.set_ylabel("True Positive Rate  (1 - FRR — live accepted)", fontsize=11)
    ax.set_title("ROC Curve — Multi-Stage Liveness Pipeline\n"
                 "ResNet18 stub backbone, face-crop preprocessing", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"ROC curve saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROC/AUC analysis for liveness pipeline")
    parser.add_argument("--data",   default="../data",      help="Dataset root directory")
    parser.add_argument("--max_n",  default=43, type=int,   help="Max samples to evaluate")
    parser.add_argument("--output", default="roc_curve.png", help="Output PNG path")
    args = parser.parse_args()

    run(args.data, args.max_n, args.output)
