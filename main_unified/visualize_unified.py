"""Render unified-model anomaly heatmaps against GT masks for a few anomalous
test images. Optionally overlays a standalone STFPM baseline if a matching
teacher/student pair is provided (--baseline_student + --baseline_teacher).

Usage:
    python -m main_scripts.visualize_unified --category hazelnut --device cuda \
        --dataframe_path cbm_data/mvtec/hazelnut_dataset_automated.csv \
        --model_path cbm_models/mvtec/unified/hazelnut/mobilenet_v2.pth \
        --save_dir plots/unified --n 6
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import pandas as pd
from datasets.concept_dataset import ConceptDataset
from main_unified.models_unified import unified_model


def unnormalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img


def unified_heatmap(model, image, device):
    with torch.no_grad():
        t_features, s_features, _, _ = model(image.unsqueeze(0).to(device))
    score_map = 1.0
    for j in range(len(t_features)):
        tn = F.normalize(t_features[j], dim=1)
        sn = F.normalize(s_features[j], dim=1)
        sm = torch.sum((tn - sn) ** 2, dim=1, keepdim=True)
        sm = F.interpolate(sm, size=(64, 64), mode="bilinear", align_corners=False)
        score_map = score_map * sm
    sm = score_map.squeeze().cpu().numpy()
    return (sm - sm.min()) / (sm.max() - sm.min() + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--dataframe_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone", type=str, default="mobilenet_v2")
    parser.add_argument("--save_dir", type=str, default="plots/unified")
    parser.add_argument("--n", type=int, default=6, help="number of anomalous images")
    parser.add_argument("--inject_diffs", action="store_true", help="unified++ checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    df = pd.read_csv(args.dataframe_path)
    dataset = ConceptDataset(df, split="test", use_attr=False, load_mask=True)

    num_attr = len(dataset.attr_cols)
    state_dict = torch.load(args.model_path, map_location="cpu")
    model = unified_model(num_attr=num_attr, backbone=args.backbone,
                          model_state_dict=state_dict, mode="test",
                          inject_diffs=args.inject_diffs).to(device)
    model.eval()

    # pick anomalous images, spread across anomaly types
    anom = dataset.df[dataset.df["label_index"] == 1]
    if "anomaly_type" in anom.columns:
        anom = anom.groupby("anomaly_type", group_keys=False).apply(
            lambda x: x.sample(min(2, len(x)), random_state=args.seed)
        )
    idxs = anom.sample(min(args.n, len(anom)), random_state=args.seed).index.tolist()

    n = len(idxs)
    fig, axs = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axs = axs[None, :]

    for r, idx in enumerate(idxs):
        image, label, mask = dataset[idx]
        hm = unified_heatmap(model, image, device)
        size = image.shape[-2:]
        hm_r = F.interpolate(torch.tensor(hm)[None, None], size=size,
                             mode="bilinear", align_corners=False).squeeze().numpy()

        inp = unnormalize(image).permute(1, 2, 0).numpy()
        inp = np.clip(inp, 0, 1)
        # binarize mask (>0: VisA encodes anomaly as ~5/255) and resize to the image
        # size so the overlay aligns (masks are stored at original resolution).
        mask_bin = (mask.numpy().squeeze() > 0).astype(np.float32)
        mask_r = F.interpolate(torch.tensor(mask_bin)[None, None], size=size,
                               mode="nearest").squeeze().numpy()
        gt = gaussian_filter(mask_r, sigma=2)

        axs[r, 0].imshow(inp); axs[r, 0].set_title("Input")
        axs[r, 1].imshow(inp); axs[r, 1].imshow(gt, cmap="jet", alpha=0.5); axs[r, 1].set_title("GT mask")
        axs[r, 2].imshow(inp); axs[r, 2].imshow(hm_r, cmap="jet", alpha=0.5); axs[r, 2].set_title("Unified heatmap")
        for c in range(3):
            axs[r, c].axis("off")

    os.makedirs(args.save_dir, exist_ok=True)
    out = os.path.join(args.save_dir, f"{args.category}_unified_heatmaps.png")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
