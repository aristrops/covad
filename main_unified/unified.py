import argparse
import os
import torch
import pandas as pd

from datasets.concept_dataset import ConceptDataset
from main_unified.models_unified import unified_model
from main_unified.trainer_unified import UnifiedTrainer
from main_unified.evaluator_unified import UnifiedEvaluator


def make_dataloader(dataset, batch_size, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )


RESULT_COLS = ["dataset", "category", "cbm_image_auc", "stfpm_image_auc",
               "pixel_auc", "concept_auc", "concept_f1", "cbm_image_f1", "pixel_f1"]


def update_results_table(results_path, row):
    """Upsert a row keyed by (dataset, category) into a CSV and regenerate a
    markdown table next to it, so the report updates as categories finish."""
    import pandas as pd
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
    else:
        df = pd.DataFrame(columns=RESULT_COLS)
    df = df[~((df["dataset"] == row["dataset"]) & (df["category"] == row["category"]))]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values(["dataset", "category"]).reset_index(drop=True)
    df.to_csv(results_path, index=False)

    md_path = os.path.splitext(results_path)[0] + ".md"
    header = ("| dataset | category | CBM I-AUC | STFPM I-AUC | P-AUC | "
              "concept AUC | concept F1 |\n|---|---|---|---|---|---|---|\n")
    lines = []
    for ds, g in df.groupby("dataset"):
        for _, r in g.iterrows():
            lines.append(f"| {r['dataset']} | {r['category']} | {r['cbm_image_auc']:.3f} | "
                         f"{r['stfpm_image_auc']:.3f} | {r['pixel_auc']:.3f} | "
                         f"{r['concept_auc']:.3f} | {r['concept_f1']:.3f} |")
        m = g[["cbm_image_auc", "stfpm_image_auc", "pixel_auc", "concept_auc", "concept_f1"]].mean()
        lines.append(f"| **{ds}** | **mean** | **{m['cbm_image_auc']:.3f}** | "
                     f"**{m['stfpm_image_auc']:.3f}** | **{m['pixel_auc']:.3f}** | "
                     f"**{m['concept_auc']:.3f}** | **{m['concept_f1']:.3f}** |")
    with open(md_path, "w") as f:
        f.write("# Unified model results\n\n"
                "CBM I-AUC = final image-level anomaly score from the concept head "
                "(primary). STFPM I-AUC / P-AUC = teacher-student heatmap (localization).\n\n")
        f.write(header + "\n".join(lines) + "\n")


def train_model(category, dataframe_path, teacher_path, save_path, device, backbone,
                expand_dim, lambda_, batch_size, optimizer, lr, epochs, seed,
                inject_diffs=False, mask_student=False):

    def init_optimizer(params):
        if optimizer == "adam":
            opt = torch.optim.Adam(params, lr=lr)
        elif optimizer == "sgd":
            opt = torch.optim.SGD(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=5)
        return opt, scheduler

    dataframe = pd.read_csv(dataframe_path)

    # full train split (normal + anomalous), fully-supervised
    train_dataset = ConceptDataset(dataframe, split="train", use_attr=True, random_state=seed)
    val_dataset = ConceptDataset(dataframe, split="val", use_attr=True, random_state=seed)

    train_dataloader = make_dataloader(train_dataset, batch_size, shuffle=True)
    val_dataloader = make_dataloader(val_dataset, batch_size, shuffle=False)

    num_attr = len(train_dataset.attr_cols)
    weight_attr = train_dataset.find_class_imbalance("attributes")
    weight_main, _ = train_dataset.find_class_imbalance("main")

    print(f"\nTraining unified model for {category}: "
          f"{len(train_dataset)} train / {len(val_dataset)} val images, {num_attr} concepts")

    model = unified_model(num_attr=num_attr, expand_dim=expand_dim, backbone=backbone,
                          teacher_path=teacher_path, mode="train", inject_diffs=inject_diffs)
    model.to(device)

    # only student + concept net + main head are trainable (teacher frozen)
    params = [p for p in model.parameters() if p.requires_grad]
    opt, scheduler = init_optimizer(params)

    trainer = UnifiedTrainer(model, num_attr, train_dataloader, val_dataloader, opt, scheduler,
                             device, num_epochs=epochs, lambda_=lambda_,
                             weight_attr=weight_attr, weight_main=weight_main,
                             mask_student=mask_student, save_path=save_path)
    trainer.train()


def eval_model(category, dataframe_path, save_path, device, backbone, expand_dim, batch_size,
               dataset="mvtec", results_path=None, inject_diffs=False):
    dataframe = pd.read_csv(dataframe_path)

    test_dataset = ConceptDataset(dataframe, split="test", use_attr=True, load_mask=True)
    test_dataloader = make_dataloader(test_dataset, batch_size, shuffle=False)
    num_attr = len(test_dataset.attr_cols)
    attr_cols = test_dataset.attr_cols

    print(f"\nEvaluating unified model for {category}: {len(test_dataset)} test images")

    state_dict = torch.load(save_path, map_location="cpu")
    model = unified_model(num_attr=num_attr, expand_dim=expand_dim, backbone=backbone,
                          model_state_dict=state_dict, mode="test", inject_diffs=inject_diffs)
    model.to(device)

    evaluator = UnifiedEvaluator(model, num_attr, attr_cols, test_dataloader, device)
    metrics = evaluator.evaluate()

    if results_path is not None:
        row = {"dataset": dataset, "category": category, **metrics}
        update_results_table(results_path, row)
        print(f"Updated results table: {os.path.splitext(results_path)[0]}.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="train or eval")
    parser.add_argument("--dataframe_path", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--teacher_path", type=str, default=None,
                        help="Optional fine-tuned backbone; falls back to ImageNet if missing")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save/load the unified model")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone", type=str, default="mobilenet_v2")
    parser.add_argument("--expand_dim", type=int, default=0)
    parser.add_argument("--lambda_", type=float, default=0.55,
                        help="concept-loss weight (joint-CBM lambda)")
    parser.add_argument("--dataset", type=str, default="mvtec", help="mvtec or visa (for results table)")
    parser.add_argument("--inject_diffs", action="store_true",
                        help="unified++: also add deeper feature diffs into the concept net")
    parser.add_argument("--mask_student", action="store_true",
                        help="ablation: block anomalous samples from updating the student backbone")
    parser.add_argument("--results_path", type=str, default=None,
                        help="CSV path; an updating markdown table is written alongside")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if args.mode == "train":
        train_model(args.category, args.dataframe_path, args.teacher_path, args.save_path,
                    device, args.backbone, args.expand_dim, args.lambda_,
                    args.batch_size, args.optimizer, args.lr, args.epochs, args.seed,
                    inject_diffs=args.inject_diffs, mask_student=args.mask_student)
    elif args.mode == "eval":
        eval_model(args.category, args.dataframe_path, args.save_path, device,
                   args.backbone, args.expand_dim, args.batch_size,
                   args.dataset, args.results_path, inject_diffs=args.inject_diffs)


if __name__ == "__main__":
    main()
