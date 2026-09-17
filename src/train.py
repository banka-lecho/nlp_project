import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data import load_dataset, split_dataset
from evaluate import format_report, macro_f1, subgroup_report
from model import TwoHeadClassifier, pick_device

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ai-forever/ruRoberta-large")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(df, tokenizer, max_length, batch_size, shuffle):
    def collate(rows):
        encoded = tokenizer(
            [row["content"] for row in rows],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        labels = {
            "stance": torch.tensor([row["stance_label"] for row in rows]),
            "premise": torch.tensor([row["premise_label"] for row in rows]),
        }
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}, labels

    records = df[["content", "stance_label", "premise_label"]].to_dict("records")
    return DataLoader(records, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def autocast(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    stance, premise = [], []
    for inputs, _ in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with autocast(device):
            stance_logits, premise_logits = model(**inputs)
        stance.append(stance_logits.argmax(-1).cpu())
        premise.append(premise_logits.argmax(-1).cpu())
    return torch.cat(stance).numpy(), torch.cat(premise).numpy()


def scores(df, stance_pred, premise_pred):
    return {
        "stance": macro_f1(df["stance_label"], stance_pred),
        "premise": macro_f1(df["premise_label"], premise_pred),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device()
    output_dir = args.output_dir or ROOT / "runs" / f"{args.model.split('/')[-1]}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = split_dataset(load_dataset(), args.split_seed)
    if args.limit:
        train_df, val_df, test_df = (d.head(args.limit) for d in (train_df, val_df, test_df))
    print(f"device={device} train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_loader = make_loader(train_df, tokenizer, args.max_length, args.batch_size, shuffle=True)
    val_loader = make_loader(val_df, tokenizer, args.max_length, args.batch_size, shuffle=False)
    test_loader = make_loader(test_df, tokenizer, args.max_length, args.batch_size, shuffle=False)

    model = TwoHeadClassifier(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)

    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (inputs, labels) in enumerate(train_loader, 1):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = {k: v.to(device) for k, v in labels.items()}
            with autocast(device):
                loss = model.loss(model(**inputs), labels)
            (loss / args.grad_accum).backward()
            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if device.type == "mps":
                    torch.mps.empty_cache()
            running_loss += loss.item()
            if step % 20 == 0 or step == len(train_loader):
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss={running_loss / step:.4f}", flush=True)
        val_scores = scores(val_df, *predict(model, val_loader, device))
        history.append({"epoch": epoch, "train_loss": running_loss / len(train_loader), "val": val_scores})
        print(f"epoch {epoch} val stance={val_scores['stance']:.4f} premise={val_scores['premise']:.4f}", flush=True)

    stance_pred, premise_pred = predict(model, test_loader, device)
    predictions = test_df.drop(columns=["content"]).assign(stance_pred=stance_pred, premise_pred=premise_pred)
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    report = subgroup_report(predictions)
    report.to_csv(output_dir / "report.csv", index=False)
    print(format_report(report))

    summary = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "device": str(device),
        "train_minutes": (time.time() - start) / 60,
        "history": history,
        "test": scores(test_df, stance_pred, premise_pred),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["test"], indent=2))


if __name__ == "__main__":
    main()
