"""Bounded training for the isolated candidate regret scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from gppo_world.minimal_consequence_model import CandidateRegretNet, graph_sample, instance_loss


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rows_for(data: dict[str, object], split: str) -> list[dict[str, object]]:
    return list(data["splits"][split]["records"])  # type: ignore[index]


def selection_metrics(model: CandidateRegretNet, rows: list[dict[str, object]], device: torch.device) -> dict[str, float]:
    regrets = []
    optimal = []
    errors = []
    pair_total = 0
    pair_correct = 0
    for row in rows:
        sample = graph_sample(row, device)
        with torch.no_grad():
            pred = model(sample)
        if not torch.isfinite(pred).all():
            raise FloatingPointError("non-finite validation prediction")
        chosen = int(torch.argmin(pred).item())
        labels = sample.labels.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        regrets.append(float(labels[chosen]))
        optimal.append(float(abs(labels[chosen]) <= 1e-8))
        errors.extend(np.abs(pred_np - labels).tolist())
        for i in range(len(labels)):
            for j in range(len(labels)):
                if labels[i] < labels[j]:
                    pair_total += 1
                    pair_correct += int(pred_np[i] < pred_np[j])
    return {"instances": float(len(rows)), "mean_normalized_selected_regret": float(np.mean(regrets)) if regrets else float("nan"), "optimal_first_action_rate": float(np.mean(optimal)) if optimal else float("nan"), "mean_absolute_label_error": float(np.mean(errors)) if errors else float("nan"), "non_tie_pair_accuracy": float(pair_correct / pair_total) if pair_total else float("nan"), "non_tie_pairs": float(pair_total)}


def save_checkpoint(path: Path, kind: str, model: CandidateRegretNet, optimizer: torch.optim.Optimizer, *, seed: int, epoch: int, update: int, config: dict[str, object], best_epoch: int, best_metric: float, patience_count: int, order_history: list[list[int]]) -> None:
    payload = {"format": "m10-minimal-scheduling-candidate-regret-training-v1", "checkpoint_kind": kind, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "seed": seed, "epoch": epoch, "optimizer_updates": update, "best_epoch": best_epoch, "best_validation_normalized_regret": best_metric, "patience_count": patience_count, "order_history": order_history, "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, "config": config}
    torch.save(payload, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--validation-split", default="validation")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--max-updates", type=int, default=800)
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--wall-seconds", type=float, default=20 * 60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--run-kind", choices=("smoke", "formal"), default="formal")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; refusing silent device fallback")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.data.read_text(encoding="utf-8"))
    train = rows_for(data, args.split)
    validation = rows_for(data, args.validation_split)
    if len(train) % args.batch_size:
        raise SystemExit("batch must contain complete instances for this registered run")
    seed_all(args.seed)
    device = torch.device(args.device)
    model = CandidateRegretNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    config = {"data": str(args.data.resolve()), "data_sha256": sha256(args.data), "train_split": args.split, "validation_split": args.validation_split, "seed": args.seed, "device": args.device, "hidden": 64, "message_passing_rounds": 2, "score_mlp_width": 64, "batch_size": args.batch_size, "max_epochs": args.max_epochs, "max_updates": args.max_updates, "patience": args.patience, "huber_delta": 1.0, "rank_weight": 0.1, "lr": 0.001, "weight_decay": 0.0, "run_kind": args.run_kind, "input_contract": "public positions/process/eligibility/precedence/candidate flight; no solver outputs, IDs, split, seed, path, or sample order"}
    dump(args.out / "run-identity.json", {"schema": "m10-minimal-scheduling-candidate-regret-training-run-v1", "config": config, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "runtime": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}})
    started = time.monotonic()
    order_rng = random.Random(args.seed + 1000003)
    order_history: list[list[int]] = []
    update = 0
    best_metric = float("inf")
    best_epoch = -1
    patience_count = 0
    stopped = "budget_exhausted"
    with (args.out / "updates.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(args.max_epochs):
            if time.monotonic() - started >= args.wall_seconds or update >= args.max_updates:
                stopped = "wall_or_update_budget"
                break
            order = list(range(len(train)))
            order_rng.shuffle(order)
            order_history.append(order)
            model.train()
            epoch_losses = []
            for begin in range(0, len(order), args.batch_size):
                if update >= args.max_updates or time.monotonic() - started >= args.wall_seconds:
                    stopped = "wall_or_update_budget"
                    break
                optimizer.zero_grad(set_to_none=True)
                batch_total = []
                parts = []
                for idx in order[begin:begin + args.batch_size]:
                    sample = graph_sample(train[idx], device)
                    loss, part = instance_loss(model, sample)
                    batch_total.append(loss)
                    parts.append(part)
                loss = torch.stack(batch_total).mean()
                if not torch.isfinite(loss):
                    stopped = "non_finite_loss"
                    raise FloatingPointError("non-finite loss")
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf")))
                if not torch.isfinite(torch.tensor(grad_norm)) or not all(torch.isfinite(p).all() for p in model.parameters()):
                    stopped = "non_finite_gradient_or_parameter"
                    raise FloatingPointError("non-finite gradient or parameter")
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    stopped = "non_finite_parameter_after_step"
                    raise FloatingPointError("non-finite parameter after optimizer step")
                update += 1
                epoch_losses.append(float(loss.detach().cpu()))
                log.write(json.dumps({"epoch": epoch, "optimizer_update": update, "loss": float(loss.detach().cpu()), "huber": float(np.mean([p["huber"] for p in parts])), "rank": float(np.mean([p["rank"] for p in parts])), "rank_pairs": float(np.sum([p["rank_pairs"] for p in parts])), "gradient_norm": grad_norm, "wall_seconds": time.monotonic() - started}) + "\n")
                log.flush()
            model.eval()
            val = selection_metrics(model, validation, device)
            val["epoch"] = float(epoch)
            val["optimizer_updates"] = float(update)
            val["wall_seconds"] = float(time.monotonic() - started)
            dump(args.out / f"validation-epoch-{epoch:03d}.json", val)
            metric = val["mean_normalized_selected_regret"]
            if metric < best_metric:
                best_metric = metric
                best_epoch = epoch
                patience_count = 0
                save_checkpoint(args.out / "best-inference.pt", "best-inference", model, optimizer, seed=args.seed, epoch=epoch, update=update, config=config, best_epoch=best_epoch, best_metric=best_metric, patience_count=patience_count, order_history=order_history)
            else:
                patience_count += 1
            save_checkpoint(args.out / "last-recovery.pt", "last-recovery", model, optimizer, seed=args.seed, epoch=epoch, update=update, config=config, best_epoch=best_epoch, best_metric=best_metric, patience_count=patience_count, order_history=order_history)
            if patience_count >= args.patience:
                stopped = "validation_patience"
                break
            if update >= args.max_updates or time.monotonic() - started >= args.wall_seconds:
                stopped = "wall_or_update_budget"
                break
        else:
            stopped = "epoch_budget"
    model.eval()
    dump(args.out / "training-summary.json", {"schema": "m10-minimal-scheduling-candidate-regret-training-summary-v1", "seed": args.seed, "run_kind": args.run_kind, "actual_epochs": len(order_history), "optimizer_updates": update, "best_epoch": best_epoch, "best_validation_normalized_regret": best_metric, "stopped_reason": stopped, "elapsed_seconds": time.monotonic() - started, "train_instances": len(train), "validation_instances": len(validation), "checkpoint_files": ["best-inference.pt", "last-recovery.pt"]})
    print(json.dumps({"status": "complete", "seed": args.seed, "updates": update, "epochs": len(order_history), "stopped": stopped, "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
