"""Diagnose frozen auxiliary predictions on the already saved train rollouts.

This is a read-only replay of recorded training actions.  It does not sample
new actions, update parameters, use final-test data, or alter any checkpoint.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.eawm_auxiliary import EventAwareHistoryPolicy  # noqa: E402
from gppo_world.m10_environment import M10Config, M10Environment, scenario_from_dict  # noqa: E402


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def public_digest(value: dict) -> str:
    import hashlib
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def confusion_metrics(matrix: np.ndarray) -> dict:
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    total = int(matrix.sum())
    recalls = []
    for index, count in enumerate(support):
        if count > 0:
            recalls.append(float(matrix[index, index] / count))
    precisions = [float(matrix[i, i] / predicted[i]) if predicted[i] else 0.0 for i in range(matrix.shape[0])]
    f1 = []
    for index in range(matrix.shape[0]):
        if support[index] > 0:
            p, r = precisions[index], float(matrix[index, index] / support[index])
            f1.append(2 * p * r / (p + r) if p + r else 0.0)
    majority = int(np.argmax(support)) if total else None
    majority_correct = int(support[majority]) if majority is not None else 0
    return {
        "support": support.astype(int).tolist(), "predicted": predicted.astype(int).tolist(),
        "confusion": matrix.astype(int).tolist(), "valid_count": total,
        "accuracy": float(np.trace(matrix) / total) if total else None,
        "macro_f1_supported_classes": float(np.mean(f1)) if f1 else None,
        "supported_class_recall": recalls,
        "majority_class": majority, "majority_accuracy": float(majority_correct / total) if total else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    matrix = json.loads((args.matrix_root / "frozen-tapes.json").read_text(encoding="utf-8"))
    train_tape = [scenario_from_dict(item) for item in matrix["train"]]
    scenario_by_id = {item.tape_id: item for item in train_tape}
    config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    runs = sorted((args.matrix_root / "runs").glob("m10-eawm-aux-e-*-formal"))
    result = {"schema": "gppo-eawm-auxiliary-prediction-diagnostics-v1", "read_only_recorded_train_replay": True, "parameter_updates": 0, "runs": []}
    for run in runs:
        payload = torch.load(run / "last.pt", map_location=args.device, weights_only=False)
        policy = EventAwareHistoryPolicy(__import__("gppo_world.m10_training", fromlist=["M10ActorCritic"]).M10ActorCritic(
            uav_count=4, task_capacity=6, action_count=25, encoder="graph", type_count=5, history=True,
        )).to(args.device)
        policy.load_state_dict(payload["state_dict"])
        policy.eval()
        matrices = {"position_x": np.zeros((3, 3), dtype=np.int64), "position_y": np.zeros((3, 3), dtype=np.int64), "energy": np.zeros((3, 3), dtype=np.int64), "task_state": np.zeros((2, 2), dtype=np.int64)}
        mismatches = {"current_public": 0, "next_public": 0}
        records = 0
        effective_steps = 0
        hidden = None
        env = None
        with gzip.open(run / "training-rollout-ledger.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["episode_start"] or env is None:
                    scenario = scenario_by_id[row["tape_id"]]
                    env = M10Environment(config, scenario)
                    obs = env.reset()
                    hidden = None
                expected = {"uavs": np.asarray(row["public_observation"]["uavs"], dtype=np.float32).tolist(), "tasks": np.asarray(row["public_observation"]["tasks"], dtype=np.float32).tolist(), "entity_ids": row["public_observation"]["entity_ids"], "mask": row["public_observation"]["mask"], "time": float(row["public_observation"]["time"])}
                actual = {"uavs": np.asarray(obs["uavs"], dtype=np.float32).tolist(), "tasks": np.asarray(obs["tasks"], dtype=np.float32).tolist(), "entity_ids": obs["entity_ids"], "mask": np.asarray(obs["mask"], dtype=bool).tolist(), "time": float(obs["time"])}
                if public_digest(expected) != public_digest(actual):
                    mismatches["current_public"] += 1
                with torch.inference_mode():
                    public = torch.as_tensor(obs["flat"], dtype=torch.float32, device=args.device)[None, :]
                    action = torch.tensor([int(row["action"])], dtype=torch.long, device=args.device)
                    _, _, predictions, hidden = policy.forward_with_aux(public, action, hidden)
                targets = row["targets"]
                for key, matrix_key, classes in (("position_class", "position_x", 3), ("position_class", "position_y", 3), ("energy_class", "energy", 3), ("task_state_changed", "task_state", 2)):
                    mask_key = "position_mask" if key == "position_class" else "energy_mask" if key == "energy_class" else "task_state_mask"
                    target = np.asarray(targets[key], dtype=np.int64)
                    mask = np.asarray(targets[mask_key], dtype=bool)
                    if key == "position_class":
                        axis = 0 if matrix_key == "position_x" else 1
                        prediction = predictions["position_logits"][0, :, axis].argmax(dim=-1).cpu().numpy()
                        target = target[:, axis]
                        mask = mask[:, axis]
                    elif key == "energy_class":
                        prediction = predictions["energy_logits"][0].argmax(dim=-1).cpu().numpy()
                    else:
                        prediction = (torch.sigmoid(predictions["task_state_logits"][0]) >= 0.5).to(torch.int64).cpu().numpy()
                    if target.shape != prediction.shape:
                        raise RuntimeError(f"label/prediction shape mismatch in {run.name}: {key} {target.shape} vs {prediction.shape}")
                    for truth, guess in zip(target[mask].tolist(), prediction[mask].tolist()):
                        matrices[matrix_key][truth, guess] += 1
                    effective_steps += int(mask.sum()) if key == "position_class" else 0
                next_obs, _, done, _ = env.step(int(row["action"]), submit_command=True)
                expected_next = {"uavs": np.asarray(row["next_public_observation"]["uavs"], dtype=np.float32).tolist(), "tasks": np.asarray(row["next_public_observation"]["tasks"], dtype=np.float32).tolist(), "entity_ids": row["next_public_observation"]["entity_ids"], "mask": row["next_public_observation"]["mask"], "time": float(row["next_public_observation"]["time"])}
                actual_next = {"uavs": np.asarray(next_obs["uavs"], dtype=np.float32).tolist(), "tasks": np.asarray(next_obs["tasks"], dtype=np.float32).tolist(), "entity_ids": next_obs["entity_ids"], "mask": np.asarray(next_obs["mask"], dtype=bool).tolist(), "time": float(next_obs["time"])}
                if public_digest(expected_next) != public_digest(actual_next):
                    mismatches["next_public"] += 1
                if done:
                    env = None
                    hidden = None
                else:
                    obs = next_obs
                records += 1
        result["runs"].append({"run_id": run.name, "records": records, "replay_public_mismatches": mismatches, "metrics": {key: confusion_metrics(value) for key, value in matrices.items()}, "model_updates": 0, "checkpoint": str(run / "last.pt")})
    dump(args.out, result)
    print(json.dumps({"runs": len(result["runs"]), "out": str(args.out), "parameter_updates": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
