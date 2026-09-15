"""Candidate-action regret model for the isolated minimum scheduling benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


HIDDEN = 64


@dataclass
class GraphSample:
    uav_features: torch.Tensor
    task_features: torch.Tensor
    capability_edges: torch.Tensor  # [E, 2], uav index then task index
    precedence_edges: torch.Tensor  # [P, 2], predecessor then successor
    candidate_actions: list[tuple[int, int]]  # task index, uav index
    labels: torch.Tensor


class CandidateRegretNet(nn.Module):
    """Two-type, two-round message-passing graph scorer.

    It has no access to labels, solver outputs, IDs, split names, or sample
    order. A score is produced independently for every legal candidate.
    Lower score means a better predicted normalized regret.
    """

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.uav_input = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.task_input = nn.Sequential(nn.Linear(7, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.cap_edge = nn.Linear(2, hidden)
        self.pre_edge = nn.Linear(1, hidden)
        self.uav_update = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.task_update = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.score = nn.Sequential(nn.Linear(hidden * 2 + 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, sample: GraphSample) -> torch.Tensor:
        uav = self.uav_input(sample.uav_features)
        task = self.task_input(sample.task_features)
        for _ in range(2):
            cap_uav = torch.zeros_like(uav)
            cap_task = torch.zeros_like(task)
            if sample.capability_edges.numel():
                ui = sample.capability_edges[:, 0]
                ti = sample.capability_edges[:, 1]
                edge = self.cap_edge(sample.task_features[ti, 0:2] - sample.uav_features[ui, 0:2])
                cap_uav.index_add_(0, ui, task[ti] + edge)
                cap_task.index_add_(0, ti, uav[ui] + edge)
            pre = torch.zeros_like(task)
            if sample.precedence_edges.numel():
                src = sample.precedence_edges[:, 0]
                dst = sample.precedence_edges[:, 1]
                pre.index_add_(0, dst, task[src] + self.pre_edge(torch.ones((len(src), 1), device=task.device)))
            uav = uav + self.uav_update(torch.cat([uav, cap_uav], dim=-1))
            task = task + self.task_update(torch.cat([task, cap_task, pre], dim=-1))
        scores = []
        for task_index, uav_index in sample.candidate_actions:
            edge_features = torch.stack([
                torch.abs(sample.task_features[task_index, 0] - sample.uav_features[uav_index, 0]) + torch.abs(sample.task_features[task_index, 1] - sample.uav_features[uav_index, 1]),
                sample.task_features[task_index, 2],
            ])
            scores.append(self.score(torch.cat([uav[uav_index], task[task_index], edge_features], dim=0)))
        return torch.cat(scores, dim=0) if scores else torch.empty(0, device=task.device)


def graph_sample(instance: dict[str, Any], device: torch.device) -> GraphSample:
    record = instance
    scene = instance.get("instance", instance)
    uavs = scene["uavs"]
    tasks = scene["subtasks"]
    uav_features = torch.tensor([[u["position"][0] / 10.0, u["position"][1] / 10.0, 1.0] for u in uavs], dtype=torch.float32, device=device)
    task_index = {t["task_id"]: i for i, t in enumerate(tasks)}
    task_features = []
    for t in tasks:
        task_features.append([
            t["position"][0] / 10.0, t["position"][1] / 10.0,
            t["process_time"] / 5.0, len(t["eligible_uavs"]) / 2.0,
            len(t["predecessors"]) / 1.0, t["index"] / 1.0,
            1.0,
        ])
    task_features_tensor = torch.tensor(task_features, dtype=torch.float32, device=device)
    capability = torch.tensor([[u, task_index[t["task_id"]]] for t in tasks for u in t["eligible_uavs"]], dtype=torch.long, device=device)
    precedence = torch.tensor([[task_index[p], task_index[t["task_id"]]] for t in tasks for p in t["predecessors"]], dtype=torch.long, device=device)
    if not len(capability):
        capability = torch.empty((0, 2), dtype=torch.long, device=device)
    if not len(precedence):
        precedence = torch.empty((0, 2), dtype=torch.long, device=device)
    candidates = [(int(a["task_index"]), int(a["uav_id"])) for a in record["candidates"]]
    labels = torch.tensor([float(a["normalized_regret"]) for a in record["candidates"]], dtype=torch.float32, device=device)
    return GraphSample(uav_features, task_features_tensor, capability, precedence, candidates, labels)


def instance_loss(model: CandidateRegretNet, sample: GraphSample) -> tuple[torch.Tensor, dict[str, float]]:
    pred = model(sample)
    if not torch.isfinite(pred).all():
        raise FloatingPointError("non-finite candidate predictions")
    huber = F.huber_loss(pred, sample.labels, delta=1.0, reduction="mean") if len(pred) else pred.sum() * 0.0
    pairs = []
    for i in range(len(sample.labels)):
        for j in range(len(sample.labels)):
            if sample.labels[i] < sample.labels[j]:
                pairs.append(F.softplus(pred[i] - pred[j]))
    rank = torch.stack(pairs).mean() if pairs else pred.sum() * 0.0
    total = huber + 0.1 * rank
    return total, {"huber": float(huber.detach().cpu()), "rank": float(rank.detach().cpu()), "total": float(total.detach().cpu()), "rank_pairs": float(len(pairs))}
