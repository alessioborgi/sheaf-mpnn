"""Regression coverage for the upstream Molecular evaluation convention."""

import argparse

import pytest
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from exp.hsnn_run import evaluation_metric, metrics


def test_molecular_f1_matches_reference_formula():
    target = torch.tensor([0, 0, 0, 1, 1, 2])
    prediction = torch.tensor([0, 0, 1, 1, 2, 2])
    logits = torch.nn.functional.one_hot(prediction, 4).float()
    result = metrics(logits, target)
    precision = precision_score(target, prediction, average="macro", zero_division=0)
    recall = recall_score(target, prediction, average="macro", zero_division=0)
    assert result["f1"] == pytest.approx(2 * precision * recall / (precision + recall))
    assert result["f1"] != pytest.approx(f1_score(target, prediction, average="macro"))
    assert evaluation_metric(argparse.Namespace(task="hyperedge")) == "f1"
    assert evaluation_metric(argparse.Namespace(task="node")) == "accuracy"


def test_zero_precision_and_recall():
    result = metrics(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), torch.tensor([0, 1]))
    assert result["f1"] == 0
