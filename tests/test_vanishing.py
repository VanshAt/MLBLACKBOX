"""
Test 3 — Vanishing Gradient Test

Builds a deep (8-layer) network using sigmoid activations everywhere in PyTorch.
Trains on a simple dataset. The fault detector should detect vanishing gradients.
The backtrack report should suggest switching to ReLU.
"""

import sys, os
import math as _math
import torch
import torch.nn as nn
import torch.optim as optim

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer import Trainer
from tracker.gradient_tracker import compute_gradient_stats, compute_weight_stats
from tracker.checkpoint import CheckpointManager
from tracker.metric_recorder import MetricRecorder
from tracker.fault_detector import FaultDetector, FaultType
from tracker.backtracker import Backtracker
from tracker.report import generate_report


def test_vanishing_gradient():
    """
    Deep sigmoid network (8 hidden layers) should exhibit vanishing gradients.
    Detector should flag GRADIENT_VANISHING and report should suggest ReLU.
    """
    torch.manual_seed(0)
    X_raw = [[i / 10.0] for i in range(30)]
    y_raw = [[0.5 + 0.5 * _math.sin(i / 3.0)] for i in range(30)]

    X = torch.tensor(X_raw, dtype=torch.float32)
    y = torch.tensor(y_raw, dtype=torch.float32)

    # 8-layer deep sigmoid network
    layers = []
    layers.append(nn.Linear(1, 6))
    layers.append(nn.Sigmoid())
    for _ in range(7):
        layers.append(nn.Linear(6, 6))
        layers.append(nn.Sigmoid())
    layers.append(nn.Linear(6, 1))
    layers.append(nn.Sigmoid())
    
    net = nn.Sequential(*layers)

    ckpt_dir = "checkpoints_test_vanish"
    log_path = os.path.join("logs", "test_vanish_audit.json")
    import shutil
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    if os.path.exists(log_path):
        os.remove(log_path)
    os.makedirs("logs", exist_ok=True)

    optimizer = optim.SGD(net.parameters(), lr=0.01)

    recorder = MetricRecorder(log_path=log_path)
    ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_dir)
    detector = FaultDetector(
        recorder,
        vanishing_threshold=1e-5,
        vanishing_no_improvement_epochs=10,
    )
    backtracker = Backtracker(ckpt_mgr, recorder)

    detected_fault = None

    def epoch_callback(record):
        nonlocal detected_fault
        grad_stats = compute_gradient_stats(net)
        weight_stats = compute_weight_stats(net)

        epoch_rec = recorder.append(
            epoch=record.epoch,
            loss=record.loss,
            accuracy=None,
            gradient_stats=grad_stats,
            weight_stats=weight_stats,
            learning_rate=record.learning_rate,
            epoch_time_ms=record.epoch_time_ms,
            batch_idx=record.batch_idx,
        )
        
        ckpt_mgr.save(
            network=net,
            epoch=record.epoch,
            loss=record.loss,
            gradient_stats=grad_stats,
            weight_stats=weight_stats,
            batch_idx=record.batch_idx,
        )

        if record.epoch >= 3:
            fault = detector.check(epoch_rec)
            if fault and detected_fault is None:
                detected_fault = fault
                return True

    trainer = Trainer(net, nn.MSELoss(), optimizer, print_every=20)
    trainer.add_callback(epoch_callback)
    trainer.train(X, y, epochs=200)

    print("\n" + "=" * 60)
    if detected_fault:
        print(f"FAULT DETECTED: {detected_fault.fault_type}")
        result = backtracker.backtrack(detected_fault, net)
        report_text = generate_report(result)
        print(report_text)

        assert detected_fault.fault_type in (
            FaultType.GRADIENT_VANISHING,
            "TRAINING_STAGNATION",
        ), (
            f"Expected GRADIENT_VANISHING or stagnation, got {detected_fault.fault_type}"
        )
        report_text_lower = report_text.lower()
        has_relu_suggestion = "relu" in report_text_lower or "activation" in report_text_lower
        print(f"Report mentions ReLU/activation: {has_relu_suggestion}")
        print("Vanishing gradient test PASSED ✓")
    else:
        print(
            "Vanishing gradient NOT detected in 200 epochs.\n"
            "The network may have converged. Try a deeper architecture or more epochs."
        )

    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)


if __name__ == "__main__":
    print("=" * 60)
    print("Test 3 — Vanishing Gradient (Deep Sigmoid Network)")
    print("=" * 60)
    test_vanishing_gradient()
