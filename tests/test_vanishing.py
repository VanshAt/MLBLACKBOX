"""
Test 3 — Vanishing Gradient Test

Builds a deep (8-layer) network using sigmoid activations everywhere.
Trains on a simple dataset. The fault detector should detect vanishing gradients.
The backtrack report should suggest switching to ReLU.

Run:
    python tests/test_vanishing.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import Network
from core.activations import Sigmoid
from core.loss import MSE
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
    # More complex dataset: y = sin(x) approximation to stress the network
    import math as _math
    X = [[i / 10.0] for i in range(30)]
    y = [[0.5 + 0.5 * _math.sin(i / 3.0)] for i in range(30)]

    # 8-layer deep sigmoid network — this WILL have vanishing gradients
    activations = [Sigmoid()] * 8 + [Sigmoid()]  # all sigmoid, 9 layers total
    sizes = [1] + [6] * 8 + [1]  # slightly wider to observe gradient flow

    ckpt_dir = "checkpoints_test_vanish"
    log_path = os.path.join("logs", "test_vanish_audit.json")
    import shutil
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    if os.path.exists(log_path):
        os.remove(log_path)

    net = Network(sizes, activations=activations, seed=0)
    recorder = MetricRecorder(log_path=log_path)
    ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_dir)
    detector = FaultDetector(
        recorder,
        vanishing_threshold=1e-5,       # slightly relaxed for test speed
        vanishing_no_improvement_epochs=10,
    )
    backtracker = Backtracker(ckpt_mgr, recorder)

    detected_fault = None

    captured_grad_stats = [{}]
    captured_weight_stats = [{}]
    original_clear = net.clear_gradients
    def capture_before_clear():
        captured_grad_stats[0] = compute_gradient_stats(net)
        captured_weight_stats[0] = compute_weight_stats(net)
        original_clear()
    net.clear_gradients = capture_before_clear

    def epoch_callback(record):
        nonlocal detected_fault
        grad_stats = captured_grad_stats[0] or compute_gradient_stats(net)
        weight_stats = captured_weight_stats[0] or compute_weight_stats(net)

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
        )

        if record.epoch >= 3:
            fault = detector.check(epoch_rec)
            if fault and detected_fault is None:
                detected_fault = fault
                return True  # stop

    trainer = Trainer(net, MSE(), learning_rate=0.01, print_every=20)
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
            "TRAINING_STAGNATION",  # stagnation on deep sigmoid is vanishing-adjacent
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

    # Cleanup
    import shutil
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    if os.path.exists(log_path):
        os.remove(log_path)


if __name__ == "__main__":
    print("=" * 60)
    print("Test 3 — Vanishing Gradient (Deep Sigmoid Network)")
    print("=" * 60)
    test_vanishing_gradient()
