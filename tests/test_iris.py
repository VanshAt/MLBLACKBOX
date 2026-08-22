"""
Test 4 — Iris Full Run Test

Trains on the Iris dataset (binary classification: setosa vs rest).
Validates:
  1. Training accuracy exceeds 90%
  2. Full audit_log.json is produced after the run
  3. Checkpoints are saved for every epoch

Run:
    python tests/test_iris.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import Network
from core.activations import ReLU, Sigmoid
from core.loss import BinaryCrossEntropy
from training.trainer import Trainer
from tracker.gradient_tracker import compute_gradient_stats, compute_weight_stats
from tracker.checkpoint import CheckpointManager
from tracker.metric_recorder import MetricRecorder
from tracker.fault_detector import FaultDetector
from data.iris_loader import get_iris_binary


def test_iris_binary():
    """
    Binary classification on Iris (setosa vs rest).
    Expected: accuracy > 90% after 500 epochs.
    """
    X, y = get_iris_binary(normalize_features=True, shuffle=True, seed=42)

    ckpt_dir = "checkpoints_iris"
    log_path = os.path.join("logs", "iris_audit.json")

    if os.path.exists(log_path):
        os.remove(log_path)

    # 4 inputs → 8 hidden (ReLU) → 4 hidden (ReLU) → 1 output (Sigmoid)
    net = Network([4, 8, 4, 1], activations=[ReLU(), ReLU(), Sigmoid()], seed=42)
    recorder = MetricRecorder(log_path=log_path)
    ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_dir)
    detector = FaultDetector(recorder)

    fault_log = []

    captured_grad_stats = [{}]
    captured_weight_stats = [{}]
    original_clear = net.clear_gradients
    def capture_before_clear():
        captured_grad_stats[0] = compute_gradient_stats(net)
        captured_weight_stats[0] = compute_weight_stats(net)
        original_clear()
    net.clear_gradients = capture_before_clear

    def epoch_callback(record):
        grad_stats = captured_grad_stats[0] or compute_gradient_stats(net)
        weight_stats = captured_weight_stats[0] or compute_weight_stats(net)

        epoch_rec = recorder.append(
            epoch=record.epoch,
            loss=record.loss,
            accuracy=record.accuracy,
            gradient_stats=grad_stats,
            weight_stats=weight_stats,
            learning_rate=record.learning_rate,
            epoch_time_ms=record.epoch_time_ms,
            batch_idx=record.batch_idx,
            nan_layer=record.nan_layer,
        )

        ckpt_mgr.save(
            network=net,
            epoch=record.epoch,
            loss=record.loss,
            accuracy=record.accuracy,
            gradient_stats=grad_stats,
            weight_stats=weight_stats,
            batch_idx=record.batch_idx,
        )

        fault = detector.check(epoch_rec)
        if fault:
            fault_log.append(fault.fault_type)
            print(f"  [Fault at epoch {record.epoch}] {fault.fault_type}: {fault.description}")

    trainer = Trainer(
        net,
        BinaryCrossEntropy(),
        learning_rate=0.05,
        classification=True,
        print_every=100,
    )
    trainer.add_callback(epoch_callback)
    history = trainer.train(X, y, epochs=500)

    final = history[-1]
    print(f"\nIris Final — loss={final['loss']:.4f}  accuracy={final['accuracy']:.4f}")

    # Check audit log was created
    assert os.path.exists(log_path), "Audit log was not created!"
    all_records = recorder.read()
    print(f"Audit log: {len(all_records)} epoch records written to '{log_path}'")

    # Check checkpoints
    saved = ckpt_mgr.list_epochs()
    print(f"Checkpoints saved: {len(saved)} (epochs {saved[:3]}...{saved[-3:]})")

    # Check accuracy
    assert final["accuracy"] >= 0.90, (
        f"Iris test FAILED: accuracy {final['accuracy']:.4f} < 0.90. "
        "Try more epochs or a larger network."
    )

    if fault_log:
        print(f"Faults detected during run: {fault_log}")
    else:
        print("No faults detected during training ✓")

    print("Iris full run test PASSED ✓")

    # Cleanup
    import shutil
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)


if __name__ == "__main__":
    print("=" * 60)
    print("Test 4 — Iris Dataset Full Run")
    print("=" * 60)
    test_iris_binary()
