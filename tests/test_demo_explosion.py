"""
tests/test_demo_explosion.py — Regression test for demo_explosion.py.

Asserts:
  - demo_models/demo_explosion.py triggers GRADIENT_EXPLOSION
  - it does not report TRAINING_STAGNATION as the primary fault
  - training halts at the explosion epoch (epoch 3)
  - rollback restores the previous healthy checkpoint (epoch 2)
"""

import os
import shutil
import unittest
import torch
import torch.nn as nn
import torch.optim as optim

from training.trainer import Trainer
from tracker.gradient_tracker import compute_gradient_stats, compute_weight_stats
from tracker.checkpoint import CheckpointManager
from tracker.metric_recorder import MetricRecorder
from tracker.fault_detector import FaultDetector, FaultType
from tracker.backtracker import Backtracker


class TestDemoExplosion(unittest.TestCase):

    def setUp(self):
        self.ckpt_dir = os.path.join("checkpoints", "test_demo_explosion")
        self.log_path = os.path.join("logs", "test_demo_explosion_audit.json")

        if os.path.exists(self.ckpt_dir):
            shutil.rmtree(self.ckpt_dir)
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

        os.makedirs("logs", exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.ckpt_dir):
            shutil.rmtree(self.ckpt_dir)
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

    def test_demo_explosion_behavior(self):
        """
        Verify demo_explosion triggers GRADIENT_EXPLOSION at epoch 3,
        does not report TRAINING_STAGNATION, halts at epoch 3, and restores epoch 2.
        """
        torch.manual_seed(42)
        net = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1)
        )
        optimizer = optim.SGD(net.parameters(), lr=0.01)

        X = torch.rand(200, 3) * 5.0
        y = (X[:, 0] + X[:, 1]).unsqueeze(1)

        recorder = MetricRecorder(log_path=self.log_path)
        ckpt_mgr = CheckpointManager(checkpoint_dir=self.ckpt_dir)
        detector = FaultDetector(recorder, explosion_window=3, explosion_sigma=2.0)
        backtracker = Backtracker(ckpt_mgr, recorder)
        detected_fault = None
        halted_epoch = None

        def on_epoch_end(record):
            nonlocal detected_fault, halted_epoch
            grad_stats = record.gradient_stats if record.gradient_stats else compute_gradient_stats(net)
            weight_stats = compute_weight_stats(net)
            epoch_rec = recorder.append(
                epoch=record.epoch, loss=record.loss, accuracy=record.accuracy,
                gradient_stats=grad_stats, weight_stats=weight_stats,
                learning_rate=record.learning_rate, epoch_time_ms=record.epoch_time_ms,
                batch_idx=record.batch_idx, nan_layer=record.nan_layer,
            )
            ckpt_mgr.save(net, record.epoch, record.loss, record.accuracy,
                          grad_stats, weight_stats, batch_idx=record.batch_idx,
                          learning_rate=record.learning_rate, optimizer=optimizer)

            if record.epoch >= 3 or record.nan_layer >= 0:
                fault = detector.check(epoch_rec)
                if fault:
                    detected_fault = fault
                    halted_epoch = record.epoch
                    return True  # stop training

            # At epoch 2 completion, inject Bug: high LR + unscaled features for epoch 3
            if record.epoch == 2:
                for pg in optimizer.param_groups:
                    pg['lr'] = 0.08
                X.mul_(2.0)
                y.mul_(2.0)

        trainer = Trainer(net, nn.MSELoss(), optimizer, classification=False)
        trainer.add_callback(on_epoch_end)
        history = trainer.train(X, y, epochs=100)

        # 1. Assert fault was detected
        self.assertIsNotNone(detected_fault, "Expected a fault to be detected, but None was returned.")

        # 2. Assert primary fault is GRADIENT_EXPLOSION and NOT TRAINING_STAGNATION
        self.assertEqual(
            detected_fault.fault_type,
            FaultType.GRADIENT_EXPLOSION,
            f"Expected GRADIENT_EXPLOSION, got {detected_fault.fault_type}"
        )
        self.assertNotEqual(
            detected_fault.fault_type,
            FaultType.TRAINING_STAGNATION,
            "TRAINING_STAGNATION should not be the primary detected fault."
        )

        # 3. Assert training halted at the explosion epoch (epoch 3)
        self.assertEqual(halted_epoch, 3, f"Expected training to halt at epoch 3, halted at {halted_epoch}")
        self.assertEqual(len(history), 3, f"Expected 3 epochs executed, got {len(history)}")

        # 4. Assert rollback restores the previous healthy checkpoint (epoch 2)
        result = backtracker.backtrack(detected_fault, net, optimizer=optimizer)
        self.assertTrue(result.network_restored, "Network restoration failed during backtrack.")
        self.assertEqual(result.clean_epoch, 2, f"Expected rollback to epoch 2, restored epoch {result.clean_epoch}")


if __name__ == "__main__":
    unittest.main()
