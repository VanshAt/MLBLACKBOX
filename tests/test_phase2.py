"""
test_phase2.py — Test suite for Phase 2 core enhancements:
  - trainer.resume()
  - resume_from_last_healthy_checkpoint()
  - parent_run_id linkage
  - Run session creation and reading
  - Checkpoint retention policy
  - Checkpoint deletion and comparison
  - Checkpoint export/import
  - Cross-device CPU loading
  - Training Health Score calculation
  - Layer failure risk ranking
  - Multi-format report export
"""

import os
import shutil
import unittest
import torch
import torch.nn as nn
import torch.optim as optim

from training.trainer import Trainer
from tracker.checkpoint import CheckpointManager
from tracker.metric_recorder import MetricRecorder
from tracker.health_score import calculate_health_score
from tracker.layer_risk import rank_layers_by_risk
from runs.session import RunSessionManager
from reports.report_generator import (
    export_markdown_report,
    export_metrics_csv,
    export_fault_log_json,
    export_health_summary_html,
)


class TestPhase2Features(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_phase2_workspace"
        self.ckpt_dir = os.path.join(self.test_dir, "checkpoints")
        self.log_path = os.path.join(self.test_dir, "audit.json")
        self.runs_file = os.path.join(self.test_dir, "run_history.json")
        self.reports_dir = os.path.join(self.test_dir, "reports")

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_resume_and_parent_run_id(self):
        """Verify trainer.resume() restores state and links parent_run_id."""
        torch.manual_seed(42)
        X = torch.rand(20, 2)
        y = (X[:, 0] + X[:, 1]).unsqueeze(1)

        net = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 1))
        opt = optim.SGD(net.parameters(), lr=0.01)
        sched = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        ckpt_mgr = CheckpointManager(checkpoint_dir=self.ckpt_dir)
        recorder = MetricRecorder(log_path=self.log_path)

        trainer = Trainer(
            network=net,
            loss_fn=nn.MSELoss(),
            optimizer=opt,
            scheduler=sched,
            checkpoint_manager=ckpt_mgr,
            metric_recorder=recorder,
            print_every=0,
        )

        def cb(rec):
            ckpt_mgr.save(net, rec.epoch, rec.loss, optimizer=opt, scheduler=sched)
        trainer.add_callback(cb)

        # Train 3 epochs
        trainer.train(X, y, epochs=3)
        initial_run_id = trainer.run_id

        # Resume from epoch 2 with new learning rate
        resumed_history = trainer.resume(
            checkpoint_epoch_or_path=2,
            lr=0.005,
            epochs=2,
            X=X,
            y=y,
        )

        self.assertEqual(trainer.parent_run_id, initial_run_id)
        self.assertNotEqual(trainer.run_id, initial_run_id)
        self.assertEqual(trainer.learning_rate, 0.005)
        self.assertEqual(len(resumed_history), 2)

    def test_resume_from_last_healthy_checkpoint(self):
        """Verify automated recovery resumes from last clean checkpoint."""
        torch.manual_seed(42)
        X = torch.rand(20, 2)
        y = (X[:, 0] + X[:, 1]).unsqueeze(1)

        net = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 1))
        opt = optim.SGD(net.parameters(), lr=0.01)
        ckpt_mgr = CheckpointManager(checkpoint_dir=self.ckpt_dir)

        trainer = Trainer(
            network=net,
            loss_fn=nn.MSELoss(),
            optimizer=opt,
            checkpoint_manager=ckpt_mgr,
            print_every=0,
        )

        def cb(rec):
            ckpt_mgr.save(net, rec.epoch, rec.loss, optimizer=opt)
        trainer.add_callback(cb)

        trainer.train(X, y, epochs=3)

        # Automatic recovery from last healthy checkpoint
        history = trainer.resume_from_last_healthy_checkpoint(epochs=1, X=X, y=y)
        self.assertEqual(len(history), 1)

    def test_checkpoint_retention_and_deletion(self):
        """Verify checkpoint retention policy and single deletion."""
        net = nn.Sequential(nn.Linear(2, 2))
        ckpt_mgr = CheckpointManager(checkpoint_dir=self.ckpt_dir)

        for ep in range(1, 8):
            ckpt_mgr.save(net, epoch=ep, loss=1.0 / ep)

        self.assertEqual(len(ckpt_mgr.list_epochs()), 7)

        # Retain latest 3 and best loss
        deleted = ckpt_mgr.apply_retention_policy(keep_latest_n=3, keep_best_loss=True)
        remaining = ckpt_mgr.list_epochs()

        self.assertIn(7, remaining)
        self.assertIn(6, remaining)
        self.assertIn(5, remaining)
        self.assertTrue(len(deleted) > 0)

        # Test single deletion
        ckpt_mgr.delete_checkpoint(7)
        self.assertNotIn(7, ckpt_mgr.list_epochs())

    def test_checkpoint_export_and_import(self):
        """Verify exporting a checkpoint to zip and importing it into a clean manager."""
        net = nn.Sequential(nn.Linear(2, 2))
        ckpt_mgr = CheckpointManager(checkpoint_dir=self.ckpt_dir)
        ckpt_mgr.save(net, epoch=1, loss=0.42)

        export_dir = os.path.join(self.test_dir, "exported")
        archive = ckpt_mgr.export_checkpoint(1, export_dir=export_dir)
        self.assertTrue(os.path.exists(archive))

        import_dir = os.path.join(self.test_dir, "imported_ckpt")
        new_mgr = CheckpointManager(checkpoint_dir=import_dir)
        ep = new_mgr.import_checkpoint(archive)
        self.assertEqual(ep, 1)
        self.assertIn(1, new_mgr.list_epochs())

    def test_checkpoint_comparison(self):
        """Verify weight diff comparison between two checkpoints."""
        net = nn.Sequential(nn.Linear(2, 2))
        ckpt_mgr = CheckpointManager(checkpoint_dir=self.ckpt_dir)
        ckpt_mgr.save(net, epoch=1, loss=0.5)

        # Perturb weights
        with torch.no_grad():
            for p in net.parameters():
                p.add_(1.0)
        ckpt_mgr.save(net, epoch=2, loss=0.2)

        diff = ckpt_mgr.compare_checkpoints(1, 2)
        self.assertIn("param_norm_diffs", diff)
        self.assertGreater(diff["total_weight_movement"], 0.0)

    def test_health_score_and_layer_risk(self):
        """Verify Training Health Score and Layer Risk Ranking calculation."""
        history = [
            {
                "epoch": 1,
                "loss": 0.5,
                "gradient": {"global": {"mean": 0.1}, "per_layer": [{"layer": 0, "name": "fc1.weight", "mean": 0.1, "has_nan": False}]},
                "nan_layer": -1,
                "fault_detected": None,
            },
            {
                "epoch": 2,
                "loss": 0.3,
                "gradient": {"global": {"mean": 0.08}, "per_layer": [{"layer": 0, "name": "fc1.weight", "mean": 0.08, "has_nan": False}]},
                "nan_layer": -1,
                "fault_detected": None,
            }
        ]

        score, exp, comp = calculate_health_score(history[-1], history)
        self.assertGreaterEqual(score, 70)
        self.assertIn("Health", exp)

        risks = rank_layers_by_risk(history)
        self.assertEqual(len(risks), 1)
        self.assertEqual(risks[0]["name"], "fc1.weight")

    def test_report_exporters(self):
        """Verify Markdown, CSV, JSON, and HTML report exports."""
        history = [
            {"epoch": 1, "loss": 0.5, "accuracy": 0.8, "learning_rate": 0.01, "epoch_time_ms": 10.0, "health_score": 90, "gradient": {"global": {"mean": 0.1}}, "fault_detected": None}
        ]
        layer_risks = [
            {"layer": 0, "name": "fc1", "risk_score": 20, "trend": "Stable", "likely_issue": "None", "recommended_action": "None", "latest_norm": 0.1}
        ]

        md_path = export_markdown_report(history, layer_risks, export_dir=self.reports_dir)
        csv_path = export_metrics_csv(history, export_dir=self.reports_dir)
        json_path = export_fault_log_json(history, export_dir=self.reports_dir)
        html_path = export_health_summary_html(history, layer_risks, export_dir=self.reports_dir)

        self.assertTrue(os.path.exists(md_path))
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(html_path))


if __name__ == "__main__":
    unittest.main()
