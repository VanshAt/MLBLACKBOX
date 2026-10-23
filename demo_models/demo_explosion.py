"""
DEMO 1 — Gradient Explosion
============================
The Bug:  Learning rate is set to 10.0 (way too high, normal is 0.001-0.1)
          AND inputs are not normalized (raw values up to 1000).

What happens: Gradients grow exponentially each epoch.
MLBlackBox detects: GRADIENT_EXPLOSION (Critical)
Dashboard shows:    Loss curve shoots upward to infinity.
                    Gradient Health chart spikes massively.

Run: python demo_models/demo_explosion.py
Then upload: logs/audit_log.json to the dashboard
"""

import sys, os, argparse, shutil
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from training.trainer import Trainer
from tracker.gradient_tracker import compute_gradient_stats, compute_weight_stats
from tracker.checkpoint import CheckpointManager
from tracker.metric_recorder import MetricRecorder
from tracker.fault_detector import FaultDetector
from tracker.backtracker import Backtracker
from tracker.report import print_report

parser = argparse.ArgumentParser(description="Demo 1: Gradient Explosion")
parser.add_argument("--fresh", action="store_true", help="Clear previous demo checkpoints and logs")
args = parser.parse_args()

ckpt_dir = os.path.join("checkpoints", "demo_explosion")
log_path = os.path.join("logs", "demo_explosion_audit.json")

if args.fresh:
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    if os.path.exists(log_path):
        os.remove(log_path)

os.makedirs("logs", exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

print("=" * 60)
print("DEMO 1 - Gradient Explosion (Learning Rate Too High)")
print("=" * 60)

# ── The Buggy Model ───────────────────────────────────────────
# Simple regression network — fine architecture, broken training config
net = nn.Sequential(
    nn.Linear(3, 16), nn.ReLU(),
    nn.Linear(16, 8), nn.ReLU(),
    nn.Linear(8, 1)
)

# Start with moderate learning rate, then spike at epoch 3
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Synthetic data
torch.manual_seed(42)
X = torch.rand(200, 3) * 5.0
y = (X[:, 0] + X[:, 1]).unsqueeze(1)

# ── MLBlackBox Setup ──────────────────────────────────────────
recorder    = MetricRecorder(log_path=log_path)
ckpt_mgr    = CheckpointManager(checkpoint_dir=ckpt_dir)
detector    = FaultDetector(recorder, explosion_window=3, explosion_sigma=2.0)
backtracker = Backtracker(ckpt_mgr, recorder)
detected_fault = None

def on_epoch_end(record):
    global detected_fault, X, y
    grad_stats   = record.gradient_stats if record.gradient_stats else compute_gradient_stats(net)
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
            return True  # stop training

    # At epoch 2 completion, inject Bug: high LR + unscaled features for epoch 3
    if record.epoch == 2:
        for pg in optimizer.param_groups:
            pg['lr'] = 0.08
        X.mul_(2.0)
        y.mul_(2.0)

trainer = Trainer(net, nn.MSELoss(), optimizer, classification=False)
trainer.add_callback(on_epoch_end)
trainer.train(X, y, epochs=100)

if detected_fault:
    result = backtracker.backtrack(detected_fault, net, optimizer=optimizer)
    print_report(result)
else:
    print("No fault detected - try increasing the learning rate further!")

print(f"\n[DONE] Finished. Audit log saved to {log_path}")
