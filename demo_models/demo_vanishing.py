"""
DEMO 2 — Gradient Vanishing
=============================
The Bug:  A deep network (8 layers) using Sigmoid activations everywhere.

What happens: Sigmoid squashes gradients toward 0. By the time the gradient
              travels from layer 8 back to layer 1 (backprop), it's been
              multiplied by ~0.25 eight times → 0.25^8 ≈ 0.0000153.
              The early layers learn nothing.

MLBlackBox detects: TRAINING_STAGNATION or GRADIENT_VANISHING
Dashboard shows:    Gradient Health chart flatlines near zero.
                    Loss barely decreases despite many epochs.

Run: python demo_models/demo_vanishing.py
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

parser = argparse.ArgumentParser(description="Demo 2: Gradient Vanishing")
parser.add_argument("--fresh", action="store_true", help="Clear previous demo checkpoints and logs")
args = parser.parse_args()

ckpt_dir = os.path.join("checkpoints", "demo_vanishing")
log_path = os.path.join("logs", "demo_vanishing_audit.json")

if args.fresh:
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    if os.path.exists(log_path):
        os.remove(log_path)

os.makedirs("logs", exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

print("=" * 60)
print("DEMO 2 - Gradient Vanishing (Too Many Sigmoid Layers)")
print("=" * 60)

# ── The Buggy Model ───────────────────────────────────────────
# BUG: 8 layers all using Sigmoid — gradients vanish by layer 3
net = nn.Sequential(
    nn.Linear(4, 32), nn.Sigmoid(),   # layer 1 — gradient: 0.25 of output
    nn.Linear(32, 32), nn.Sigmoid(),  # layer 2 — gradient: 0.25 * 0.25
    nn.Linear(32, 32), nn.Sigmoid(),  # layer 3 — gradient: ~0.016
    nn.Linear(32, 32), nn.Sigmoid(),  # layer 4 — gradient: ~0.004
    nn.Linear(32, 16), nn.Sigmoid(),  # layer 5 — gradient: ~0.001
    nn.Linear(16, 8),  nn.Sigmoid(),  # layer 6 — gradient: ~0.0002
    nn.Linear(8, 4),   nn.Sigmoid(),  # layer 7 — gradient: ~0.00006
    nn.Linear(4, 1),   nn.Sigmoid()   # output
)

# FIX would be: replace all Sigmoid with ReLU except the last layer

optimizer = optim.SGD(net.parameters(), lr=0.01)

# Binary classification task — predict if sum of inputs > 2.0
torch.manual_seed(7)
X = torch.rand(300, 4)
y = ((X.sum(dim=1) > 2.0).float()).unsqueeze(1)

# ── MLBlackBox Setup ──────────────────────────────────────────
recorder    = MetricRecorder(log_path=log_path)
ckpt_mgr    = CheckpointManager(checkpoint_dir=ckpt_dir)
detector    = FaultDetector(recorder)
backtracker = Backtracker(ckpt_mgr, recorder)
detected_fault = None

def on_epoch_end(record):
    global detected_fault
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
    if record.epoch >= 3:
        fault = detector.check(epoch_rec)
        if fault:
            detected_fault = fault
            return True

trainer = Trainer(net, nn.BCELoss(), optimizer, classification=True)
trainer.add_callback(on_epoch_end)
trainer.train(X, y, epochs=200)

if detected_fault:
    result = backtracker.backtrack(detected_fault, net, optimizer=optimizer)
    print_report(result)
else:
    print("No fault detected yet - the network is silently failing!")

print(f"\n[DONE] Finished. Audit log saved to {log_path}")
