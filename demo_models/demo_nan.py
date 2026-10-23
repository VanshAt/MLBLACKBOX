"""
DEMO 3 — NaN Propagation
==========================
The Bug:  The dataset contains a single data point with value = 0.0
          fed into a log() operation inside a custom loss function.
          log(0) = -infinity → NaN spreads to all weights.

What happens: One bad value → loss becomes NaN → all gradients become NaN
              → every weight in the network becomes NaN.
              After this the model is completely broken.

MLBlackBox detects: NAN_INF_PROPAGATION (Critical)
Dashboard shows:    Loss suddenly shows NaN/Inf.
                    Gradient Health shows "Has NaN: True" for every layer.

Run: python demo_models/demo_nan.py
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

parser = argparse.ArgumentParser(description="Demo 3: NaN Propagation")
parser.add_argument("--fresh", action="store_true", help="Clear previous demo checkpoints and logs")
args = parser.parse_args()

ckpt_dir = os.path.join("checkpoints", "demo_nan")
log_path = os.path.join("logs", "demo_nan_audit.json")

if args.fresh:
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    if os.path.exists(log_path):
        os.remove(log_path)

os.makedirs("logs", exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

print("=" * 60)
print("DEMO 3 - NaN Propagation (log(0) in Custom Loss)")
print("=" * 60)


class UnsafeBCELoss(nn.Module):
    """
    A manually written BCE loss WITHOUT the numerical safety clamp.
    Real PyTorch BCELoss clamps predictions to [1e-7, 1-1e-7] to avoid log(0).
    This version doesn't — so when prediction = 0.0 or 1.0, log(0) = -inf -> NaN.
    """
    def forward(self, pred, target):
        # BUG: no clamp! If pred = 0.0 or 1.0, log(0) = -inf, leading to NaN
        eps = 0.0  # intentional bug
        return -(target * torch.log(pred + eps) + (1 - target) * torch.log(1 - pred + eps)).mean()


net = nn.Sequential(
    nn.Linear(2, 8), nn.ReLU(),
    nn.Linear(8, 4), nn.ReLU(),
    nn.Linear(4, 1), nn.Sigmoid()
)

optimizer = optim.SGD(net.parameters(), lr=0.1)

# Synthetic data
torch.manual_seed(99)
X = torch.rand(100, 2)
y = (X[:, 0] > X[:, 1]).float().unsqueeze(1)

# In order to trigger NaN at epoch 3 (so checkpoints 1 & 2 are saved cleanly):
# We inject extreme input that forces Sigmoid to saturate to 0.0 or 1.0 after epoch 2
def inject_nan_trigger(epoch):
    if epoch >= 3:
        # Extreme negative input forces Sigmoid output to exactly 0.0 with target 1.0 -> log(0) = -inf -> NaN
        X[50] = torch.tensor([-100.0, -100.0])
        y[50] = 1.0

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

    if record.epoch >= 3 or record.nan_layer >= 0:
        fault = detector.check(epoch_rec)
        if fault:
            detected_fault = fault
            return True

    # After epoch 2, inject extreme input that forces Sigmoid to saturate to 0.0 with target 1.0 -> log(0) = -inf -> NaN in epoch 3
    if record.epoch == 2:
        X[50] = torch.tensor([-100.0, -100.0])
        y[50] = 1.0

trainer = Trainer(net, UnsafeBCELoss(), optimizer, classification=True)
trainer.add_callback(on_epoch_end)
trainer.train(X, y, epochs=100)

if detected_fault:
    result = backtracker.backtrack(detected_fault, net, optimizer=optimizer)
    print_report(result)
else:
    print("Ran without NaN - the bug didn't trigger this time.")

print(f"\n[DONE] Finished. Audit log saved to {log_path}")
