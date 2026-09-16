"""
Test 1 — XOR Convergence Test

Validates that the PyTorch neural network engine works end-to-end:
  forward pass → loss → backprop → weight update → convergence
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer import Trainer
from data.xor_data import get_xor_data


def test_xor_convergence():
    """XOR loss should converge below 0.05 after 3000 epochs."""
    torch.manual_seed(42)
    X_raw, y_raw = get_xor_data()
    X = torch.tensor(X_raw, dtype=torch.float32)
    y = torch.tensor(y_raw, dtype=torch.float32)

    # Architecture: 2 inputs → 4 hidden (ReLU) → 1 output (Sigmoid)
    net = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
        nn.Sigmoid()
    )

    optimizer = optim.SGD(net.parameters(), lr=0.1)
    
    trainer = Trainer(
        network=net,
        loss_fn=nn.MSELoss(),
        optimizer=optimizer,
        classification=False,
        print_every=500,
    )

    history = trainer.train(X, y, epochs=3000)
    final_loss = history[-1]["loss"]

    print(f"\nXOR Final Loss: {final_loss:.6f}")
    assert final_loss < 0.05, (
        f"XOR test FAILED: final loss {final_loss:.6f} >= 0.05. "
        "Check backpropagation and weight update implementation."
    )
    print("XOR test PASSED ✓")
    return final_loss


def test_xor_predictions():
    """After training, predictions should match XOR truth table."""
    torch.manual_seed(42)
    X_raw, y_raw = get_xor_data()
    X = torch.tensor(X_raw, dtype=torch.float32)
    y = torch.tensor(y_raw, dtype=torch.float32)

    net = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
        nn.Sigmoid()
    )

    optimizer = optim.SGD(net.parameters(), lr=0.1)

    trainer = Trainer(
        network=net, 
        loss_fn=nn.MSELoss(), 
        optimizer=optimizer,
        print_every=0
    )
    trainer.train(X, y, epochs=3000)

    print("\nXOR Prediction Check:")
    all_correct = True
    for i in range(len(X)):
        x_sample = X[i:i+1]
        y_sample = y[i:i+1]
        
        pred = net(x_sample).item()
        pred_class = 1 if pred >= 0.5 else 0
        expected = int(y_sample.item())
        status = "✓" if pred_class == expected else "✗"
        print(f"  Input {X_raw[i]} → pred={pred:.4f} class={pred_class} expected={expected} {status}")
        if pred_class != expected:
            all_correct = False

    assert all_correct, "XOR predictions test FAILED: not all inputs classified correctly."
    print("XOR prediction test PASSED ✓")


if __name__ == "__main__":
    print("=" * 50)
    print("Test 1 — XOR Convergence")
    print("=" * 50)
    test_xor_convergence()
    test_xor_predictions()
