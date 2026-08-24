"""
Test 1 — XOR Convergence Test

Validates that the neural network engine works end-to-end:
  forward pass → loss → backprop → weight update → convergence

Expected: loss < 0.05 after 3000 epochs on the XOR problem.
This test proves the entire Half 1 (engine) is correctly implemented.

Run:
    python -m pytest tests/test_xor.py -v
    or
    python tests/test_xor.py
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import Network
from core.activations import ReLU, Sigmoid
from core.loss import MSE
from training.trainer import Trainer
from data.xor_data import get_xor_data


def test_xor_convergence():
    """XOR loss should converge below 0.05 after 3000 epochs."""
    X, y = get_xor_data()

    # Architecture: 2 inputs → 4 hidden (ReLU) → 1 output (Sigmoid)
    net = Network([2, 4, 1], activations=[ReLU(), Sigmoid()], seed=42)

    trainer = Trainer(
        network=net,
        loss_fn=MSE(),
        learning_rate=0.1,
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
    X, y = get_xor_data()
    net = Network([2, 4, 1], activations=[ReLU(), Sigmoid()], seed=42)
    trainer = Trainer(net, MSE(), learning_rate=0.1, print_every=0)
    trainer.train(X, y, epochs=3000)

    print("\nXOR Prediction Check:")
    all_correct = True
    for x_sample, y_sample in zip(X, y):
        pred = net.forward(x_sample)[0]
        pred_class = 1 if pred >= 0.5 else 0
        expected = int(y_sample[0])
        status = "✓" if pred_class == expected else "✗"
        print(f"  Input {x_sample} → pred={pred:.4f} class={pred_class} expected={expected} {status}")
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
