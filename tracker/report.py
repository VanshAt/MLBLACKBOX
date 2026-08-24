"""
report.py — Plain English fault report generator for MLBlackBox.

Converts a BacktrackResult into a structured, human-readable report.
This is the original contribution of the system: no other framework
produces this kind of plain-language root cause explanation.

Output format (also returned as a dict for the Streamlit dashboard):
--------------------------------------------------------------------
FAULT DETECTED
Type: Gradient Explosion
First appeared: Epoch 14
Detected at:   Epoch 16

EVIDENCE:
  Gradient magnitude: 0.31 (epoch 13) → 0.38 (epoch 14) → 847 (epoch 15)
  Fault layer:        Layer 2
  Loss at fault:      2847.00 (was 0.34 at clean epoch)
  Batch index:        47

ROOT CAUSE:
  [plain English explanation of why this happened]

SUGGESTED FIXES:
  1. Normalize input features to range [0, 1]
  2. Reduce learning rate from 0.01 to 0.001
  3. Add gradient clipping
  4. Resume training from checkpoint at epoch 13

RESTORED STATE:
  Network restored to epoch 13 weights.
  Ready to resume training.
--------------------------------------------------------------------
"""

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tracker.backtracker import BacktrackResult


def generate_report(result: "BacktrackResult") -> str:
    """
    Generate a plain-English fault report from a BacktrackResult.

    Args:
        result: BacktrackResult from Backtracker.backtrack()

    Returns:
        str: formatted multi-line report string
    """
    fault = result.fault
    rc = result.root_cause
    diff = rc.get("metric_diff", {})

    sep = "=" * 60
    lines = [sep, "  MLBLACKBOX FAULT REPORT", sep, ""]

    # --- Fault summary ---
    lines.append("FAULT DETECTED")
    lines.append(f"  Type:          {fault.fault_type}")
    lines.append(f"  Severity:      {fault.severity.upper()}")
    lines.append(f"  First appeared: Epoch {fault.first_epoch}")
    lines.append(f"  Detected at:   Epoch {fault.detected_epoch}")
    lines.append("")

    # --- Evidence ---
    lines.append("EVIDENCE:")

    clean_ep = diff.get("clean_epoch", "?")
    fault_ep = diff.get("fault_epoch", "?")

    clean_loss = diff.get("loss", {}).get("clean", "N/A")
    fault_loss = diff.get("loss", {}).get("fault", "N/A")
    clean_grad = diff.get("gradient_mean", {}).get("clean", "N/A")
    fault_grad = diff.get("gradient_mean", {}).get("fault", "N/A")

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    lines.append(
        f"  Loss:               {fmt(clean_loss)} (epoch {clean_ep})"
        f" -> {fmt(fault_loss)} (epoch {fault_ep})"
    )
    lines.append(
        f"  Gradient mean:      {fmt(clean_grad)} (epoch {clean_ep})"
        f" -> {fmt(fault_grad)} (epoch {fault_ep})"
    )

    suspected_layer = diff.get("suspected_layer")
    if suspected_layer is not None:
        lines.append(f"  Fault layer:        Layer {suspected_layer}")

    batch_idx = diff.get("batch_idx")
    if batch_idx is not None:
        lines.append(f"  Causative batch:    Batch index {batch_idx}")

    # Raw fault evidence
    if fault.evidence:
        lines.append("  Additional evidence:")
        for k, v in fault.evidence.items():
            lines.append(f"    {k}: {fmt(v) if isinstance(v, float) else v}")

    lines.append("")

    # --- Root Cause ---
    lines.append("ROOT CAUSE:")
    cause_text = rc.get("cause", "Unknown.")
    # Wrap long lines at ~70 chars
    for sentence in cause_text.split(". "):
        sentence = sentence.strip()
        if sentence:
            lines.append(f"  {sentence}.")
    lines.append("")

    # --- Fixes ---
    lines.append("SUGGESTED FIXES:")
    fixes = rc.get("fixes", [])
    for i, fix in enumerate(fixes, 1):
        lines.append(f"  {i}. {fix}")
    lines.append("")

    # --- Restoration status ---
    lines.append("RESTORED STATE:")
    if result.network_restored:
        lines.append(
            f"  Network weights restored to epoch {result.clean_epoch} checkpoint."
        )
        lines.append("  Ready to resume training with adjusted hyperparameters.")
    else:
        lines.append("  WARNING: No clean checkpoint available. Could not restore.")
        lines.append("  Recommendation: restart training from epoch 1.")

    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


def report_to_dict(result: "BacktrackResult") -> dict:
    """
    Convert a BacktrackResult to a structured dict for the Streamlit dashboard.

    Returns:
        dict with keys: fault_type, severity, first_epoch, detected_epoch,
        clean_epoch, cause, fixes, metric_diff, network_restored
    """
    fault = result.fault
    rc = result.root_cause

    return {
        "fault_type": fault.fault_type,
        "severity": fault.severity,
        "first_epoch": fault.first_epoch,
        "detected_epoch": fault.detected_epoch,
        "clean_epoch": result.clean_epoch,
        "description": fault.description,
        "cause": rc.get("cause", ""),
        "fixes": rc.get("fixes", []),
        "metric_diff": rc.get("metric_diff", {}),
        "evidence": fault.evidence,
        "suspected_layer": fault.suspected_layer,
        "batch_idx": fault.batch_idx,
        "network_restored": result.network_restored,
    }


def print_report(result: "BacktrackResult"):
    """Convenience function: generate and print the report to stdout."""
    print(generate_report(result))
