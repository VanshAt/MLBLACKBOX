# MLBlackBox Product Roadmap

Strategic technical milestones for subsequent MLBlackBox releases.

---

## 1. Near-Term Improvements (v3.3)
- **Live WebSocket Telemetry Streamer**: Real-time bidirectional streaming between remote training scripts and the Streamlit dashboard without requiring manual file refreshes.
- **Preemptive Emergency Auto-Checkpointing**: Automatically capture a designated emergency snapshot whenever the Failure Forecaster flags `High` or `Critical` risk.
- **Enhanced Learning Rate Schedulers**: Deeper native integrations with PyTorch cosine annealing and warm restart schedules.

---

## 2. Medium-Term Improvements (v3.5)
- **Distributed Training Observability (DDP & FSDP)**: Multi-GPU and multi-node telemetry aggregation across ranks using PyTorch `torch.distributed` collective reductions.
- **HuggingFace & Lightning Integrations**: Seamless one-line plugin callbacks for HuggingFace `Trainer` and PyTorch Lightning `Callback`.
- **Automated Hyperparameter Branching**: Interactive "Fork & Tune" buttons that spin up isolated trial branches directly from the dashboard.

---

## 3. Long-Term Vision (v4.0)
- **Autonomous Training Self-Healing**: Automated policy-driven remediation that pauses training on critical faults, applies the AI Training Doctor's recommended rate adjustments, and resumes automatically with human oversight logging.
- **Cluster Fleet Monitoring**: Unified multi-node dashboard for tracking dozens of concurrent training jobs across cloud Kubernetes GPU clusters.
