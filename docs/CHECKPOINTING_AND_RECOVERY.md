# Checkpointing, State Validation & Recovery

MLBlackBox provides an architecture-safe state persistence and rollback mechanism designed to eliminate corrupt states and size-mismatch crashes.

---

## 1. Checkpoint Contents

Every checkpoint snapshot contains complete execution state:
1. `model_state_dict`: Parameter tensors across all layers.
2. `architecture_signature`: Layer dimensions, parameter counts, and parameter keys.
3. `optimizer_state_dict`: Momentums, velocity vectors, and Adam squared-gradient accumulators.
4. `scheduler_state_dict`: Current learning rate schedules and step counters.
5. `epoch`: Exact epoch milestone.
6. `rng_state`: Random number generator states across `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`, `numpy.random.get_state()`, and Python's `random.getstate()`.

---

## 2. Architecture Safety & Mismatch Rejection

Loading weights from one architecture into an incompatible model is a major source of silent corruptions or shape crashes.

```python
from tracker.checkpoint import CheckpointManager

ckpt_mgr = CheckpointManager(checkpoint_dir="checkpoints/run_01")
# If network layer keys or tensor shapes differ from the snapshot:
# CheckpointManager raises ArchitectureMismatchError and refuses to corrupt weights.
```

---

## 3. Seamless Resume Workflow

Resuming training is performed in a single call with optional hyperparameter overrides:

```python
from training.trainer import Trainer

# Resume from epoch 2 with learning rate override and gradient clipping:
trainer.resume(
    checkpoint_target=2,
    epochs=50,
    learning_rate=0.001,
    max_grad_norm=1.0
)
```

### Automatic Recovery Flow:
```python
# Automatically finds the last known healthy checkpoint and resumes:
trainer.resume_from_last_healthy_checkpoint()
```

---

## 4. Run Lineage & Parent/Child Sessions

When a training run resumes from a prior run or rollback, MLBlackBox tracks lineage in `runs/run_history.json`:
- `run_id`: New unique identifier for the continued run.
- `parent_run_id`: Direct pointer to the preceding run from which state was restored.
- Audit logs record a `RECOVERY_RESUME` event with timestamp and target checkpoint epoch.

---

## 5. Checkpoint Retention & Archive Portability

- **Retention Policies**: `apply_retention_policy(keep_latest_n=5, preserve_last_healthy=True, preserve_best_loss=True)` automates storage management without deleting critical rollback points.
- **Archive Portability**: Export any checkpoint and metadata as a portable `.tar.gz` bundle:
  ```python
  ckpt_mgr.export_checkpoint(epoch=2, output_tar_path="checkpoints/model_ep2.tar.gz")
  ckpt_mgr.import_checkpoint(archive_path="checkpoints/model_ep2.tar.gz")
  ```
