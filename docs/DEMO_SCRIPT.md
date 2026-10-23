# MLBlackBox Demonstration Scripts

Presenting MLBlackBox in live reviews, tech talks, or investor demos.

---

## 1. The 60-Second Elevator Pitch Demo

### Terminal Commands:
```bash
python demo_models/demo_explosion.py --fresh
```

### Script & Speaking Points:
1. **Show the Problem** (15s):
   *“We set an aggressive learning rate on unnormalized inputs. In standard PyTorch, this silently diverges into infinity and wastes compute.”*
2. **Watch MLBlackBox Intervene** (20s):
   *“Notice what just happened at Epoch 3: MLBlackBox detected a gradient explosion immediately before parameter corruption, halted execution, and automatically rewound model weights, optimizer buffers, and RNG states back to the clean Epoch 2 checkpoint.”*
3. **Show Root-Cause Explanation** (25s):
   *“It didn't just crash. It output an automated diagnostic report pinpointing the exact offending layer (`Layer 4`), the causative batch index (`199`), and the exact fix: add `clip_grad_norm_` or reduce learning rate by 10x.”*

---

## 2. The 3-Minute Comprehensive Technical Walkthrough

### Step 1: Launch the Control Room Dashboard (30s)
```bash
streamlit run dashboard/app.py
```
- Point out the **dark Control Room aesthetic**: the **Training Health Score Dial** (`94/100`) and the **Early-Warning Failure Forecast Card** in the hero section.

### Step 2: Precursor Forecasting & Risk Detection (45s)
- Navigate to **8. Failure Forecast**.
- Show the **Risk-Over-Time chart** and the **Per-Fault Risk Breakdown**.
- Explain how rolling z-scores detect gradient acceleration *before* fatal crashes occur, without training heavy surrogate neural nets.

### Step 3: Layer Health & Forensic Heatmaps (45s)
- Navigate to **3. Layer Health**.
- Review the **Per-Layer Risk Matrix** and top-3 alert cards showing trend indicators (`EXPLODING`, `VANISHING`, `STABLE`).

### Step 4: AI Training Doctor in Action (60s)
- Navigate to **7. AI Training Doctor**.
- Highlight that the engine is **100% offline, zero-dependency, and private by design**.
- Review the structured diagnosis, evidence citations, and one-click copyable PyTorch code snippets.
