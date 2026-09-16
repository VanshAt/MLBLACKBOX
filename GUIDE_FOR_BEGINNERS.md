# MLBlackBox — A Beginner's Complete Guide 📖

> This guide assumes you know **nothing** about machine learning, neural networks, or Python beyond the basics.
> By the end, you'll understand exactly what every file in this project does and why.

---

## Part 1 — What is a Neural Network?

### The simplest possible explanation

Imagine you want to teach a computer to tell the difference between a cat photo and a dog photo.

You can't write rules like "if it has pointy ears it's a cat" — there are too many exceptions. Instead, you **show it thousands of examples** and let it figure out the rules on its own. That process is called **training a neural network**.

### How does it actually learn?

A neural network is just a bunch of numbers (called **weights**) connected together in layers:

```
Input Layer        Hidden Layer        Output Layer
(your data)    →   (computations)  →   (your answer)

[Photo pixels] → [Layer 1] → [Layer 2] → [Cat? Dog?]
```

At the start, all the weights are **random** — the network knows nothing and gives garbage answers.

Training works like this:
1. Feed it an example (a cat photo)
2. It makes a guess (e.g., "70% dog")
3. We measure how wrong it was — this is called the **loss** (here, very wrong)
4. We nudge all the weights slightly in the direction that would have made a better guess
5. Repeat millions of times

After enough repetitions, the weights have been nudged into values that produce correct answers. The network has "learned".

### What are Gradients?

Step 4 above — "nudge the weights" — is where **gradients** come in.

A gradient is just a number that tells us:
- **Which direction** to nudge a weight (increase or decrease it)
- **How much** to nudge it

Think of it like standing on a hilly landscape and trying to find the lowest valley (lowest loss). The gradient tells you which direction is downhill. You take a small step that direction. That's called **gradient descent**.

```
High Loss ↑
           \
            \   ← You are here
             \
              \_____ Low Loss (goal)
```

---

## Part 2 — What Can Go Wrong During Training?

Neural network training fails **all the time**. Here are the most common ways:

### 🔴 Gradient Explosion
The gradient numbers become astronomically large (like `1e15`).
When you multiply a huge number by the weight, the weight update becomes enormous.
The network stops learning and produces NaN (Not a Number) everywhere.

**Analogy:** You're trying to walk downhill but you take a step so large you fly off the mountain entirely.

### 🟠 Gradient Vanishing
The gradient becomes so tiny (like `0.000000001`) that the weight barely moves at all.
This happens especially in very deep networks using the "Sigmoid" activation function.
The network appears to train but makes no real progress.

**Analogy:** You're walking downhill but each step is 1 millimetre. You'll never reach the valley in your lifetime.

### ⚫ NaN / Inf Propagation
`NaN` = "Not a Number". This happens when a calculation produces an impossible result (like `log(0)` or dividing by zero).
Once one NaN appears, it spreads — any math involving NaN produces more NaN.
The entire network becomes useless.

**Analogy:** One corrupted cell in a spreadsheet formula spreads to every cell that depends on it.

### 🟡 Loss Spike
The loss was going down nicely, then suddenly jumps 5x in one epoch.
Usually caused by one single "outlier" data point with extreme values contaminating the gradient.

**Analogy:** You're averaging test scores of 80, 82, 79, 81... then someone scores 0 and the average crashes.

### 🔵 Training Stagnation
The loss stops decreasing. The network is stuck.
This can mean the network is in a "saddle point" — a flat region of the loss landscape where every direction looks the same.

**Analogy:** You're trying to find the lowest valley but you're standing in a perfectly flat desert. Every direction looks identical, so you don't move.

---

## Part 3 — What is MLBlackBox?

MLBlackBox is a system that **watches your PyTorch neural network train** and:

1. **Records** everything that happens every epoch (like a flight data recorder)
2. **Detects** the 5 fault types described above, automatically
3. **Saves snapshots** of the model at every epoch (so you can go back in time)
4. **Rewinds** the model to the last healthy state when a fault is found
5. **Explains** in plain English what went wrong and how to fix it
6. **Visualises** everything on a live web dashboard

---

## Part 4 — Key Vocabulary

Before reading the code, you need to know these words:

| Word | What it means |
|---|---|
| **Epoch** | One complete pass through your entire training dataset |
| **Loss** | A number measuring how wrong the network is. Lower = better. |
| **Gradient** | The "nudge direction" for each weight. Computed automatically by PyTorch. |
| **Weight** | A single number inside the network that gets adjusted during training |
| **Checkpoint** | A saved snapshot of all the network's weights at a specific epoch |
| **Callback** | A function that automatically runs after each epoch |
| **Audit Log** | A file recording every epoch's stats — like a pilot's black box |
| **Backtracking** | Loading a previous checkpoint to undo recent training damage |
| **Root Cause** | The actual reason a fault happened (not just "the loss went up") |
| **`nn.Module`** | PyTorch's name for a neural network model |
| **`state_dict`** | PyTorch's way of saving all weights as a dictionary |
| **`.pt` file** | A PyTorch saved model file (like `.docx` but for neural networks) |
| **`.json` file** | A simple text file that stores data in key-value pairs |

---

## Part 5 — The Project File by File

### 📄 `training/trainer.py` — The Training Loop

This is the **engine** of the project. It runs the actual training.

```
Your data → Trainer → PyTorch model → Loss → Gradients → Updated weights
                ↓
           Calls your callback after each epoch
```

**What it does step by step:**
1. Takes your model, loss function, and optimizer as inputs
2. For each epoch:
   - Runs all your data through the model (forward pass)
   - Calculates how wrong the answer was (loss)
   - Calculates gradients using `loss.backward()`
   - Updates weights using `optimizer.step()`
   - Checks if any gradients are NaN
   - Calls your callback function with a record of what happened
3. If your callback returns `True`, training stops early (used to halt on fault detection)

**The `EpochRecord` it gives your callback contains:**
- `epoch` — which epoch just finished
- `loss` — the loss value
- `accuracy` — percentage correct (for classification tasks)
- `learning_rate` — how big the weight update steps are
- `epoch_time_ms` — how long the epoch took in milliseconds
- `nan_layer` — which layer first produced NaN (-1 if none)

---

### 📄 `tracker/gradient_tracker.py` — Watching the Gradients

This file reads the gradient numbers out of your PyTorch model after each backward pass.

```python
grad_stats = compute_gradient_stats(net)
```

**What it computes:**
- For each layer in your network, it reads `param.grad` (PyTorch stores gradients here automatically after `.backward()`)
- It calculates: **mean**, **max**, **min**, **std** (standard deviation), and whether any value is **NaN**
- It also computes the same stats **globally** across all layers combined

**Why per-layer?**
Global stats hide problems. If Layer 1 has a gradient of 1000 and Layer 8 has a gradient of 0.000001, the global average might look "normal". Per-layer tracking catches this.

---

### 📄 `tracker/checkpoint.py` — Saving Snapshots

Every epoch, this saves a complete copy of your model so you can rewind to it later.

**It creates two files per epoch:**
- `checkpoint_epoch_00001.pt` — the actual model weights (binary, PyTorch format)
- `checkpoint_epoch_00001.json` — the metadata: loss, accuracy, gradient stats, timestamp

**Why two files?**
The Backtracker needs to scan metadata for thousands of checkpoints quickly (to find the last "clean" one). Reading `.pt` files is slow. The `.json` files are tiny text files it can scan instantly. Only when it decides which checkpoint to restore does it load the `.pt`.

---

### 📄 `tracker/metric_recorder.py` — The Flight Data Recorder

This appends every epoch's full record to `logs/audit_log.json`.

Think of it as the **permanent memory** of the training run. The dashboard reads this file to draw all the charts. The fault detector reads it to compute rolling statistics.

**Each record looks like:**
```json
{
  "epoch": 42,
  "loss": 0.0341,
  "accuracy": 0.9133,
  "gradient": {
    "global": { "mean": 0.0023, "max": 0.018, "has_nan": false },
    "per_layer": [...]
  },
  "learning_rate": 0.05,
  "epoch_time_ms": 12.4
}
```

---

### 📄 `tracker/fault_detector.py` — The Alarm System

This is the brain of the fault detection system. After each epoch, it checks the latest record against 5 algorithms:

**Algorithm 1 — Gradient Explosion:**
Keeps a rolling window of the last 10 gradient means.
Computes `rolling_mean` and `rolling_std`.
If current gradient > `rolling_mean + 3 × rolling_std` → EXPLOSION.
(3 standard deviations above average is statistically unusual)

**Algorithm 2 — Gradient Vanishing:**
If the global gradient mean is below `1e-6` AND accuracy hasn't improved in 20 epochs → VANISHING.
(Both conditions needed to avoid false alarms during normal early training)

**Algorithm 3 — NaN / Inf:**
If the trainer detected a NaN layer, or if the gradient stats contain NaN/Inf → NAN_INF_PROPAGATION.

**Algorithm 4 — Loss Spike:**
If current loss > previous loss × 5 → LOSS_SPIKE.

**Algorithm 5 — Stagnation:**
Keeps a counter of epochs where loss hasn't decreased by at least 0.1%.
If counter > 15 and loss is still high → TRAINING_STAGNATION.

When a fault is found, it returns a `Fault` object with: `fault_type`, `severity`, `description`, `epoch`, `evidence`.

---

### 📄 `tracker/backtracker.py` — The Time Machine

When a fault is detected, the Backtracker:

1. **Finds the last clean checkpoint** — scans the `.json` metadata files backwards from the fault epoch, looking for the last epoch where gradients were normal and loss wasn't spiking
2. **Loads that checkpoint** — calls `net.load_state_dict()` to restore the model weights to that earlier state
3. **Generates a root cause report** — looks up the fault type in a cause library and picks the matching explanation and fix list

```python
result = backtracker.backtrack(detected_fault, net)
# net is now restored to epoch 13 (or wherever the last clean state was)
```

---

### 📄 `tracker/report.py` — The Plain English Explainer

Takes the `BacktrackResult` from the Backtracker and formats it into a human-readable report. It maps each `FaultType` to:
- A plain-English explanation of what caused it
- An ordered list of concrete fixes to try
- The evidence (which epoch, which layer, what values changed)

---

### 📄 `dashboard/app.py` — The Web Dashboard

A Streamlit web app that reads `logs/audit_log.json` and displays it visually.

**Tab 1 — Training Curves:** Line charts of loss and accuracy over all epochs. Marks fault epochs in red.

**Tab 2 — Gradient Health:** Line charts of gradient mean and max. Table of per-layer stats for the latest epoch.

**Tab 3 — Fault Log:** Cards showing every detected fault, when it happened, and its severity.

**Tab 4 — Audit Table:** The full epoch-by-epoch raw data table.

**Sidebar:** Shows the audit log path, a Refresh button, and a Backtrack control to restore the model to any saved checkpoint.

---

### 📄 `tests/` — The Test Files

These are scripts that demonstrate the system working end-to-end:

| File | What it tests | What to expect |
|---|---|---|
| `test_xor.py` | Can the system train a model to solve XOR? | Loss drops below 0.05, all 4 XOR inputs correct |
| `test_fault_injection.py` | Does the detector catch injected extreme values? | Fault detected, report printed |
| `test_vanishing.py` | Does the detector catch a deep sigmoid network? | Stagnation or vanishing fault detected |
| `test_iris.py` | Does the full system work on a real dataset? | >90% accuracy, audit log + checkpoints created |

---

### 📄 `data/` — The Datasets

**`xor_data.py`:** Returns the 4 XOR inputs and outputs:
```
[0,0] → 0,  [0,1] → 1,  [1,0] → 1,  [1,1] → 0
```
XOR is the classic test because it's **not linearly separable** — you need at least one hidden layer to solve it.

**`iris_loader.py` + `iris.csv`:** The Iris dataset — 150 flower measurements (petal length, width, sepal length, width) labelled as one of 3 species. Used as a real-world classification test.

---

### 📄 `requirements.txt` — Dependencies

```
torch>=2.0.0    ← PyTorch: the ML framework that runs the neural network
streamlit       ← The web dashboard framework
pandas          ← Data tables in the dashboard
```

---

## Part 6 — How Everything Connects

Here is the full flow from start to finish:

```
You write:  net = nn.Sequential(...)     ← Your PyTorch model
            optimizer = optim.SGD(...)   ← How to update weights

You set up: MetricRecorder              → writes logs/audit_log.json
            CheckpointManager           → writes checkpoints/*.pt + *.json
            FaultDetector               → reads audit log history
            Backtracker                 → reads checkpoints + fault

You define: on_epoch_end(record)
              │
              ├─ compute_gradient_stats(net)   → reads param.grad tensors
              ├─ recorder.append(...)           → appends to audit_log.json
              ├─ ckpt_mgr.save(net, epoch,...) → saves .pt + .json files
              └─ detector.check(epoch_rec)     → runs 5 fault algorithms
                    │
                    └─ fault found?
                          │
                          ├─ return True  →  Trainer stops training
                          │
                          └─ backtracker.backtrack(fault, net)
                                │
                                ├─ Scans .json files for last clean epoch
                                ├─ Loads .pt file → net.load_state_dict()
                                └─ print_report() → plain English output

Meanwhile:  streamlit run dashboard/app.py
              └─ reads audit_log.json every few seconds
              └─ draws live charts in your browser
```

---

## Part 7 — Running the Project Step by Step

### Step 1: Install the requirements
```bash
pip install -r requirements.txt
```

### Step 2: Run the simplest test first (XOR)
```bash
python tests/test_xor.py
```
You'll see the loss printing every 500 epochs, decreasing from ~0.25 down to ~0.001.

### Step 3: Run the fault injection test
```bash
python tests/test_fault_injection.py
```
This deliberately puts a bad data point (value = `1e15`) into the training set. Watch the system catch it.

### Step 4: Generate some audit data with the Iris test
```bash
python tests/test_iris.py
```
This runs 500 epochs on a real dataset and writes to `logs/audit_log.json`.

### Step 5: Open the dashboard
```bash
python -m streamlit run dashboard/app.py
```
Open `http://localhost:8501` in your browser. You'll see all the training data visualised.

---

## Part 8 — Common Questions

**Q: Why does loss go up sometimes?**
A: Normal early in training. Gradients are large and chaotic. The loss bounces before settling. Only a sudden 5x jump after stable training is a "Loss Spike" fault.

**Q: What is the difference between an epoch and a batch?**
A: A **batch** is a small chunk of your data (e.g., 32 examples). An **epoch** is when you've processed all your batches once. If you have 1000 examples and a batch size of 32, one epoch = ~31 batches.

**Q: Why save a checkpoint every epoch? That is a lot of files.**
A: Because you don't know in advance which epoch will be the last "clean" one before a fault. You need all of them so the Backtracker can find the exact right one to restore to.

**Q: What is a learning rate?**
A: It controls how big each weight update step is. Too large → gradient explosion. Too small → training takes forever or stagnates. Typical values: `0.001` to `0.1`.

**Q: What is `nn.Sequential`?**
A: A PyTorch shorthand to stack layers in a straight line. `nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,1))` means: first do a linear transformation (4 inputs → 8 outputs), then apply ReLU activation, then another linear transformation (8 → 1).

**Q: What is ReLU?**
A: An **activation function** — a simple mathematical operation applied after each layer to let the network learn non-linear patterns. ReLU is just `max(0, x)` — negative values become 0, positive values pass through unchanged. This helps gradients flow through deep networks without vanishing.

**Q: What is Sigmoid?**
A: Another activation function. It squashes any number into the range [0, 1]. Used in the output layer for binary classification (e.g., "is this a cat? 0.87"). The problem is it also squashes gradients — for very large or very small inputs, the gradient is nearly zero. That's why stacking many Sigmoid layers causes vanishing gradients.

**Q: What does `loss.backward()` do?**
A: This is PyTorch's magic. It automatically computes the gradient of the loss with respect to every weight in the network using the chain rule of calculus. You don't write the gradient math yourself — PyTorch does it for you. After calling this, every `param.grad` in the network is filled with its gradient value.

**Q: What does `optimizer.step()` do?**
A: It reads each weight's gradient (`param.grad`) and updates the weight accordingly. For SGD (Stochastic Gradient Descent): `new_weight = old_weight - learning_rate × gradient`.
