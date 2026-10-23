/**
 * MLBlackBox Flight Recorder & Control Room Interactive Simulator
 */
(function() {
  'use strict';

  // Scenarios Data
  const SCENARIOS = {
    explosion: {
      id: 'explosion',
      name: 'Gradient Explosion',
      badge: 'FAULT TRIGGERED',
      badgeClass: 'badge-error',
      epochs: [
        { epoch: 1, loss: 2.14, gradNorm: 0.18, health: 98, status: 'NOMINAL', event: 'Checkpoint saved at epoch 1 (disk size: 420 KB)' },
        { epoch: 2, loss: 1.62, gradNorm: 0.24, health: 96, status: 'NOMINAL', event: 'Checkpoint saved at epoch 2 (healthy baseline)' },
        { epoch: 3, loss: 12.85, gradNorm: 14.82, health: 22, status: 'FAULT: GRADIENT_EXPLOSION', event: 'CRITICAL: Gradient norm 14.82 exceeds threshold (10.0) at layer 16.weight!' },
        { epoch: 4, loss: 1.62, gradNorm: 0.24, health: 92, status: 'RECOVERED', event: 'FLIGHT RECORDER INTERVENTION: Training halted. Restored clean checkpoint from epoch 2.' },
        { epoch: 5, loss: 1.38, gradNorm: 0.21, health: 95, status: 'NOMINAL', event: 'Resumed with gradient clipping enabled (clip_norm=1.0).' }
      ],
      doctor: {
        title: 'Diagnostic: Gradient Explosion in Output Projection',
        layer: 'Layer 16 (fc_out.weight)',
        confidence: '99.4%',
        rootCause: 'Large learning rate (0.1) combined with unscaled cross-entropy logits caused gradient norm to surge to 14.82 in backpropagation.',
        prescription: 'Apply gradient clipping at norm threshold 1.0 and reduce initial learning rate to 1e-4.',
        code: `# AI Training Doctor Prescription:
import torch.nn.utils as nn_utils

# 1. Add gradient clipping in training loop:
nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Adjust optimizer learning rate:
for param_group in optimizer.param_groups:
    param_group['lr'] = 1e-4`
      }
    },

    vanishing: {
      id: 'vanishing',
      name: 'Vanishing Gradient',
      badge: 'STAGNATION DETECTED',
      badgeClass: 'badge-warn',
      epochs: [
        { epoch: 1, loss: 2.45, gradNorm: 0.08, health: 92, status: 'NOMINAL', event: 'Checkpoint saved at epoch 1' },
        { epoch: 2, loss: 2.44, gradNorm: 0.009, health: 80, status: 'WARNING', event: 'Warning: Gradient norm dropping rapidly across deep layers' },
        { epoch: 3, loss: 2.44, gradNorm: 0.0004, health: 62, status: 'WARNING', event: 'Layer 0.weight gradient norm: 3.2e-7 (below noise floor)' },
        { epoch: 4, loss: 2.43, gradNorm: 0.00008, health: 45, status: 'STAGNANT', event: 'Loss change < 0.001 for 3 consecutive epochs' },
        { epoch: 5, loss: 2.43, gradNorm: 0.00002, health: 38, status: 'FAULT: VANISHING_GRADIENT', event: 'FAULT: Vanishing gradients detected on layers 0..4 (attenuation factor: 420x)' },
        { epoch: 6, loss: 1.95, gradNorm: 0.12, health: 88, status: 'RECOVERED', event: 'Rollback to epoch 1. Replaced Sigmoid activations with LeakyReLU.' }
      ],
      doctor: {
        title: 'Diagnostic: Vanishing Gradient via Sigmoid Saturation',
        layer: 'Layers 0–4 (backbone.conv1, conv2)',
        confidence: '96.8%',
        rootCause: 'Repeated non-linearities (Sigmoid/Tanh) compressed gradients below floating point precision in early layers.',
        prescription: 'Switch activation function to GELU/LeakyReLU and employ He (Kaiming) weight initialization.',
        code: `# AI Training Doctor Prescription:
# Replace saturated activations:
model.act = torch.nn.GELU()

# Apply Kaiming normal initialization:
for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.kaiming_normal_(p)`
      }
    },

    nan_loss: {
      id: 'nan_loss',
      name: 'NaN Loss Instability',
      badge: 'NUMERICAL COLLAPSE',
      badgeClass: 'badge-error',
      epochs: [
        { epoch: 1, loss: 1.85, gradNorm: 0.22, health: 97, status: 'NOMINAL', event: 'Checkpoint saved at epoch 1' },
        { epoch: 2, loss: 1.41, gradNorm: 0.28, health: 95, status: 'NOMINAL', event: 'Checkpoint saved at epoch 2' },
        { epoch: 3, loss: NaN, gradNorm: NaN, health: 0, status: 'FAULT: NAN_LOSS', event: 'EMERGENCY: Loss evaluated to NaN! Tensor corruption imminent.' },
        { epoch: 4, loss: 1.41, gradNorm: 0.28, health: 90, status: 'RECOVERED', event: 'FLIGHT RECORDER: Atomic rollback restored Epoch 2 weights without disk corruption.' }
      ],
      doctor: {
        title: 'Diagnostic: Numerical Instability / Zero Division',
        layer: 'Loss Function (criterion)',
        confidence: '99.9%',
        rootCause: 'Logarithm of zero in un-clamped cross entropy or 16-bit float underflow in mixed precision.',
        prescription: 'Enable torch.autograd.set_detect_anomaly(True) and add epsilon clamp to probability distributions.',
        code: `# AI Training Doctor Prescription:
import torch

# Enable anomaly tracking for exact stack trace:
torch.autograd.set_detect_anomaly(True)

# Clamp logits or probabilities before log:
probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)`
      }
    },

    nominal: {
      id: 'nominal',
      name: 'Healthy Training Run',
      badge: 'NOMINAL FLIGHT',
      badgeClass: 'badge-success',
      epochs: [
        { epoch: 1, loss: 2.35, gradNorm: 0.25, health: 96, status: 'NOMINAL', event: 'Flight recorder active. Baseline metrics recorded.' },
        { epoch: 2, loss: 1.82, gradNorm: 0.22, health: 97, status: 'NOMINAL', event: 'Steady convergence. Checkpoint saved at epoch 2.' },
        { epoch: 3, loss: 1.45, gradNorm: 0.19, health: 98, status: 'NOMINAL', event: 'Gradient variance stable across all 18 layers.' },
        { epoch: 4, loss: 1.15, gradNorm: 0.17, health: 98, status: 'NOMINAL', event: 'Loss decreasing monotonically. Checkpoint saved.' },
        { epoch: 5, loss: 0.88, gradNorm: 0.15, health: 99, status: 'NOMINAL', event: 'Learning rate scheduler stepped: lr = 0.005.' },
        { epoch: 6, loss: 0.65, gradNorm: 0.14, health: 99, status: 'NOMINAL', event: 'Validation accuracy reached 94.2%.' },
        { epoch: 7, loss: 0.42, gradNorm: 0.12, health: 99, status: 'NOMINAL', event: 'Training completed successfully with zero flight faults.' }
      ],
      doctor: {
        title: 'Diagnostic: System Nominal — Healthy Optimization',
        layer: 'All 18 layers healthy',
        confidence: '98.5%',
        rootCause: 'No gradient anomalies or numerical instabilities detected. Model parameters converging smoothly.',
        prescription: 'No intervention required. Current hyperparameters are optimal.',
        code: `# Status: All parameters optimal
# MLBlackBox Flight Recorder is monitoring in background.
# Overhead: 1.18% wall-clock latency.`
      }
    }
  };

  // State
  let currentScenario = 'explosion';
  let currentStep = 0;
  let isPlaying = false;
  let playInterval = null;

  // DOM Elements
  const canvas = document.getElementById('telemetryCanvas');
  const ctx = canvas ? canvas.getContext('2d') : null;
  const terminal = document.getElementById('terminalLog');
  const healthScoreEl = document.getElementById('healthScoreVal');
  const healthBarEl = document.getElementById('healthBarFill');
  const systemStatusBadge = document.getElementById('systemStatusBadge');
  const playBtn = document.getElementById('playBtn');
  const stepBtn = document.getElementById('stepBtn');
  const resetBtn = document.getElementById('resetBtn');
  const scenarioBtns = document.querySelectorAll('.scenario-btn');

  // Doctor elements
  const docTitle = document.getElementById('doctorTitle');
  const docLayer = document.getElementById('doctorLayer');
  const docRootCause = document.getElementById('doctorRootCause');
  const docPrescription = document.getElementById('doctorPrescription');
  const docCode = document.getElementById('doctorCode');

  // Initialize Canvas DPI
  function setupCanvas() {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    if (ctx) {
      ctx.scale(dpr, dpr);
    }
  }

  // Draw Telemetry Chart
  function drawChart() {
    if (!ctx || !canvas) return;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    ctx.clearRect(0, 0, width, height);

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    const gridCols = 6;
    const gridRows = 4;
    for (let i = 1; i < gridCols; i++) {
      const x = (width / gridCols) * i;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let j = 1; j < gridRows; j++) {
      const y = (height / gridRows) * j;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    const scenario = SCENARIOS[currentScenario];
    const visibleData = scenario.epochs.slice(0, currentStep + 1);
    if (visibleData.length === 0) return;

    const totalEpochs = scenario.epochs.length;
    const paddingX = 40;
    const paddingY = 25;
    const chartW = width - paddingX * 2;
    const chartH = height - paddingY * 2;

    // Determine max values for scaling
    let maxLoss = 5;
    let maxGrad = 2;
    scenario.epochs.forEach(e => {
      if (!isNaN(e.loss) && e.loss > maxLoss) maxLoss = e.loss * 1.1;
      if (!isNaN(e.gradNorm) && e.gradNorm > maxGrad) maxGrad = e.gradNorm * 1.1;
    });

    // Helper for coords
    const getX = (idx) => paddingX + (chartW / (totalEpochs - 1 || 1)) * idx;
    const getYLoss = (val) => isNaN(val) ? height - paddingY : height - paddingY - (val / maxLoss) * chartH;
    const getYGrad = (val) => isNaN(val) ? height - paddingY : height - paddingY - (val / maxGrad) * chartH;

    // 1. Draw Loss Curve (Cyan)
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    let started = false;
    visibleData.forEach((d, idx) => {
      if (!isNaN(d.loss)) {
        const x = getX(idx);
        const y = getYLoss(d.loss);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
    });
    ctx.stroke();

    // 2. Draw Gradient Norm Curve (Amber)
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    started = false;
    visibleData.forEach((d, idx) => {
      if (!isNaN(d.gradNorm)) {
        const x = getX(idx);
        const y = getYGrad(d.gradNorm);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
    });
    ctx.stroke();
    ctx.setLineDash([]); // Reset line dash

    // 3. Draw Data Points
    visibleData.forEach((d, idx) => {
      const x = getX(idx);

      // Loss Point
      if (!isNaN(d.loss)) {
        const yLoss = getYLoss(d.loss);
        ctx.fillStyle = d.loss > 10 ? '#ef4444' : '#00f0ff';
        ctx.beginPath();
        ctx.arc(x, yLoss, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#060913';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Anomaly indicator on latest point if fault
      if (idx === currentStep && d.status.startsWith('FAULT')) {
        ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
        ctx.beginPath();
        ctx.arc(x, getYLoss(d.loss || 5), 16, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // Chart Legend in corner
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.fillStyle = '#00f0ff';
    ctx.fillText('— Loss', width - 140, 20);
    ctx.fillStyle = '#f59e0b';
    ctx.fillText('- - Grad Norm', width - 85, 20);
  }

  // Update UI Elements
  function updateUI() {
    const scenario = SCENARIOS[currentScenario];
    const data = scenario.epochs[currentStep] || scenario.epochs[0];

    // Health score
    if (healthScoreEl) {
      healthScoreEl.textContent = `${data.health}%`;
      healthScoreEl.style.color = data.health > 80 ? '#10b981' : data.health > 50 ? '#f59e0b' : '#ef4444';
    }
    if (healthBarEl) {
      healthBarEl.style.width = `${data.health}%`;
      healthBarEl.style.backgroundColor = data.health > 80 ? '#10b981' : data.health > 50 ? '#f59e0b' : '#ef4444';
    }

    // System badge
    if (systemStatusBadge) {
      systemStatusBadge.textContent = data.status;
      systemStatusBadge.className = 'badge ' + (
        data.status.startsWith('FAULT') ? 'badge-error' :
        data.status === 'RECOVERED' ? 'badge-info' :
        data.status === 'WARNING' || data.status === 'STAGNANT' ? 'badge-warn' : 'badge-success'
      );
    }

    // Doctor details
    if (docTitle) docTitle.textContent = scenario.doctor.title;
    if (docLayer) docLayer.textContent = `Target: ${scenario.doctor.layer} (Confidence: ${scenario.doctor.confidence})`;
    if (docRootCause) docRootCause.textContent = scenario.doctor.rootCause;
    if (docPrescription) docPrescription.textContent = scenario.doctor.prescription;
    if (docCode) docCode.textContent = scenario.doctor.code;

    // Terminal log
    renderLog();

    // Chart
    drawChart();
  }

  // Render Log
  function renderLog() {
    if (!terminal) return;
    const scenario = SCENARIOS[currentScenario];
    const lines = [];

    lines.push(`<div class="log-line"><span class="log-time">[BOOT]</span> <span class="log-info">MLBlackBox Flight Recorder v1.0.0 initialized on PyTorch model.</span></div>`);
    lines.push(`<div class="log-line"><span class="log-time">[HOOK]</span> <span class="log-info">Per-layer gradient tracking enabled across 18 parameter tensors.</span></div>`);

    for (let i = 0; i <= currentStep; i++) {
      const e = scenario.epochs[i];
      const timeStr = `[EP ${e.epoch.toString().padStart(2, '0')}]`;
      let styleClass = 'log-info';
      if (e.status.startsWith('FAULT')) styleClass = 'log-error';
      else if (e.status === 'RECOVERED') styleClass = 'log-recovery';
      else if (e.status === 'WARNING' || e.status === 'STAGNANT') styleClass = 'log-warn';
      else if (e.status === 'NOMINAL') styleClass = 'log-success';

      const lossDisp = isNaN(e.loss) ? 'NaN' : e.loss.toFixed(3);
      const gradDisp = isNaN(e.gradNorm) ? 'NaN' : e.gradNorm.toFixed(3);

      lines.push(`<div class="log-line"><span class="log-time">${timeStr}</span> <span class="${styleClass}">Loss: ${lossDisp} | Grad: ${gradDisp} | Health: ${e.health}% — ${e.event}</span></div>`);
    }

    terminal.innerHTML = lines.join('');
    terminal.scrollTop = terminal.scrollHeight;
  }

  // Step Forward
  function step() {
    const scenario = SCENARIOS[currentScenario];
    if (currentStep < scenario.epochs.length - 1) {
      currentStep++;
      updateUI();
    } else {
      pause();
    }
  }

  // Play / Pause
  function togglePlay() {
    if (isPlaying) {
      pause();
    } else {
      play();
    }
  }

  function play() {
    const scenario = SCENARIOS[currentScenario];
    if (currentStep >= scenario.epochs.length - 1) {
      currentStep = 0;
    }
    isPlaying = true;
    if (playBtn) playBtn.innerHTML = `<span>⏸</span> Pause`;
    playInterval = setInterval(() => {
      const sc = SCENARIOS[currentScenario];
      if (currentStep < sc.epochs.length - 1) {
        step();
      } else {
        pause();
      }
    }, 1200);
  }

  function pause() {
    isPlaying = false;
    clearInterval(playInterval);
    if (playBtn) playBtn.innerHTML = `<span>▶</span> Play Run`;
  }

  function reset() {
    pause();
    currentStep = 0;
    updateUI();
  }

  // Select Scenario
  function setScenario(scenarioKey) {
    if (!SCENARIOS[scenarioKey]) return;
    pause();
    currentScenario = scenarioKey;
    currentStep = 0;

    scenarioBtns.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.scenario === scenarioKey);
    });

    updateUI();
  }

  // Copy Snippet
  window.copySnippet = function(elemId, btnElem) {
    const textElem = document.getElementById(elemId);
    if (!textElem) return;
    navigator.clipboard.writeText(textElem.innerText).then(() => {
      const originalText = btnElem.innerHTML;
      btnElem.innerHTML = `<span>✓</span> Copied!`;
      setTimeout(() => {
        btnElem.innerHTML = originalText;
      }, 2000);
    });
  };

  // Setup Event Listeners
  function initListeners() {
    if (playBtn) playBtn.addEventListener('click', togglePlay);
    if (stepBtn) stepBtn.addEventListener('click', step);
    if (resetBtn) resetBtn.addEventListener('click', reset);

    scenarioBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        setScenario(btn.dataset.scenario);
      });
    });

    window.addEventListener('resize', () => {
      setupCanvas();
      drawChart();
    });
  }

  // On DOM Ready
  document.addEventListener('DOMContentLoaded', () => {
    setupCanvas();
    initListeners();
    updateUI();
  });

})();
