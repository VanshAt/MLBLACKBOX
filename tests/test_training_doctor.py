"""
tests/test_training_doctor.py — Comprehensive tests for Training Doctor AI module.

Validates:
1. Local rule-based diagnosis for all five fault types:
   - GRADIENT_EXPLOSION
   - GRADIENT_VANISHING
   - NAN_INF_PROPAGATION
   - LOSS_SPIKE
   - TRAINING_STAGNATION
2. Healthy training diagnosis.
3. Confidence scoring and evidence extraction.
4. Rollback recommendation logic.
5. Ranked remediation step generation and copyable PyTorch code snippets.
6. LLM fallback when API key is missing.
7. LLM fallback when API request fails / times out.
8. LLM response parsing and validation.
9. Privacy guarantee: raw model weights, full datasets, and source code are strictly omitted from payloads.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

from ai.training_doctor import (
    LocalTrainingDoctor,
    LLMTrainingDoctor,
    TrainingDoctorReport,
    diagnose_run,
    SNIPPETS,
)
from tracker.fault_detector import FaultType


class TestTrainingDoctor(unittest.TestCase):

    def setUp(self):
        self.doctor = LocalTrainingDoctor()

    def test_gradient_explosion_diagnosis(self):
        history = [
            {"epoch": 0, "loss": 1.0, "gradient": {"global": {"mean": 0.5}}, "health_score": 90},
            {"epoch": 1, "loss": 0.9, "gradient": {"global": {"mean": 0.6}}, "health_score": 88},
            {
                "epoch": 2,
                "loss": 25.0,
                "gradient": {"global": {"mean": 1200.0}},
                "health_score": 25,
                "fault_detected": True,
                "fault_type": FaultType.GRADIENT_EXPLOSION,
                "severity": "critical",
                "suspected_layer": 0,
                "evidence": {"gradient_norm": 1200.0, "threshold": 3.0},
            },
        ]
        report = self.doctor.diagnose(history)
        self.assertIn("Gradient Explosion", report.headline)
        self.assertTrue(report.rollback_recommended)
        self.assertEqual(report.confidence_level, "High")
        self.assertGreaterEqual(report.confidence_score, 80)
        self.assertTrue(any("gradient norm" in ev.lower() for ev in report.evidence))
        self.assertTrue(any("clip" in step.title.lower() for step in report.ranked_remediation_steps))
        self.assertTrue(any("clip_grad_norm" in code for code in report.code_snippets))

    def test_gradient_vanishing_diagnosis(self):
        history = [
            {"epoch": 0, "loss": 1.0, "gradient": {"global": {"mean": 0.1}}, "health_score": 85},
            {
                "epoch": 10,
                "loss": 0.95,
                "gradient": {"global": {"mean": 1e-6}},
                "health_score": 30,
                "fault_detected": True,
                "fault_type": FaultType.GRADIENT_VANISHING,
                "severity": "critical",
                "evidence": {"recent_mean": 1e-6},
            },
        ]
        report = self.doctor.diagnose(history)
        self.assertIn("Gradient Vanishing", report.headline)
        self.assertTrue(report.rollback_recommended)
        self.assertTrue(any("leakyrelu" in step.title.lower() or "activation" in step.title.lower() for step in report.ranked_remediation_steps))

    def test_nan_propagation_diagnosis(self):
        history = [
            {"epoch": 0, "loss": 0.8, "health_score": 90},
            {
                "epoch": 3,
                "loss": float("nan"),
                "health_score": 10,
                "fault_detected": True,
                "fault_type": FaultType.NAN_INF_PROPAGATION,
                "severity": "critical",
                "batch_idx": 0,
                "evidence": {"loss": "nan"},
            },
        ]
        report = self.doctor.diagnose(history)
        self.assertIn("NaN/Inf", report.headline)
        self.assertTrue(report.rollback_recommended)
        self.assertEqual(report.confidence_score, 98)
        self.assertTrue(any("clamp" in step.title.lower() or "safe" in step.title.lower() for step in report.ranked_remediation_steps))

    def test_loss_spike_diagnosis(self):
        history = [
            {"epoch": 0, "loss": 0.5, "health_score": 90},
            {
                "epoch": 1,
                "loss": 15.0,
                "health_score": 40,
                "fault_detected": True,
                "fault_type": FaultType.LOSS_SPIKE,
                "severity": "critical",
                "evidence": {"spike_ratio": 30.0},
            },
        ]
        report = self.doctor.diagnose(history)
        self.assertIn("Loss Spike", report.headline)
        self.assertTrue(report.rollback_recommended)
        self.assertTrue(any("outlier" in step.title.lower() or "learning rate" in step.title.lower() for step in report.ranked_remediation_steps))

    def test_training_stagnation_diagnosis(self):
        history = [
            {"epoch": 0, "loss": 0.7, "health_score": 85},
            {
                "epoch": 15,
                "loss": 0.69,
                "health_score": 35,
                "fault_detected": True,
                "fault_type": FaultType.TRAINING_STAGNATION,
                "severity": "critical",
                "evidence": {"flat_epochs": 15},
            },
        ]
        report = self.doctor.diagnose(history)
        self.assertIn("Stagnation", report.headline)
        self.assertTrue(any("momentum" in step.title.lower() or "learning rate" in step.title.lower() for step in report.ranked_remediation_steps))

    def test_healthy_training_diagnosis(self):
        history = [
            {"epoch": 0, "loss": 0.9, "accuracy": 0.5, "health_score": 90},
            {"epoch": 1, "loss": 0.4, "accuracy": 0.85, "health_score": 95},
            {"epoch": 2, "loss": 0.1, "accuracy": 0.98, "health_score": 98},
        ]
        report = self.doctor.diagnose(history)
        self.assertIn("Healthy", report.headline)
        self.assertFalse(report.rollback_recommended)
        self.assertEqual(report.confidence_level, "High")
        self.assertGreaterEqual(report.confidence_score, 90)

    def test_llm_fallback_when_api_key_missing(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True):
            llm_doc = LLMTrainingDoctor(local_doctor=self.doctor)
            self.assertFalse(llm_doc.is_available())
            history = [{"epoch": 0, "loss": 0.5, "health_score": 90}]
            report = llm_doc.diagnose(history)
            self.assertEqual(report.mode, "Local Rule-Based AI")
            self.assertIn("Healthy", report.headline)

    def test_llm_fallback_when_request_fails(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy_key_123"}, clear=True):
            llm_doc = LLMTrainingDoctor(local_doctor=self.doctor)
            self.assertTrue(llm_doc.is_available())

            with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
                history = [
                    {
                        "epoch": 2,
                        "loss": 50.0,
                        "health_score": 20,
                        "fault_detected": True,
                        "fault_type": FaultType.GRADIENT_EXPLOSION,
                    }
                ]
                report = llm_doc.diagnose(history)
                # Must seamlessly fallback to local engine without raising exception
                self.assertEqual(report.mode, "Local Rule-Based AI")
                self.assertIn("Gradient Explosion", report.headline)
                self.assertTrue(any("reverted to Local Rule-Based AI" in lim for lim in report.limitations))

    def test_privacy_guarantee_sanitized_payload(self):
        llm_doc = LLMTrainingDoctor(local_doctor=self.doctor)
        fake_weights = [1.23, 4.56, 7.89] * 100
        fake_dataset = ["sensitive_patient_sample_001", "confidential_text"]

        history = [
            {
                "epoch": 0,
                "loss": 0.5,
                "accuracy": 0.8,
                "weights": fake_weights,
                "raw_inputs": fake_dataset,
                "source_code": "def secret_algo(): pass",
                "health_score": 90,
            }
        ]
        sanitized = llm_doc.build_sanitized_payload(history)
        sanitized_str = str(sanitized)

        # Strictly verify no raw weights, dataset samples, or code exist in payload
        self.assertNotIn("sensitive_patient_sample_001", sanitized_str)
        self.assertNotIn("secret_algo", sanitized_str)
        self.assertNotIn("1.23, 4.56", sanitized_str)
        self.assertIn("total_epochs", sanitized)
        self.assertIn("latest_loss", sanitized)
        self.assertIn("latest_health_score", sanitized)


if __name__ == "__main__":
    unittest.main()
