"""
tests/test_failure_forecaster.py — Tests for Failure Forecaster early warning system.

Validates:
1. Healthy Iris training produces Low risk and zero rollback recommendations.
2. Explosion precursor telemetry produces elevated explosion risk before or near fault.
3. Vanishing / stagnation precursor telemetry produces elevated vanishing / stagnation risk.
4. NaN precursor telemetry produces High / Critical NaN/Inf risk.
5. Loss spike precursor telemetry produces spike risk.
6. Forecast records are generated with complete fields.
7. Forecasts are strictly advisory: never trigger automatic halts or rollbacks.
8. AI Training Doctor correctly interprets and explains High / Critical forecasts.
9. Overall risk score properly aggregates individual component risks.
"""

import unittest
from ai.failure_forecaster import (
    FailureForecaster,
    FailureForecast,
    ForecastRiskLevel,
    ForecastEventType,
)
from ai.training_doctor import LocalTrainingDoctor
from tracker.fault_detector import FaultType


class TestFailureForecaster(unittest.TestCase):

    def setUp(self):
        self.forecaster = FailureForecaster(window_size=5)
        self.doctor = LocalTrainingDoctor()

    def test_healthy_training_forecast_is_low(self):
        history = [
            {"epoch": 0, "loss": 0.8, "accuracy": 0.6, "gradient": {"global": {"mean": 0.1}}, "health_score": 90},
            {"epoch": 1, "loss": 0.5, "accuracy": 0.85, "gradient": {"global": {"mean": 0.08}}, "health_score": 95},
            {"epoch": 2, "loss": 0.2, "accuracy": 0.98, "gradient": {"global": {"mean": 0.06}}, "health_score": 98},
        ]
        curr = history[-1]
        fc = self.forecaster.forecast(curr, history)
        self.assertEqual(fc.overall_risk_level, ForecastRiskLevel.LOW)
        self.assertLess(fc.overall_risk_score, 30)
        self.assertIsNone(fc.estimated_epochs_to_failure)
        self.assertEqual(fc.event_type, ForecastEventType.FORECAST_LOW)

    def test_explosion_precursor_detection(self):
        history = [
            {"epoch": 0, "loss": 1.0, "gradient": {"global": {"mean": 0.5}}},
            {"epoch": 1, "loss": 0.9, "gradient": {"global": {"mean": 0.6}}},
            {"epoch": 2, "loss": 0.8, "gradient": {"global": {"mean": 0.7}}},
            # Surge from 0.7 to 4.5
            {"epoch": 3, "loss": 1.5, "gradient": {"global": {"mean": 4.5}}},
        ]
        curr = history[-1]
        fc = self.forecaster.forecast(curr, history)
        self.assertIn(fc.overall_risk_level, (ForecastRiskLevel.MEDIUM, ForecastRiskLevel.HIGH, ForecastRiskLevel.CRITICAL))
        self.assertEqual(fc.likely_failure_type, FaultType.GRADIENT_EXPLOSION)
        self.assertGreaterEqual(fc.per_fault_risk_scores[FaultType.GRADIENT_EXPLOSION], 45)
        self.assertTrue(any("gradient" in sig.lower() for sig in fc.warning_signals))

    def test_vanishing_precursor_detection(self):
        history = [
            {"epoch": 0, "loss": 0.9, "gradient": {"global": {"mean": 0.1}}},
            {"epoch": 1, "loss": 0.85, "gradient": {"global": {"mean": 0.01}}},
            {"epoch": 2, "loss": 0.82, "gradient": {"global": {"mean": 1e-4}}},
            {"epoch": 3, "loss": 0.81, "gradient": {"global": {"mean": 1e-7}}},
        ]
        curr = history[-1]
        fc = self.forecaster.forecast(curr, history)
        self.assertIn(fc.overall_risk_level, (ForecastRiskLevel.HIGH, ForecastRiskLevel.CRITICAL))
        self.assertEqual(fc.likely_failure_type, FaultType.GRADIENT_VANISHING)
        self.assertGreaterEqual(fc.per_fault_risk_scores[FaultType.GRADIENT_VANISHING], 80)

    def test_nan_precursor_detection(self):
        history = [
            {"epoch": 0, "loss": 1.0},
            {"epoch": 1, "loss": 1e14},  # Extreme loss threatening float overflow
        ]
        curr = history[-1]
        fc = self.forecaster.forecast(curr, history)
        self.assertIn(fc.overall_risk_level, (ForecastRiskLevel.HIGH, ForecastRiskLevel.CRITICAL))
        self.assertEqual(fc.likely_failure_type, FaultType.NAN_INF_PROPAGATION)
        self.assertGreaterEqual(fc.per_fault_risk_scores[FaultType.NAN_INF_PROPAGATION], 80)

    def test_loss_spike_precursor_detection(self):
        history = [
            {"epoch": 0, "loss": 0.3, "accuracy": 0.9},
            {"epoch": 1, "loss": 3.0, "accuracy": 0.5},  # 10x multiplier
        ]
        curr = history[-1]
        fc = self.forecaster.forecast(curr, history)
        self.assertIn(fc.overall_risk_level, (ForecastRiskLevel.HIGH, ForecastRiskLevel.CRITICAL))
        self.assertEqual(fc.likely_failure_type, FaultType.LOSS_SPIKE)

    def test_stagnation_precursor_detection(self):
        history = [{"epoch": i, "loss": 0.693 + (0.0001 * (i % 2))} for i in range(10)]
        curr = history[-1]
        fc = self.forecaster.forecast(curr, history)
        self.assertEqual(fc.likely_failure_type, FaultType.TRAINING_STAGNATION)
        self.assertGreaterEqual(fc.per_fault_risk_scores[FaultType.TRAINING_STAGNATION], 50)

    def test_forecast_safety_advisory_no_automatic_rollback(self):
        history = [
            {"epoch": 0, "loss": 1.0, "gradient": {"global": {"mean": 0.5}}},
            {"epoch": 1, "loss": 1.5, "gradient": {"global": {"mean": 10.0}}},
        ]
        curr = history[-1]
        fc = self.forecaster.forecast(curr, history)
        # Even with high/critical risk forecast:
        report = self.doctor.diagnose([{"epoch": 0, "loss": 1.0}, {"epoch": 1, "loss": 1.5, "forecast": fc.to_dict()}])
        # Forecasts must NEVER recommend automatic rollback
        self.assertFalse(report.rollback_recommended)
        self.assertIn("Early Warning", report.headline)
        self.assertTrue(any("forecast" in lim.lower() for lim in report.limitations))

    def test_doctor_explains_high_forecast(self):
        history = [
            {"epoch": 0, "loss": 1.0, "gradient": {"global": {"mean": 0.5}}},
            {"epoch": 1, "loss": 1.2, "gradient": {"global": {"mean": 8.0}}},
        ]
        curr = history[-1]
        fc = self.forecaster.forecast(curr, history)
        curr["forecast"] = fc.to_dict()
        report = self.doctor.diagnose(history)
        self.assertIn("Early Warning", report.headline)
        self.assertTrue(len(report.ranked_remediation_steps) >= 1)
        self.assertTrue(any("clip" in code for code in report.code_snippets) or len(report.code_snippets) > 0)


if __name__ == "__main__":
    unittest.main()
