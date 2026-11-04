from __future__ import annotations

import unittest
import os
import tempfile

import numpy as np
import torch

from tactile_force_envelope.audit import _finalize_report_warnings
from tactile_force_envelope.data import TactileStream, force_window
from tactile_force_envelope.common import episode_split_four, load_config
from tactile_force_envelope.contact_recalibration import _contact_conditional_alpha
from tactile_force_envelope.model import (
    ForceEnvelopeModel,
    TransformerEnvelopeExpert,
    gaussian_nll,
)
from tactile_force_envelope.smolvla_condition import masked_mean, pool_kv_cache
from tactile_force_envelope.stable_contact import (
    StableContactHysteresis,
    hard_force_limit_exceeded,
    normalized_band_cost,
    normalized_soft_cost,
    stable_contact_from_samples,
)
from tactile_force_envelope.train import _conformal_alpha, _group_coverage
from tactile_force_envelope.transformer_train import (
    _loss as transformer_loss,
    _sustained_contact_loss,
)


class ForceLabelTest(unittest.TestCase):
    def test_real_samples_are_not_interpolated(self) -> None:
        stream = TactileStream(
            name="tip",
            timestamps=np.asarray([0.00, 0.05, 0.10, 0.15, 0.20, 0.25]),
            forces_xyz=np.asarray([[0, 0, value] for value in [0, 1, 2, 10, 4, 3]], dtype=np.float32),
            valid=np.ones(6, dtype=bool),
            feature_names=["force_fx", "force_fy", "force_fz"],
            timestamp_duplicates=0,
            timestamp_rollbacks=0,
        )
        result = force_window(
            [stream], {"tip": np.zeros(3, dtype=np.float32)}, 0.0, 0.3, median_size=1
        )
        self.assertEqual(result["sample_count"], 6)
        self.assertEqual(result["y_max"], 10.0)
        self.assertEqual(result["y_top2_mean"], 7.0)
        self.assertAlmostEqual(result["max_gap_ms"], 50.0)

    def test_top2_is_computed_per_sensor_before_global_max(self) -> None:
        def stream(name: str, values: list[float]) -> TactileStream:
            return TactileStream(
                name=name,
                timestamps=np.asarray([0.0, 0.1]),
                forces_xyz=np.asarray([[0, 0, value] for value in values], dtype=np.float32),
                valid=np.ones(2, dtype=bool),
                feature_names=["force_fx", "force_fy", "force_fz"],
                timestamp_duplicates=0,
                timestamp_rollbacks=0,
            )
        result = force_window(
            [stream("left_tip", [10, 1]), stream("right_tip", [9, 8])],
            {"left_tip": np.zeros(3), "right_tip": np.zeros(3)},
            0.0, 0.2, median_size=1,
        )
        self.assertEqual(result["per_sensor"]["left_tip"]["filtered_top2_mean"], 5.5)
        self.assertEqual(result["per_sensor"]["right_tip"]["filtered_top2_mean"], 8.5)
        self.assertEqual(result["y_filtered_top2_mean"], 8.5)
        self.assertEqual(result["y_raw_max"], 10.0)


class CalibrationTest(unittest.TestCase):
    def test_finite_sample_quantile(self) -> None:
        alpha, rank = _conformal_alpha(np.arange(20, dtype=np.float32), 0.90)
        self.assertEqual(rank, 19)
        self.assertEqual(alpha, 18.0)

    def test_group_coverage(self) -> None:
        rows = [
            {"task": "insert", "tactile": {"sensor_counts": {"left": 6, "right": 6}}},
            {"task": "insert", "tactile": {"sensor_counts": {"left": 4, "right": 4}}},
        ]
        groups = _group_coverage(
            rows, np.asarray([6.0, 1.0]), np.asarray([5.0, 2.0]),
            contact_threshold=5.0, window_s=0.3, sampling_rate_bins=[18.0, 21.0],
        )
        self.assertEqual(groups["by_task"]["insert"]["samples"], 2)
        self.assertEqual(groups["by_contact_state"]["contact"]["coverage"], 0.0)
        self.assertEqual(groups["by_contact_state"]["no_contact"]["coverage"], 1.0)

    def test_four_way_episode_split_is_disjoint_and_complete(self) -> None:
        episode_ids = list(range(20))
        splits = episode_split_four(episode_ids, 0.7, 0.1, 0.1, seed=42)
        self.assertEqual(set(splits), {"train", "validation", "calibration", "test"})
        flattened = [episode for values in splits.values() for episode in values]
        self.assertEqual(sorted(flattened), episode_ids)
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_config_expands_environment_variables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "config.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("data:\n  root: ${TACTILE_TEST_ROOT}/dataset\n")
            previous = os.environ.get("TACTILE_TEST_ROOT")
            os.environ["TACTILE_TEST_ROOT"] = directory
            try:
                config = load_config(path)
            finally:
                if previous is None:
                    os.environ.pop("TACTILE_TEST_ROOT", None)
                else:
                    os.environ["TACTILE_TEST_ROOT"] = previous
            self.assertEqual(
                os.path.normpath(config["data"]["root"]),
                os.path.join(directory, "dataset"),
            )

    def test_contact_conditional_alpha_uses_only_requested_side_contacts(self) -> None:
        values = {
            "target": np.asarray([100.0, 2.0, 9.0, 12.0], dtype=np.float32),
            "mu": np.asarray([0.0, 1.0, 3.0, 4.0], dtype=np.float32),
            "sigma": np.ones(4, dtype=np.float32),
            "raw_max": np.asarray([0.0, 1.0, 6.0, 8.0], dtype=np.float32),
        }
        alpha, rank, contact = _contact_conditional_alpha(values, 5.0, 0.5)
        self.assertEqual(contact.tolist(), [False, False, True, True])
        self.assertEqual(rank, 2)
        self.assertAlmostEqual(alpha, 8.0, places=5)


class AuditWarningTest(unittest.TestCase):
    def test_camera_rollbacks_are_reported_once_per_camera(self) -> None:
        report = {
            "warnings": [],
            "timestamp_range": {"start": 0.0, "end": 1.0},
            "cameras": {
                "camera_a": {"obvious_delay_count": 0, "timestamp_rollbacks": 2},
                "camera_b": {"obvious_delay_count": 0, "timestamp_rollbacks": 3},
            },
            "tactile": {},
        }
        _finalize_report_warnings(report, has_action_chunk=True, robot_timestamp_rollbacks=0)
        self.assertEqual(report["warnings"], [
            "camera_a: timestamps rolled back 2 times.",
            "camera_b: timestamps rolled back 3 times.",
        ])

    def test_no_cameras_does_not_raise(self) -> None:
        report = {
            "warnings": [],
            "timestamp_range": {"start": None, "end": None},
            "cameras": {},
            "tactile": {},
        }
        _finalize_report_warnings(report, has_action_chunk=True, robot_timestamp_rollbacks=0)
        self.assertEqual(report["warnings"], [])


class ModelTest(unittest.TestCase):
    def test_output_and_nll(self) -> None:
        config = {
            "branch_dim": 16, "hidden_dim": 32, "task_embedding_dim": 8,
            "dropout": 0.0, "log_var_min": -8.0, "log_var_max": 4.0,
        }
        model = ForceEnvelopeModel(12, 8, 8, 9, 2, config)
        mu, log_var = model(
            torch.zeros(4, 12), torch.zeros(4, 8), torch.zeros(4, 9, 8),
            torch.ones(4, 9), torch.zeros(4, dtype=torch.long),
        )
        self.assertEqual(tuple(mu.shape), (4,))
        self.assertTrue(torch.isfinite(gaussian_nll(mu, log_var, torch.ones(4))))

    def test_smolvla_feature_can_disable_duplicate_task_branch(self) -> None:
        config = {
            "branch_dim": 16, "hidden_dim": 32, "task_embedding_dim": 8,
            "dropout": 0.0, "log_var_min": -8.0, "log_var_max": 4.0,
            "use_extra_robot_state": True, "use_extra_task_embedding": False,
        }
        model = ForceEnvelopeModel(960, 8, 8, 9, 1, config)
        mu, log_var = model(
            torch.zeros(2, 960), torch.zeros(2, 8), torch.zeros(2, 9, 8),
            torch.ones(2, 9), torch.zeros(2, dtype=torch.long),
        )
        self.assertEqual(tuple(mu.shape), (2,))
        self.assertEqual(tuple(log_var.shape), (2,))

    def test_masked_mean_excludes_padding(self) -> None:
        tokens = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
        mask = torch.tensor([[True, True, False]])
        torch.testing.assert_close(masked_mean(tokens, mask), torch.tensor([[2.0, 4.0]]))

    def test_kv_pooling_excludes_padding_and_preserves_layer_order(self) -> None:
        key = torch.tensor([[[[1.0, 3.0]], [[3.0, 5.0]], [[100.0, 100.0]]]])
        value = key + 10.0
        cache = {0: {"key_states": key, "value_states": value}}
        feature = pool_kv_cache(cache, torch.tensor([[True, True, False]]))
        torch.testing.assert_close(feature, torch.tensor([[2.0, 4.0, 12.0, 14.0]]))

    def test_transformer_envelope_shapes_and_masks(self) -> None:
        config = {
            "d_model": 32, "num_heads": 4, "ffn_dim": 64, "num_layers": 2,
            "dropout": 0.0, "log_var_min": -8.0, "log_var_max": 4.0,
        }
        model = TransformerEnvelopeExpert(16, 8, 8, 9, config)
        output = model(
            torch.zeros(3, 11, 16),
            torch.tensor([[True] * 9 + [False] * 2] * 3),
            torch.zeros(3, 8),
            torch.zeros(3, 9, 8),
            torch.tensor([[True] * 8 + [False]] * 3),
        )
        self.assertEqual(set(output), {
            "mu_left", "log_var_left", "mu_right", "log_var_right",
            "contact_logit_left", "contact_logit_right",
        })
        self.assertTrue(all(tuple(value.shape) == (3,) for value in output.values()))

    def test_transformer_accepts_masked_causal_tactile_history_tokens(self) -> None:
        config = {
            "d_model": 32, "num_heads": 4, "ffn_dim": 64, "num_layers": 2,
            "dropout": 0.0, "log_var_min": -8.0, "log_var_max": 4.0,
        }
        model = TransformerEnvelopeExpert(
            16, 8, 8, 9, config, history_dim=5, history_steps=6
        )
        history = torch.zeros(3, 2, 6, 5)
        history_mask = torch.tensor([[[False, True, True, True, True, True]] * 2] * 3)
        shapes = model.token_shapes(
            torch.zeros(3, 11, 16), torch.ones(3, 11, dtype=torch.bool),
            torch.zeros(3, 8), torch.zeros(3, 9, 8),
            torch.ones(3, 9, dtype=torch.bool), history, history_mask,
        )
        output = model(
            torch.zeros(3, 11, 16), torch.ones(3, 11, dtype=torch.bool),
            torch.zeros(3, 8), torch.zeros(3, 9, 8),
            torch.ones(3, 9, dtype=torch.bool), history, history_mask,
        )
        self.assertEqual(shapes["tactile_history_raw"], [3, 2, 6, 5])
        self.assertEqual(shapes["tactile_history_tokens"], [3, 12, 32])
        self.assertTrue(all(tuple(value.shape) == (3,) for value in output.values()))

    def test_transformer_can_remove_action_branch_instead_of_zeroing_actions(self) -> None:
        config = {
            "d_model": 32, "num_heads": 4, "ffn_dim": 64, "num_layers": 2,
            "dropout": 0.0, "log_var_min": -8.0, "log_var_max": 4.0,
            "use_action_conditioning": False,
        }
        model = TransformerEnvelopeExpert(16, 8, 8, 9, config)
        self.assertIsNone(model.action_projection)
        condition = torch.zeros(2, 11, 16)
        condition_mask = torch.ones(2, 11, dtype=torch.bool)
        state = torch.zeros(2, 8)
        action_mask = torch.ones(2, 9, dtype=torch.bool)
        output_a = model(
            condition, condition_mask, state, torch.zeros(2, 9, 8), action_mask
        )
        output_b = model(
            condition, condition_mask, state, torch.randn(2, 9, 8), action_mask
        )
        for key in output_a:
            torch.testing.assert_close(output_a[key], output_b[key])
        shapes = model.token_shapes(
            condition, condition_mask, state, torch.zeros(2, 9, 8), action_mask
        )
        self.assertFalse(shapes["action_conditioning_enabled"])
        self.assertEqual(shapes["action_tokens"], [])
        self.assertEqual(shapes["decoder_input"], [2, 3, 32])

    def test_contact_weight_applies_to_whole_window(self) -> None:
        outputs = {
            "mu_left": torch.zeros(2), "log_var_left": torch.zeros(2),
            "mu_right": torch.zeros(2), "log_var_right": torch.zeros(2),
        }
        batch = {
            "target_left": torch.tensor([2.0, 0.0]),
            "target_right": torch.tensor([2.0, 0.0]),
            # The first window is contact because only its left side is contact.
            "window_contact": torch.tensor([1.0, 0.0]),
        }
        total, nll, smooth = transformer_loss(
            outputs, batch, torch.device("cpu"), contact_sample_weight=2.0,
            smooth_l1_weight=0.5,
        )
        self.assertAlmostEqual(float(nll), 4.0 / 3.0, places=6)
        self.assertAlmostEqual(float(smooth), 1.0, places=6)
        self.assertAlmostEqual(float(total), 11.0 / 6.0, places=6)

    def test_sustained_loss_masks_left_and_right_independently(self) -> None:
        outputs = {
            "mu_left": torch.zeros(2), "log_var_left": torch.zeros(2),
            "mu_right": torch.zeros(2), "log_var_right": torch.zeros(2),
        }
        batch = {
            "target_left": torch.tensor([2.0, 100.0]),
            "target_right": torch.tensor([100.0, 4.0]),
            "sustained_mask_left": torch.tensor([True, False]),
            "sustained_mask_right": torch.tensor([False, True]),
        }
        total, nll, smooth = _sustained_contact_loss(
            outputs, batch, torch.device("cpu"), smooth_l1_weight=0.5
        )
        self.assertAlmostEqual(float(nll), 5.0, places=6)
        self.assertAlmostEqual(float(smooth), 2.5, places=6)
        self.assertAlmostEqual(float(total), 6.25, places=6)

    def test_sustained_smooth_l1_ablation_disables_gaussian_term(self) -> None:
        outputs = {
            "mu_left": torch.zeros(1), "log_var_left": torch.tensor([4.0]),
            "mu_right": torch.zeros(1), "log_var_right": torch.tensor([-4.0]),
        }
        batch = {
            "target_left": torch.tensor([2.0]),
            "target_right": torch.tensor([4.0]),
            "sustained_mask_left": torch.tensor([True]),
            "sustained_mask_right": torch.tensor([True]),
        }
        total, nll, smooth = _sustained_contact_loss(
            outputs, batch, torch.device("cpu"), smooth_l1_weight=1.0,
            gaussian_nll_weight=0.0,
        )
        self.assertGreater(float(nll), 0.0)
        self.assertAlmostEqual(float(smooth), 2.5, places=6)
        self.assertAlmostEqual(float(total), 2.5, places=6)


class StableContactGateTest(unittest.TestCase):
    def test_entry_requires_three_of_last_four_valid_samples(self) -> None:
        self.assertTrue(stable_contact_from_samples(
            [1.0, 6.0, 7.0, 8.0], [True] * 4, tau_on=5.0
        ))
        self.assertFalse(stable_contact_from_samples(
            [6.0, 7.0, 1.0], [True] * 3, tau_on=5.0
        ))

    def test_hysteresis_enters_holds_and_exits_without_future_samples(self) -> None:
        gate = StableContactHysteresis(tau_on=5.0, tau_off=3.0)
        for value in (6.0, 7.0, 1.0):
            self.assertFalse(gate.update("left", value))
        self.assertTrue(gate.update("left", 8.0))
        self.assertTrue(gate.update("left", 2.0))
        self.assertFalse(gate.update("left", 2.0))

    def test_soft_cost_is_gated_normalized_and_clipped(self) -> None:
        self.assertEqual(normalized_soft_cost(20.0, 10.0, False, force_scale=5.0, cost_clip=2.0), 0.0)
        self.assertEqual(normalized_soft_cost(12.5, 10.0, True, force_scale=5.0, cost_clip=2.0), 0.25)
        self.assertEqual(normalized_soft_cost(15.0, 10.0, True, force_scale=5.0, cost_clip=2.0), 1.0)
        self.assertEqual(normalized_soft_cost(20.0, 10.0, True, force_scale=5.0, cost_clip=2.0), 4.0)
        self.assertEqual(normalized_soft_cost(30.0, 10.0, True, force_scale=5.0, cost_clip=2.0), 4.0)

    def test_two_sided_band_cost_keeps_low_and_high_separate(self) -> None:
        self.assertEqual(
            normalized_band_cost(
                5.0, 7.0, 12.0, True, force_scale=2.0, cost_clip=2.0
            ),
            (1.0, 0.0),
        )
        self.assertEqual(
            normalized_band_cost(
                14.0, 7.0, 12.0, True, force_scale=2.0, cost_clip=2.0
            ),
            (0.0, 1.0),
        )
        self.assertEqual(
            normalized_band_cost(
                9.0, 7.0, 12.0, True, force_scale=2.0, cost_clip=2.0
            ),
            (0.0, 0.0),
        )
        self.assertEqual(
            normalized_band_cost(
                5.0, 7.0, 12.0, False, force_scale=2.0, cost_clip=2.0
            ),
            (0.0, 0.0),
        )
        self.assertFalse(hard_force_limit_exceeded(39.0, 40.0))
        self.assertTrue(hard_force_limit_exceeded(41.0, 40.0))


if __name__ == "__main__":
    unittest.main()
