import pytest
import torch
from unittest.mock import MagicMock

from hedgementation_utils.metrics.per_hedge_analysis import (
    calc_mean_hedge_accuracy,
    calc_hedge_detection_rate,
    calc_patch_per_hedge_accuracy,
    calc_per_hedge_metrics,
)
from hedgementation_utils.metrics.seg_meter import SegMeter


def make_meter():
    return SegMeter(num_classes=2)


def make_mock_io():
    return MagicMock()


# ---------------------------------------------------------------------------
# calc_mean_hedge_accuracy
# ---------------------------------------------------------------------------

class TestCalcMeanHedgeAccuracy:
    def test_known_mean(self):
        acc = torch.tensor([0.0, 0.5, 1.0])
        assert calc_mean_hedge_accuracy(acc).item() == pytest.approx(0.5)

    def test_all_perfect(self):
        acc = torch.ones(4)
        assert calc_mean_hedge_accuracy(acc).item() == pytest.approx(1.0)

    def test_all_zero(self):
        acc = torch.zeros(4)
        assert calc_mean_hedge_accuracy(acc).item() == pytest.approx(0.0)

    def test_single_element(self):
        acc = torch.tensor([0.75])
        assert calc_mean_hedge_accuracy(acc).item() == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# calc_hedge_detection_rate
# ---------------------------------------------------------------------------

class TestCalcHedgeDetectionRate:
    def test_all_detected(self):
        acc = torch.tensor([0.5, 0.6, 0.8])
        rate = calc_hedge_detection_rate(acc, threshold=0.2)
        assert rate.item() == pytest.approx(1.0)

    def test_none_detected(self):
        acc = torch.tensor([0.1, 0.15, 0.05])
        rate = calc_hedge_detection_rate(acc, threshold=0.2)
        assert rate.item() == pytest.approx(0.0)

    def test_partial_detection(self):
        acc = torch.tensor([0.1, 0.3, 0.5, 0.05])
        rate = calc_hedge_detection_rate(acc, threshold=0.2)
        assert rate.item() == pytest.approx(0.5)

    def test_default_threshold_is_0_2(self):
        acc = torch.tensor([0.19, 0.21])
        rate = calc_hedge_detection_rate(acc)
        assert rate.item() == pytest.approx(0.5)

    def test_threshold_is_exclusive(self):
        # exactly at threshold is NOT counted (strict >)
        acc = torch.tensor([0.2, 0.21])
        rate = calc_hedge_detection_rate(acc, threshold=0.2)
        assert rate.item() == pytest.approx(0.5)

    def test_returns_rate_in_0_1(self):
        acc = torch.rand(10)
        rate = calc_hedge_detection_rate(acc, threshold=0.5)
        assert 0.0 <= rate.item() <= 1.0


# ---------------------------------------------------------------------------
# calc_patch_per_hedge_accuracy
# ---------------------------------------------------------------------------

class TestCalcPatchPerHedgeAccuracy:
    """
    All tests use a flat target_raster of shape (1, 5):
        [-1, 0, 1, -1, -1]
    Background pixels are -1; hedgerow 0 is at index 1; hedgerow 1 at index 2.
    """

    def _raster(self):
        return torch.tensor([[-1, 0, 1, -1, -1]])

    def test_returns_tensor(self):
        result = calc_patch_per_hedge_accuracy(
            predictions=torch.zeros(1, 5, dtype=torch.long),
            target_raster=self._raster(),
            seg_meter=make_meter(),
            io_manager=make_mock_io(),
        )
        assert isinstance(result, torch.Tensor)

    def test_length_equals_number_of_hedgerows(self):
        # max hedge ID = 1, so there should be 2 entries (IDs 0 and 1)
        result = calc_patch_per_hedge_accuracy(
            predictions=torch.zeros(1, 5, dtype=torch.long),
            target_raster=self._raster(),
            seg_meter=make_meter(),
            io_manager=make_mock_io(),
        )
        assert len(result) == 2

    def test_last_hedgerow_not_skipped(self):
        # Raster with hedgerows 0, 1, 2 — verifies off-by-one fix
        target = torch.tensor([[-1, 0, 1, 2, -1]])
        predictions = torch.tensor([[0, 1, 1, 1, 0]])
        result = calc_patch_per_hedge_accuracy(
            predictions=predictions,
            target_raster=target,
            seg_meter=make_meter(),
            io_manager=make_mock_io(),
        )
        assert len(result) == 3

    def test_perfect_predictions_all_ones(self):
        # predictions correctly mark both hedgerow pixels as 1
        predictions = torch.tensor([[0, 1, 1, 0, 0]])
        result = calc_patch_per_hedge_accuracy(
            predictions=predictions,
            target_raster=self._raster(),
            seg_meter=make_meter(),
            io_manager=make_mock_io(),
        )
        assert result[0].item() == pytest.approx(1.0)
        assert result[1].item() == pytest.approx(1.0)

    def test_all_background_predictions_all_zeros(self):
        # predicting all background → every hedgerow missed
        predictions = torch.zeros(1, 5, dtype=torch.long)
        result = calc_patch_per_hedge_accuracy(
            predictions=predictions,
            target_raster=self._raster(),
            seg_meter=make_meter(),
            io_manager=make_mock_io(),
        )
        assert result[0].item() == pytest.approx(0.0)
        assert result[1].item() == pytest.approx(0.0)

    def test_scores_are_independent_per_hedge(self):
        # Only hedgerow 0 is correctly detected; hedgerow 1 is missed.
        # Without state reset between iterations the score for hedge 1 would
        # be inflated by hedge 0's correct predictions.
        predictions = torch.tensor([[0, 1, 0, 0, 0]])
        result = calc_patch_per_hedge_accuracy(
            predictions=predictions,
            target_raster=self._raster(),
            seg_meter=make_meter(),
            io_manager=make_mock_io(),
        )
        assert result[0].item() == pytest.approx(1.0)
        assert result[1].item() == pytest.approx(0.0)

    def test_raises_without_target_or_id(self):
        with pytest.raises(ValueError):
            calc_patch_per_hedge_accuracy(
                predictions=torch.zeros(1, 5, dtype=torch.long),
                seg_meter=make_meter(),
                io_manager=make_mock_io(),
            )

    def test_loads_raster_via_io_manager_when_no_target(self):
        mock_io = make_mock_io()
        mock_io.load_y_id.return_value = self._raster()
        predictions = torch.tensor([[0, 1, 1, 0, 0]])

        result = calc_patch_per_hedge_accuracy(
            predictions=predictions,
            target_id=42,
            seg_meter=make_meter(),
            io_manager=mock_io,
        )

        mock_io.load_y_id.assert_called_once_with(identifier=42)
        assert len(result) == 2

    def test_target_raster_takes_priority_over_target_id(self):
        mock_io = make_mock_io()
        predictions = torch.tensor([[0, 1, 1, 0, 0]])

        calc_patch_per_hedge_accuracy(
            predictions=predictions,
            target_raster=self._raster(),
            target_id=99,
            seg_meter=make_meter(),
            io_manager=mock_io,
        )

        mock_io.load_y_id.assert_not_called()


# ---------------------------------------------------------------------------
# calc_per_hedge_metrics
# ---------------------------------------------------------------------------

class TestCalcPerHedgeMetrics:
    def _raster(self):
        return torch.tensor([[-1, 0, 1, -1]])

    def test_returns_expected_keys(self):
        result = calc_per_hedge_metrics(
            predictions=torch.tensor([[0, 1, 1, 0]]),
            target_raster=self._raster(),
            seg_meter=make_meter(),
            io_manager=make_mock_io(),
        )
        assert "mean_hedge_accuracy" in result
        assert "hedge_detection_rate" in result

    def test_perfect_predictions_mean_accuracy_is_one(self):
        result = calc_per_hedge_metrics(
            predictions=torch.tensor([[0, 1, 1, 0]]),
            target_raster=self._raster(),
            seg_meter=make_meter(),
            io_manager=make_mock_io(),
        )
        assert result["mean_hedge_accuracy"].item() == pytest.approx(1.0)

    def test_no_predictions_mean_accuracy_is_zero(self):
        result = calc_per_hedge_metrics(
            predictions=torch.zeros(1, 4, dtype=torch.long),
            target_raster=self._raster(),
            seg_meter=make_meter(),
            io_manager=make_mock_io(),
        )
        assert result["mean_hedge_accuracy"].item() == pytest.approx(0.0)

    def test_detection_rate_is_fraction(self):
        # Three hedgerows (0, 1, 2); only hedgerow 0 detected
        target = torch.tensor([[-1, 0, 1, 2, -1]])
        predictions = torch.tensor([[0, 1, 0, 0, 0]])
        result = calc_per_hedge_metrics(
            predictions=predictions,
            target_raster=target,
            seg_meter=make_meter(),
            io_manager=make_mock_io(),
            detection_threshold=0.2,
        )
        # 1 of 3 hedgerows detected → rate = 1/3
        assert result["hedge_detection_rate"].item() == pytest.approx(1 / 3)

    def test_detection_rate_in_0_1(self):
        result = calc_per_hedge_metrics(
            predictions=torch.randint(0, 2, (1, 4)),
            target_raster=self._raster(),
            seg_meter=make_meter(),
            io_manager=make_mock_io(),
        )
        rate = result["hedge_detection_rate"].item()
        assert 0.0 <= rate <= 1.0
