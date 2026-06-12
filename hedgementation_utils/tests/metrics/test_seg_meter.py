import pytest
import torch
from hedgementation_utils.metrics.seg_meter import SegMeter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_meter():
    return SegMeter(num_classes=2)

@pytest.fixture
def three_class_meter():
    return SegMeter(num_classes=3)

def perfect_predictions(num_classes, size=10):
    """Target and output are identical — perfect predictions."""
    target = torch.randint(0, num_classes, (size,))
    return target.clone(), target

def all_wrong_binary(size=10):
    """All predictions are flipped — worst case for binary."""
    target = torch.zeros(size, dtype=torch.long)
    output = torch.ones(size, dtype=torch.long)
    return output, target

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_hist_is_none_before_update(self, binary_meter):
        assert binary_meter.hist is None

    def test_num_classes_stored(self, binary_meter):
        assert binary_meter.num_classes == 2


# ---------------------------------------------------------------------------
# fast_hist
# ---------------------------------------------------------------------------

class TestFastHist:
    def test_shape(self, binary_meter):
        a = torch.tensor([0, 1, 0, 1])
        b = torch.tensor([0, 1, 1, 0])
        hist = binary_meter.fast_hist(a, b, 2)
        assert hist.shape == (2, 2)

    def test_perfect_predictions_is_diagonal(self, binary_meter):
        a = torch.tensor([0, 1, 0, 1])
        hist = binary_meter.fast_hist(a, a, 2)
        assert torch.all(hist == torch.diag(torch.diag(hist)))

    def test_out_of_range_values_ignored(self, binary_meter):
        # -1 and n should be masked out
        a = torch.tensor([-1, 0, 1, 2])
        b = torch.tensor([-1, 0, 1, 2])
        hist = binary_meter.fast_hist(a, b, 2)
        assert hist.shape == (2, 2)
        assert hist.sum() == 2  # only indices 0 and 1 are valid

    def test_values_are_counts(self, binary_meter):
        a = torch.tensor([0, 0, 1, 1])
        b = torch.tensor([0, 1, 0, 1])
        hist = binary_meter.fast_hist(a, b, 2)
        # hist[i,j] = number of times target=i, pred=j
        assert hist[0, 0] == 1
        assert hist[0, 1] == 1
        assert hist[1, 0] == 1
        assert hist[1, 1] == 1

    def test_multidimensional_input_flattened(self, binary_meter):
        a = torch.zeros(4, 4, dtype=torch.long)
        b = torch.zeros(4, 4, dtype=torch.long)
        hist = binary_meter.fast_hist(a, b, 2)
        assert hist[0, 0] == 16


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_hist_initialised_on_first_update(self, binary_meter):
        output, target = perfect_predictions(2)
        binary_meter.update(output, target)
        assert binary_meter.hist is not None
        assert binary_meter.hist.shape == (2, 2)

    def test_hist_accumulates_across_updates(self, binary_meter):
        output, target = perfect_predictions(2, size=5)
        binary_meter.update(output, target)
        first_sum = binary_meter.hist.sum().item()
        binary_meter.update(output, target)
        assert binary_meter.hist.sum().item() == pytest.approx(first_sum * 2)

    def test_hist_inherits_device(self, binary_meter):
        target = torch.zeros(4, dtype=torch.long)
        output = torch.zeros(4, dtype=torch.long)
        binary_meter.update(output, target)
        assert binary_meter.hist.device == target.device


# ---------------------------------------------------------------------------
# update_batch
# ---------------------------------------------------------------------------

class TestUpdateBatch:
    def test_equivalent_to_sequential_updates(self, binary_meter):
        """update_batch should give the same histogram as calling update() per sample."""
        torch.manual_seed(0)
        output = torch.randint(0, 2, (4, 8, 8))
        target = torch.randint(0, 2, (4, 8, 8))

        batch_meter = SegMeter(num_classes=2)
        batch_meter.update_batch(output, target)

        seq_meter = SegMeter(num_classes=2)
        for i in range(4):
            seq_meter.update(output[i], target[i])

        assert torch.allclose(batch_meter.hist, seq_meter.hist)

    # def test_non_zero_batch_dim(self, binary_meter):
    #     """Should also work when the batch dimension is not 0."""
    #     torch.manual_seed(0)
    #     # shape: (8, 4, 8) — batch along dim 1
    #     output = torch.randint(0, 2, (8, 4, 8))
    #     target = torch.randint(0, 2, (8, 4, 8))

    #     batch_meter = SegMeter(num_classes=2)
    #     batch_meter.update_batch(output, target, batch_dim=1)

    #     seq_meter = SegMeter(num_classes=2)
    #     for i in range(4):
    #         seq_meter.update(output.select(1, i), target.select(1, i))

    #     assert torch.allclose(batch_meter.hist, seq_meter.hist)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

class TestScore:
    def test_perfect_predictions_all_acc_is_one(self, binary_meter):
        output, target = perfect_predictions(2, size=100)
        binary_meter.update(output, target)
        scores = binary_meter.score()
        assert scores["all_acc"].item() == pytest.approx(1.0)

    def test_perfect_predictions_iou_is_one(self, binary_meter):
        output, target = perfect_predictions(2, size=100)
        binary_meter.update(output, target)
        scores = binary_meter.score()
        # NaN where a class has no samples; only check populated classes
        iou = scores["iou"]
        valid = iou[~torch.isnan(iou)]
        assert torch.allclose(valid, torch.ones_like(valid))
        
    def test_all_wrong_binary_acc_is_zero(self, binary_meter):
        output, target = all_wrong_binary(size=20)
        binary_meter.update(output, target)
        scores = binary_meter.score()
        # class 0 recall should be 0 (always predicted as 1)
        assert scores["acc"][0].item() == pytest.approx(0.0)

    def test_score_keys_present(self, binary_meter):
        output, target = perfect_predictions(2)
        binary_meter.update(output, target)
        scores = binary_meter.score()
        assert set(scores.keys()) == {"all_acc", "acc", "pre", "f1", "iou", "freq_iou"}

    def test_freq_iou_is_scalar(self, binary_meter):
        output, target = perfect_predictions(2)
        binary_meter.update(output, target)
        scores = binary_meter.score()
        assert scores["freq_iou"].dim() == 0

    def test_all_acc_is_scalar(self, binary_meter):
        output, target = perfect_predictions(2)
        binary_meter.update(output, target)
        scores = binary_meter.score()
        assert scores["all_acc"].dim() == 0

    def test_acc_and_pre_shape(self, three_class_meter):
        output, target = perfect_predictions(3, size=30)
        three_class_meter.update(output, target)
        scores = three_class_meter.score()
        assert scores["acc"].shape == (3,)
        assert scores["pre"].shape == (3,)

    def test_f1_between_zero_and_one(self, binary_meter):
        torch.manual_seed(0)
        output = torch.randint(0, 2, (50,))
        target = torch.randint(0, 2, (50,))
        binary_meter.update(output, target)
        scores = binary_meter.score()
        f1 = scores["f1"]
        valid = f1[~torch.isnan(f1)]
        assert torch.all(valid >= 0.0) and torch.all(valid <= 1.0)

    def test_accumulation_does_not_change_perfect_score(self, binary_meter):
        """Adding more perfect predictions shouldn't change a perfect score."""
        output, target = perfect_predictions(2, size=20)
        binary_meter.update(output, target)
        binary_meter.update(output, target)
        scores = binary_meter.score()
        assert scores["all_acc"].item() == pytest.approx(1.0)

    def test_freq_iou_between_zero_and_one(self, binary_meter):
        torch.manual_seed(1)
        output = torch.randint(0, 2, (100,))
        target = torch.randint(0, 2, (100,))
        binary_meter.update(output, target)
        scores = binary_meter.score()
        assert 0.0 <= scores["freq_iou"].item() <= 1.0


# ---------------------------------------------------------------------------
# ask_metrics
# ---------------------------------------------------------------------------

class TestAskMetrics:
    def test_returns_only_per_class_metrics(self, binary_meter):
        output, target = perfect_predictions(2, size=20)
        binary_meter.update(output, target)
        retval = binary_meter.ask_metrics(cl=0)
        # all_acc and freq_iou are scalars (dim==0) so should be excluded
        assert "all_acc" not in retval
        assert "freq_iou" not in retval

    def test_returns_float_values(self, binary_meter):
        output, target = perfect_predictions(2, size=20)
        binary_meter.update(output, target)
        retval = binary_meter.ask_metrics(cl=1)
        for v in retval.values():
            assert isinstance(v, float)

    def test_perfect_class_scores(self, binary_meter):
        output, target = perfect_predictions(2, size=50)
        binary_meter.update(output, target)
        retval = binary_meter.ask_metrics(cl=0)
        assert retval["acc"] == pytest.approx(1.0)
        assert retval["iou"] == pytest.approx(1.0)

    def test_valid_class_index(self, three_class_meter):
        output, target = perfect_predictions(3, size=30)
        three_class_meter.update(output, target)
        # should not raise for any valid class index
        for cl in range(3):
            three_class_meter.ask_metrics(cl=cl)


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------

class TestStr:
    def test_str_is_string(self, binary_meter):
        output, target = perfect_predictions(2, size=20)
        binary_meter.update(output, target)
        assert isinstance(str(binary_meter), str)

    def test_str_contains_metric_names(self, binary_meter):
        output, target = perfect_predictions(2, size=20)
        binary_meter.update(output, target)
        result = str(binary_meter)
        for metric in ["all_acc", "acc", "iou", "freq_iou", "f1", "pre"]:
            assert metric in result