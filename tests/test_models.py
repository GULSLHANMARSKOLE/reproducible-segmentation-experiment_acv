import pytest
import torch

from src.models import build_model
from src.models.baseline import BOTTLENECK_DILATIONS as BASE_D
from src.models.dilated import BOTTLENECK_DILATIONS as DIL_D
from src.utils import load_config
from src.metrics import count_trainable_parameters

CFG = load_config()


@pytest.mark.parametrize("name", ["baseline", "dilated"])
def test_output_shape_is_21_channel_full_resolution_logits(name):
    model = build_model(name, CFG).eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 21, 256, 256)
    assert not torch.allclose(y.softmax(1).sum(1), y.sum(1))  # raw logits, not probabilities


def test_parameter_counts_identical_and_structure_identical():
    base, dil = build_model("baseline", CFG), build_model("dilated", CFG)
    assert count_trainable_parameters(base) == count_trainable_parameters(dil)
    sb, sd = base.state_dict(), dil.state_dict()
    assert list(sb) == list(sd) and all(sb[k].shape == sd[k].shape for k in sb)


def test_exactly_one_architectural_difference():
    base, dil = build_model("baseline", CFG), build_model("dilated", CFG)
    mb, md = dict(base.named_modules()), dict(dil.named_modules())
    assert list(mb) == list(md)
    diffs = []
    for name in mb:
        if name == "":  # root wrappers are BaselineUNet / DilatedUNet by design; all submodules must match
            continue
        assert type(mb[name]) is type(md[name])
        if isinstance(mb[name], torch.nn.Conv2d):
            cb, cd = mb[name], md[name]
            assert (cb.in_channels, cb.out_channels, cb.kernel_size, cb.stride) == (cd.in_channels, cd.out_channels, cd.kernel_size, cd.stride)
            if (cb.dilation, cb.padding) != (cd.dilation, cd.padding):
                diffs.append((name, cb.dilation, cd.dilation, cb.padding, cd.padding))
    assert diffs == [("bottleneck.conv2.conv", (1, 1), (2, 2), (1, 1), (2, 2))]
    assert BASE_D == (1, 1) and DIL_D == (1, 2)


def test_same_seed_gives_identical_initial_weights():
    torch.manual_seed(42)
    a = build_model("baseline", CFG)
    torch.manual_seed(42)
    b = build_model("dilated", CFG)
    assert all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))


def test_dilation_changes_function_not_weights():
    torch.manual_seed(0)
    a = build_model("baseline", CFG).eval()
    b = build_model("dilated", CFG).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert not torch.allclose(a(x), b(x))


def test_invalid_input_size_and_unknown_model():
    with pytest.raises(ValueError, match="divisible by 8"):
        build_model("baseline", CFG)(torch.randn(1, 3, 100, 100))
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("resnet", CFG)


def test_no_pretrained_allowed():
    cfg = {**CFG, "model": {**CFG["model"], "pretrained": True}}
    with pytest.raises(ValueError):
        build_model("baseline", cfg)
