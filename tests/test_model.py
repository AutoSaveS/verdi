"""Forward-shape checks for Stages 1-3."""

import numpy as np
import pytest
import torch

from verdi.model import (
    Stage1,
    Stage2,
    build_prompt,
    compile_context,
    factor_attribution,
    fit_typology,
    make_mask,
    reasoning_cascade,
)
from verdi.model.losses import CrossStageConsistency, StageOneLoss, StageTwoLoss, end_to_end_loss


def test_stage1_forward_shapes():
    model = Stage1(d_v=12, d_e=14)
    v, e = torch.randn(8, 12), torch.randn(8, 14)
    out = model(v, e)
    assert out["z"].shape == (8, 6)
    assert out["mu"].shape == (8, 6) and out["log_var"].shape == (8, 6)
    assert out["v_hat"].shape == v.shape and out["e_hat"].shape == e.shape
    assert out["r_hat"].shape == (8, 1)
    assert torch.all((out["r_hat"] >= 0) & (out["r_hat"] <= 1))


def test_stage1_latent_dict_names_the_six_factors():
    model = Stage1(d_v=12, d_e=14)
    out = model(torch.randn(4, 12), torch.randn(4, 14))
    names = model.latent_dict(out["z"])
    assert list(names) == ["thermal", "water", "wind", "soil", "competition", "residual"]


def test_stage1_loss_terms_are_finite():
    model = Stage1(d_v=12, d_e=14)
    v, e = torch.randn(8, 12), torch.randn(8, 14)
    r_star = torch.rand(8)
    out = model(v, e)
    loss, parts = StageOneLoss()(v, out["v_hat"], e, out["e_hat"], out["mu"],
                                 out["log_var"], out["z"], r_star, out["r_hat"])
    assert torch.isfinite(loss)
    assert {"reconstruction", "kl", "total_correlation", "resilience"} <= set(parts)


def test_stage2_zero_padding_and_variable_length():
    """Token count follows the available modalities; no padding is added."""
    model = Stage2({"m1": 3, "m4": 2}, d_v=12, d_e=14)
    for names in (["m1"], ["m1", "m4"]):
        obs = [(name, torch.randn(5, 3 if name == "m1" else 2), torch.rand(5, 2), torch.rand(5))
               for name in names]
        out = model(obs)
        assert out["tokens"].shape[1] == len(names)
        assert out["v_mu"].shape == (5, 12)
        assert model.sigma(out).shape == (5,)


def test_stage2_requires_declared_text_dimension():
    model = Stage2({"m1": 3}, d_v=12, d_e=14)
    with pytest.raises(ValueError):
        model([("m1", torch.randn(2, 3), torch.rand(2, 2), torch.rand(2))],
              text_embedding=torch.randn(2, 8))


def test_masked_sensor_mask_size():
    mask = make_mask(4, 10, ratio=0.4)
    assert mask.shape == (4, 10)
    assert mask.sum(dim=1).tolist() == [4, 4, 4, 4]


def test_stage2_loss_is_finite():
    model = Stage2({"m1": 3}, d_v=12, d_e=14)
    out = model([("m1", torch.randn(6, 3), torch.rand(6, 2), torch.rand(6))])
    loss, parts = StageTwoLoss()(torch.randn(6, 12), out["v_mu"], out["v_log_var"],
                                 torch.randn(6, 14), out["e_mu"], out["e_log_var"])
    assert torch.isfinite(loss)
    assert torch.isfinite(end_to_end_loss(loss, loss, CrossStageConsistency()(torch.randn(2, 6), torch.randn(2, 6))))


def test_stage3_attribution_and_prompts():
    torch.manual_seed(0)
    stage1 = Stage1(d_v=12, d_e=14)
    v, e = torch.randn(24, 12), torch.randn(24, 14)
    out = stage1(v, e)
    z_base = stage1.z_base(out["z"], out["r_hat"], torch.zeros(24, dtype=torch.long))
    phi = factor_attribution(out["z"], z_base)
    assert phi.shape == out["z"].shape

    typology = fit_typology(phi.detach(), seed=0)
    sigma = torch.rand(24)
    context = compile_context(typology, 0, phi.detach(), out["r_hat"], sigma,
                              ["Platanus"] * 24, v, e)
    assert context.units, "at least one unit should be listed for a cluster"
    assert len(context.units[0]["phi"]) == 6
    prompt = build_prompt(context, 1)
    assert "Evidence:" in prompt and "Task:" in prompt
    cascade = reasoning_cascade(context)
    assert [record["level"] for record in cascade] == ["1", "2", "3"]
    with pytest.raises(ValueError):
        build_prompt(context, 4)
