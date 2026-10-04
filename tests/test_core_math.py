import math

import torch
from torch import nn
from torch.nn import functional as F

from ultralytics.nn.modules import aeroguard as core
from ultralytics.utils import sage

torch.set_num_threads(2)

def finite_grads(module, inputs):
    for p in module.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    for p in inputs:
        assert p.grad is not None and torch.isfinite(p.grad).all()

def test_initialization():
    modules = [core.SharedDetailExtractor(8), core.RDET(16), core.BDRP([8, 32, 16]), core.SJRHCore(2, [16, 32, 64])]
    for module in modules:
        assert not any((isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in module.modules()))
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                assert m.bias is not None and torch.count_nonzero(m.bias) == 0
    s = modules[3].levels[0]
    assert s.cls_sampler.predictor[0].out_channels == 24
    assert s.quality_pred[0].in_channels == 192 and s.quality_pred[0].out_channels == 32
    assert not isinstance(s.cls_refine[-1], nn.SiLU)

def test_constant():
    m = core.SharedDetailExtractor(4, channels=4)
    with torch.no_grad():
        m.shallow_proj.weight.copy_(torch.eye(4).view(4, 4, 1, 1))
    x = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 4, 1, 1).expand(2, 4, 9, 13)
    (e, low), d = m.forward_with_diagnostics(x)
    assert torch.equal(low, x) and torch.count_nonzero(e) == 0
    assert torch.allclose(F.softplus(m.tau_raw), torch.full_like(m.tau_raw, 0.8))

def test_rdet():
    shared = core.SharedDetailExtractor(8)
    receivers = nn.ModuleList([core.RDET(16), core.RDET(24), core.RDET(32)])
    p2 = torch.randn(2, 8, 16, 24, requires_grad=True)
    xs = [torch.randn(2, c, h, w, requires_grad=True) for c, h, w in [(16, 8, 12), (24, 4, 6), (32, 2, 3)]]
    pair = shared(p2)
    out = []
    for r, x in zip(receivers, xs):
        y, d = r.forward_with_diagnostics([pair, x])
        assert torch.equal(d['gate'], torch.full_like(d['gate'], 0.5))
        assert torch.allclose(d['alpha'], torch.tensor(0.1))
        assert y.shape == x.shape
        assert torch.allclose(y - x, d['alpha'] * d['gate'] * d['transformed'], atol=2e-07)
        out.append(y)
    sum((y.square().mean() for y in out)).backward()
    finite_grads(shared, [p2])
    finite_grads(receivers, xs)
    assert len([m for m in shared.modules() if isinstance(m, core.FixedBinomialBlur)]) == 1

def test_zero_rdet():
    s = core.SharedDetailExtractor(4)
    r = core.RDET(8)
    p = torch.zeros(1, 4, 8, 12, requires_grad=True)
    x = torch.zeros(1, 8, 4, 6, requires_grad=True)
    y = r([s(p), x])
    assert torch.count_nonzero(y) == 0
    y.sum().backward()
    finite_grads(s, [p])
    finite_grads(r, [x])

def test_bdrp():
    m = core.BDRP([8, 32, 16])
    xs = [torch.randn(2, c, h, w, requires_grad=True) for c, h, w in [(8, 8, 12), (32, 2, 3), (16, 4, 6)]]
    y, d = m.forward_with_diagnostics(xs)
    bound = 0.35 * d['reference_rms']
    assert y.shape == (2, 64, 4, 6)
    assert torch.equal(d['gate'], torch.full_like(d['gate'], 0.5))
    assert torch.all(d['bounded_delta'].abs() <= bound + 1e-07)
    assert torch.all(d['increment'].abs() <= bound + 1e-07)
    assert torch.all(d['increment'].square().mean(1, keepdim=True).sqrt() <= bound + 1e-07)
    assert torch.allclose(y, d['base'] + d['increment'])
    y.square().mean().backward()
    finite_grads(m, xs)

def test_sampler():
    for shape in [(7, 11), (1, 5), (5, 1), (1, 1)]:
        s = core.TaskBoundedSampler(3)
        x = torch.randn(1, 3, *shape, requires_grad=True)
        y, d = s.forward_with_diagnostics(x)
        assert torch.allclose(x, y, atol=1e-06) and torch.count_nonzero(d['offset_cells']) == 0
        assert torch.equal(d['radius_cells'], torch.full_like(d['radius_cells'], 0.75))
        y.sum().backward()
        finite_grads(s, [x])
    s = core.TaskBoundedSampler(1)
    yy, xx = torch.meshgrid(torch.arange(7), torch.arange(11), indexing='ij')
    x = (xx + 10 * yy).float()[None, None]
    with torch.no_grad():
        s.predictor[-1].bias.copy_(torch.tensor([math.atanh(0.5 / 0.75), math.atanh(-0.25 / 0.75), 0.0]))
    y, d = s.forward_with_diagnostics(x)
    assert torch.allclose(y[:, :, 1:-1, 1:-1], x[:, :, 1:-1, 1:-1] - 2, atol=1e-05)
    with torch.no_grad():
        s.predictor[-1].bias.fill_(100)
    y, d = s.forward_with_diagnostics(x)
    assert d['offset_cells'].abs().max() <= 1.5
    assert torch.allclose(d['offset_cells'].square().sum(1).sqrt(), torch.full((1, 7, 11), math.sqrt(2) * 1.5))

def test_sjrh():
    m = core.SJRHCore(2, [16, 32, 64])
    xs = [torch.randn(2, c, h, w, requires_grad=True) for c, h, w in [(16, 8, 12), (32, 4, 6), (64, 2, 3)]]
    ys = m(xs)
    for y, x in zip(ys, xs):
        assert [z.shape[1] for z in y] == [64, 2, 1] and all((z.shape[-2:] == x.shape[-2:] for z in y))
    sum((z.square().mean() for y in ys for z in y)).backward()
    finite_grads(m, xs)
    assert m.levels[0].cls_sampler is not m.levels[0].reg_sampler

def test_sage_closed_form():
    gt = torch.tensor([[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 0.01, 0.01]])
    m = sage.SAGEGeometry()
    g = m(gt.clone().requires_grad_(), gt, torch.tensor([8.0, 8.0]))
    assert torch.equal(g['loss'], torch.zeros(2)) and torch.equal(g['quality'], torch.ones(2))
    p = torch.tensor([[2.0, 0.0, 10.0, 8.0]], requires_grad=True)
    g = m(p, gt[:1], torch.tensor([8.0]))
    a = 1 / (1 + math.exp(-((math.sqrt(64 + 1e-07) / (8 + 1e-07) - 3) / 0.75)))
    robust = 0.5 * (math.sqrt((2 / (math.sqrt(64 + 1e-07) + 8)) ** 2 + 0.05 ** 2) - 0.05)
    expected = a * 0.4 + (1 - a) * robust
    q = math.exp(-(a * -math.log(0.6) + (1 - a) * robust))
    assert abs(g['loss'].item() - expected) < 1e-07 and abs(g['quality'].item() - q) < 1e-07
    assert not g['quality'].requires_grad
    g['loss'].sum().backward()
    assert torch.isfinite(p.grad).all()

def test_sage_no_overlap():
    p = torch.tensor([[20.0, 0.0, 28.0, 8.0]], requires_grad=True)
    gt = torch.tensor([[0.0, 0.0, 8.0, 8.0]])
    g = sage.sage_geometry(p, gt, torch.tensor([8.0]))
    g['loss'].sum().backward()
    assert p.grad.abs().sum() > 0 and torch.isfinite(p.grad).all()
    repeated = gt.repeat(3, 1)
    g = sage.sage_geometry(repeated, repeated, torch.tensor([4.0, 8.0, 16.0]))
    assert torch.all(g['alpha_iou'][:-1] > g['alpha_iou'][1:])

def test_quality_reduction():
    p = torch.tensor([[2.0, 0.0, 10.0, 8.0], [1.0, 1.0, 9.0, 9.0]], requires_grad=True)
    gt = torch.tensor([[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 8.0, 8.0]])
    g = sage.sage_geometry(p, gt, torch.tensor([8.0, 8.0]))
    z = torch.tensor([0.2, -0.3], requires_grad=True)
    neg = torch.tensor([-0.4, 0.1, 0.0], requires_grad=True)
    w = torch.tensor([0.4, 0.2], requires_grad=True)
    losses = sage.weighted_sage_losses(g, z, neg, w)
    expected_pos = (w.detach() * F.binary_cross_entropy_with_logits(z, g['quality'], reduction='none')).sum()
    expected_neg = F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
    assert torch.allclose(losses['quality'], expected_pos + 0.05 * expected_neg)
    (7.5 * 0.5 * losses['quality']).backward()
    assert p.grad is None and w.grad is None
    assert torch.allclose(z.grad, 3.75 * w.detach() * (z.detach().sigmoid() - g['quality']), atol=1e-07)

def test_empty():
    p = torch.empty(0, 4, requires_grad=True)
    g = sage.sage_geometry(p, p.detach(), torch.empty(0))
    z = torch.empty(0, requires_grad=True)
    neg = torch.tensor([0.0, 1.0], requires_grad=True)
    losses = sage.weighted_sage_losses(g, z, neg, torch.empty(0))
    assert losses['box'] == 0 and losses['quality_positive'] == 0 and torch.isfinite(losses['quality'])
    losses['quality'].backward()
    assert torch.isfinite(neg.grad).all()
    all_empty = sage.weighted_sage_losses(g, z, torch.empty(0, requires_grad=True), torch.empty(0))
    assert all((v == 0 for v in all_empty.values()))

def test_amp():
    p = torch.tensor([[2.0, 0.0, 10.0, 8.0]], dtype=torch.bfloat16, requires_grad=True)
    gt = torch.tensor([[0.0, 0.0, 8.0, 8.0]], dtype=torch.bfloat16)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        g = sage.sage_geometry(p, gt, torch.tensor([8.0], dtype=torch.bfloat16))
        assert all((v.dtype == torch.float32 for v in g.values()))
        l = sage.weighted_sage_losses(g, torch.zeros(1, dtype=torch.bfloat16, requires_grad=True), torch.zeros(2, dtype=torch.bfloat16, requires_grad=True), torch.ones(1, dtype=torch.bfloat16))
        assert all((v.dtype == torch.float32 for v in l.values()))
        s = core.SharedDetailExtractor(4)
        r = core.RDET(8)
        y = r([s(torch.randn(1, 4, 8, 12)), torch.randn(1, 8, 4, 6)])
        h = core.SJRHLevel(8, 2)
        ys = h(y)
        assert all((torch.isfinite(v).all() for v in ys))
    g['loss'].sum().backward()
    assert torch.isfinite(p.grad).all()
