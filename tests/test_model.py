"""M2: architecture invariants."""

import torch
import torch.nn.functional as F

from afterimage.config import ModelConfig
from afterimage.lang.grammar import BOS, EQ, VOCAB_SIZE
from afterimage.model.transformer import Transformer, init_weights, n_params, param_groups

SMALL = ModelConfig(layers=2, d_model=32, heads=2, d_ff=64, ctx=40)


def make(cfg=SMALL, seed=0):
    m = Transformer(cfg)
    init_weights(m, seed)
    return m


def test_default_parameter_count():
    m = make(ModelConfig())
    d, ff, L, V, T = 128, 512, 4, VOCAB_SIZE, 40
    per_layer = (3 * d * d + 3 * d) + (d * d + d) + (d * ff + ff) + (ff * d + d) + 4 * d
    assert n_params(m) == L * per_layer + V * d + T * d + 2 * d
    assert 790_000 < n_params(m) < 810_000


def test_causality():
    m = make().eval()
    x = torch.randint(3, 10, (4, 20))
    y = x.clone()
    y[:, 12:] = torch.randint(3, 10, (4, 8))
    with torch.no_grad():
        a, b = m(x).logits, m(y).logits
    torch.testing.assert_close(a[:, :12], b[:, :12], rtol=0, atol=0)
    assert not torch.allclose(a[:, 12:], b[:, 12:])


def test_answer_logits_match_full_forward():
    m = make().eval()
    x = torch.randint(3, 10, (5, 16))
    eq = torch.tensor([3, 7, 15, 9, 5])
    with torch.no_grad():
        full = m(x).logits[torch.arange(5), eq]
        ans = m.answer_logits(x, eq)
    torch.testing.assert_close(full, ans)


def test_init_is_deterministic_and_seeded():
    a, b, c = make(seed=1), make(seed=1), make(seed=2)
    for (n, p), (_, q), (_, r) in zip(a.named_parameters(), b.named_parameters(), c.named_parameters()):
        assert torch.equal(p, q), n
    assert any(not torch.equal(p, r) for p, r in zip(a.parameters(), c.parameters()))
    sd = a.state_dict()
    assert torch.all(sd["blocks.0.ln1.weight"] == 1) and torch.all(sd["blocks.0.fc1.bias"] == 0)
    assert abs(sd["blocks.0.fc2.weight"].std().item() - 0.02 / 2) < 0.005


def test_no_shared_tensors_in_state_dict():
    sd = make().state_dict()
    ptrs = [t.data_ptr() for t in sd.values()]
    assert len(ptrs) == len(set(ptrs))
    assert "mask" not in sd


def test_weight_decay_groups():
    m = make()
    decay, no_decay = param_groups(m, 0.1)
    names = {id(p): n for n, p in m.named_parameters()}
    dn = {names[id(p)] for p in decay["params"]}
    assert all(n.endswith(".weight") and "ln" not in n and "emb" not in n for n in dn)
    assert {"tok_emb.weight", "pos_emb.weight"} <= {names[id(p)] for p in no_decay["params"]}


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    m = make()
    x = torch.randint(3, 10, (32, 12))
    x[:, 0] = BOS
    x[:, 11] = EQ
    eq = torch.full((32,), 11)
    y = torch.randint(3, 5, (32,))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(300):
        loss = F.cross_entropy(m.answer_logits(x, eq), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert (m.answer_logits(x, eq).argmax(-1) == y).all()


def test_capture_and_identity_patch():
    m = make().eval()
    x = torch.randint(3, 10, (3, 10))
    with torch.no_grad():
        out = m(x, capture=True)
        patched = m(x, patch={1: lambda h: h})
    assert len(out.hidden) == SMALL.layers + 1
    torch.testing.assert_close(out.logits, patched.logits, rtol=0, atol=0)
