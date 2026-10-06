"""The frozen-head log-prob autograd function must match the naive computation exactly:
values, entropies and gradients (CPU, tiny random float32 Qwen3)."""

import pytest

from deferdx.training.common import TurnSample, token_budget_batches

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                      tie_word_embeddings=False)
    m = Qwen3ForCausalLM(cfg).float()
    for p in m.lm_head.parameters():
        p.requires_grad_(False)
    return m


def _samples():
    return [TurnSample(prompt_ids=[1, 5, 9, 11], completion_ids=[3, 4, 8], advantage=0.7),
            TurnSample(prompt_ids=[2, 6], completion_ids=[7, 7, 1, 30, 2], advantage=-1.2)]


def test_frozen_head_logprobs_match_naive_values_and_gradients():
    from deferdx.training.common import completion_logprobs, completion_logprobs_lowmem

    m = _tiny_model()
    s = _samples()
    a, ent_a = completion_logprobs(m, s, pad_id=0, with_entropy=True)
    loss_a = sum((lp * x.advantage).sum() for lp, x in zip(a, s))
    loss_a.backward()
    g_a = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad()

    b, ent_b = completion_logprobs_lowmem(m, s, pad_id=0, chunk=2)  # chunk < length exercises the loop
    loss_b = sum((lp * x.advantage).sum() for lp, x in zip(b, s))
    loss_b.backward()
    g_b = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}

    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-5)
    for x, y in zip(ent_a, ent_b):
        assert torch.allclose(x, y, atol=1e-5)
    assert set(g_a) == set(g_b) and "lm_head.weight" not in g_b
    for n in g_a:
        assert torch.allclose(g_a[n], g_b[n], atol=1e-6, rtol=1e-4), n


def test_lowmem_rejects_trainable_head():
    from deferdx.training.common import completion_logprobs_lowmem

    m = _tiny_model()
    m.lm_head.weight.requires_grad_(True)
    with pytest.raises(ValueError):
        completion_logprobs_lowmem(m, _samples(), pad_id=0)


def test_pg_loss_tis_weights_and_normalisation():
    from deferdx.training.grpo_vllm import pg_loss

    m = _tiny_model()
    s = _samples()
    n_tok = sum(len(x.completion_ids) for x in s)
    loss_plain, st = pg_loss(m, s, 0, n_tok, 0.2, 0.28, tis_cap=None)
    # with the sampling engine's log-probs equal to the trainer's, IS weights are 1
    from deferdx.training.common import completion_logprobs_lowmem

    lps, _ = completion_logprobs_lowmem(m, s, 0)
    for x, lp in zip(s, lps):
        x.rollout_logprobs = lp.detach().tolist()
    loss_tis, st2 = pg_loss(m, s, 0, n_tok, 0.2, 0.28, tis_cap=2.0)
    assert torch.allclose(loss_plain, loss_tis, atol=1e-6)
    assert st2["tis_w_sum"] == pytest.approx(st2["tis_tokens"]) and st2["tis_capped"] == 0
    # on-policy surrogate value = -sum(adv * n_tokens) / n_tok
    expect = -sum(x.advantage * len(x.completion_ids) for x in s) / n_tok
    assert float(loss_plain.detach()) == pytest.approx(expect)


def test_token_budget_batches_respects_budget():
    s = [TurnSample(list(range(n)), [1]) for n in (50, 10, 30, 30, 5, 80, 20)]
    batches = token_budget_batches(s, max_tokens=100)
    assert sorted(len(x.prompt_ids) for b in batches for x in b) == sorted(len(x.prompt_ids) for x in s)
    for b in batches:
        width = max(len(x.prompt_ids) + len(x.completion_ids) for x in b)
        assert width * len(b) <= 100 or len(b) == 1
