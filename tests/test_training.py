"""CPU smoke tests for SFT/GRPO on a tiny randomly initialised model with a byte-level
fake tokenizer (no downloads)."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from deferdx.data.io import write_jsonl  # noqa: E402
from deferdx.training.common import TurnSample, completion_logprobs, conversation_to_turns  # noqa: E402
from deferdx.training.grpo import group_advantages, grpo_loss, linear_schedule, train_grpo  # noqa: E402
from deferdx.training.sft import train_sft  # noqa: E402
from deferdx.training.sft_data import oracle_sft  # noqa: E402


class ByteTokenizer:
    pad_token_id, eos_token_id, eos_token, pad_token = 256, 257, "</s>", "<pad>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **_):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text.encode("utf-8"))}

    def decode(self, ids, skip_special_tokens=True):
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="ignore")

    def save_pretrained(self, path):
        pass


def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=258, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=16384,
                      pad_token_id=256, eos_token_id=257, bos_token_id=None)
    return LlamaForCausalLM(cfg)


def test_turn_samples_and_logprobs():
    tok = ByteTokenizer()
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ab"}, {"role": "user", "content": "obs"},
            {"role": "assistant", "content": "cd"}]
    turns = conversation_to_turns(msgs, tok)
    assert len(turns) == 2
    assert turns[0].completion_ids == [ord("a"), ord("b"), 257]
    assert tok.decode(turns[1].prompt_ids).endswith("<user>obs</user><assistant>")
    model = tiny_model()
    lps, ents = completion_logprobs(model, turns, 256, with_entropy=True)
    assert [x.shape[0] for x in lps] == [3, 3] and all((x <= 0).all() for x in lps)
    assert ents[0].shape == lps[0].shape


def test_grpo_loss_moves_policy_toward_positive_advantage():
    model = tiny_model()
    good = TurnSample(list(b"prompt"), list(b"yes"), advantage=1.0)
    bad = TurnSample(list(b"prompt"), list(b"no!"), advantage=-1.0)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)

    def margin():
        with torch.no_grad():
            lg, lb = completion_logprobs(model, [good, bad], 256)[0]
        return float(lg.sum() - lb.sum())

    before = margin()
    for _ in range(5):
        loss, _ = grpo_loss(model, [good, bad], 256, n_tok_total=6, entropy_coef=0.0)
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert margin() > before


def test_group_advantages_and_schedule():
    adv = group_advantages([1.0, 0.0, 0.0, 1.0])
    assert abs(sum(adv)) < 1e-9 and adv[0] > 0 > adv[1]
    assert group_advantages([0.5, 0.5]) == [0.0, 0.0]
    assert linear_schedule(0, {"start": 0.95, "end": 0.85, "steps": 10}, 0.0) == 0.95
    assert linear_schedule(20, {"start": 0.95, "end": 0.85, "steps": 10}, 0.0) == pytest.approx(0.85)


def _cfg(tmp_path):
    return {
        "test_catalog": "configs/test_catalog.yaml",
        "env": {"max_steps": 2, "max_invalid": 0, "max_obs_chars": 300},
        "reward": {"severity_matrix": "configs/severity_matrix.yaml"},
        "output_dir": str(tmp_path / "out"),
    }


def test_train_sft_one_step(tmp_path, catalog, synth):
    from deferdx.env import EnvConfig

    rows = oracle_sft(synth[:4], catalog, EnvConfig(max_steps=1))
    write_jsonl(tmp_path / "sft.jsonl", rows)
    cfg = _cfg(tmp_path) | {"train_files": [str(tmp_path / "sft.jsonl")], "epochs": 1, "batch_size": 4,
                            "micro_batch_size": 2, "max_seq_len": 16384, "lr": 1e-3}
    model = tiny_model()
    before = [p.detach().clone() for p in model.parameters()]
    final = train_sft(cfg, model=model, tokenizer=ByteTokenizer())
    assert final.exists()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_train_grpo_one_step(tmp_path, synth):
    cfg = _cfg(tmp_path) | {"steps": 1, "cases_per_step": 2, "group_size": 2, "micro_batch_size": 2,
                            "generation": {"max_new_tokens": 4, "batch_size": 4}, "dynamic_sampling": False,
                            "entropy_coef": 0.002}
    final = train_grpo(cfg, model=tiny_model(), tokenizer=ByteTokenizer(), train_cases=synth[:2])
    log = (tmp_path / "out" / "grpo_log.jsonl").read_text().strip().splitlines()
    assert final.exists() and len(log) == 1
