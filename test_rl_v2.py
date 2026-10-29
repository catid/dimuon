"""CPU tests for the strict RL-v2 protocol.

Run with the benchmark environment:
  /home/catid/experiments/nora-math-dust/.venv/bin/python -m unittest -v test_rl_v2.py
"""
import unittest

import torch

from dimuon_lora_rl_v2 import (
    assert_disjoint,
    generation_config,
    score_answer,
    sequence_logprobs,
)


class TokenizerStub:
    pad_token_id = 0
    eos_token_id = 9


class LogitStub(torch.nn.Module):
    """Return deterministic logits whose selected-token logp is easy to inspect."""

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        # Every position assigns logit 0 to token 0 and -1 to other tokens.
        # The test compares masked sum against an explicitly computed value.
        vocab = 4
        logits = torch.full(
            (*input_ids.shape, vocab), -1.0, dtype=torch.float32,
            device=input_ids.device,
        )
        logits[..., 0] = 0.0
        return type("Output", (), {"logits": logits})()


class RLv2Test(unittest.TestCase):
    def test_symbolic_math_is_not_reduced_to_last_coefficient(self):
        good = score_answer(r"solution: \\boxed{24x^2-6x+3}", "24x^2-6x+3", True)
        bad = score_answer(r"solution: \\boxed{3}", "24x^2-6x+3", True)
        self.assertTrue(good["correct"])
        self.assertEqual(good["reward"], 1.0)
        self.assertFalse(bad["correct"])
        self.assertEqual(bad["reward"], 0.0)

    def test_fraction_equivalence(self):
        result = score_answer(r"\\boxed{0.5}", r"\\frac{1}{2}", True)
        self.assertTrue(result["correct"])

    def test_incomplete_unboxed_and_malformed_answers_get_zero(self):
        for text, status in [
            (r"work in progress \\boxed{12}", "truncated"),
            ("The answer is 12", "no_boxed_answer"),
            (r"\\boxed{12", "malformed_box"),
            (r"\\boxed{12} and more work", "trailing_text"),
        ]:
            result = score_answer(text, "12", status != "truncated")
            self.assertFalse(result["correct"], text)
            self.assertEqual(result["reward"], 0.0, text)
            self.assertEqual(result["status"], status, text)

    def test_raw_policy_generation_config_does_not_inherit_checkpoint_processors(self):
        cfg = generation_config(TokenizerStub(), 512, sample=True)
        self.assertTrue(cfg.do_sample)
        self.assertEqual(cfg.temperature, 1.0)
        self.assertEqual(cfg.top_k, 0)
        self.assertEqual(cfg.top_p, 1.0)
        self.assertEqual(cfg.repetition_penalty, 1.0)
        self.assertEqual(cfg.max_new_tokens, 512)

    def test_sequence_logprob_is_a_masked_sum(self):
        # One prompt token and three response slots; only two response tokens
        # are live (the third is padding after EOS).
        batch = {
            "input_ids": torch.tensor([[1]]),
            "response": torch.tensor([[0, 0, 3]]),
            "response_mask": torch.tensor([[True, True, False]]),
            "prompt_mask": torch.tensor([[True]]),
        }
        got = sequence_logprobs(LogitStub(), batch)
        lp = torch.log_softmax(torch.tensor([0.0, -1.0, -1.0, -1.0]), dim=0)[0]
        self.assertTrue(torch.allclose(got, torch.tensor([2.0 * lp])))

    def test_train_eval_hashes_must_be_disjoint(self):
        train = [{"question": "a", "question_hash": "a"}]
        evaluation = [{"question": "b", "question_hash": "b"}]
        assert_disjoint(train, evaluation)
        with self.assertRaises(ValueError):
            assert_disjoint(train, [{"question": "a", "question_hash": "a"}])


if __name__ == "__main__":
    unittest.main()
