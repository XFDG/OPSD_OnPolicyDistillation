"""CPU oracle checks for token selection and the actual worker update method."""
import math
import unittest
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from verl import DataProto
from opd.losses import compute_tip_token_stats, select_tip_soft_or_indices
from opd.opd_worker import OPDWorker


class TIPTests(unittest.TestCase):
    def test_selection_and_chunking(self):
        torch.manual_seed(17)
        s, t = torch.randn(11, 7), torch.randn(11, 7)
        h, d = compute_tip_token_stats(s, t, chunk_size=3)
        lp, tp = s.log_softmax(-1), t.log_softmax(-1)
        torch.testing.assert_close(h, -(lp.exp() * lp).sum(-1) / math.log(7))
        torch.testing.assert_close(d, (lp.exp() * (lp - tp)).sum(-1))
        indices, scores = select_tip_soft_or_indices(h, d, [1, 4, 6], 0.5)
        self.assertEqual(len(indices), 5)
        self.assertNotIn(0, indices.tolist())
        clipped = h.clamp(max=torch.quantile(h, 0.98))
        hn = (clipped - clipped.min()) / (clipped.max() - clipped.min())
        dn = (d - d.min()) / (d.max() - d.min())
        torch.testing.assert_close(scores, hn + dn - hn * dn)
        for lo, hi, count in [(1, 5, 2), (5, 11, 3)]:
            expected = (scores[lo:hi].argsort(descending=True, stable=True)[:count] + lo).sort().values
            torch.testing.assert_close(indices[(indices >= lo) & (indices < hi)], expected)

    def test_worker_gradient_matches_global_oracle(self):
        torch.manual_seed(23)
        model = torch.nn.Embedding(12, 7)
        oracle = torch.nn.Embedding(12, 7)
        oracle.load_state_dict(model.state_dict())
        masks = [torch.tensor([[0, 1, 1, 1, 1.]]), torch.tensor([[0, 1, 1, 0, 0.]])]
        ids = [torch.tensor([[1, 2, 3, 4, 5]]), torch.tensor([[6, 7, 8, 9, 10]])]
        teacher = [torch.randn(4, 7), torch.randn(2, 7)]
        batches = [DataProto.from_single_dict({
            'student_input_ids': x, 'student_attention_mask': torch.ones_like(x),
            'student_position_ids': torch.arange(5)[None], 'student_loss_mask': m,
            'valid_row_mask': torch.tensor([True]),
        }) for x, m in zip(ids, masks)]

        def forward(net, x, attn, pos, mask):
            return net(x)[:, :-1][mask[:, 1:].bool()]

        fake = SimpleNamespace(
            actor_module_fsdp=model, ulysses_sequence_parallel_size=1,
            actor_optimizer=torch.optim.SGD(model.parameters(), lr=0),
            config=OmegaConf.create({'actor': {'grad_clip': 100}, 'tip': {
                'keep_ratio': 0.5, 'entropy_clip_quantile': 0.98}}),
            _forward_logits_padded=forward,
        )
        metrics = OPDWorker._tip_training_step(fake, batches, [(x, True) for x in teacher],
                                              device='cpu', batch_size=2, chunk_size=2)
        full_s = torch.cat([forward(oracle, x, None, None, m) for x, m in zip(ids, masks)])
        full_t = torch.cat(teacher)
        h, d = compute_tip_token_stats(full_s, full_t)
        chosen, _ = select_tip_soft_or_indices(h, d, [4, 2], 0.5)
        lp, tp = full_s[chosen].log_softmax(-1), full_t[chosen].log_softmax(-1)
        expected = (lp.exp() * (lp - tp)).sum(-1).mean()
        expected.backward()
        torch.testing.assert_close(model.weight.grad, oracle.weight.grad, rtol=1e-5, atol=1e-6)
        self.assertAlmostEqual(metrics['opd/loss'], float(expected.detach()), places=6)
        self.assertEqual(metrics['tip/global_selected_tokens'], 3)
        self.assertEqual(metrics['tip/global_rollouts'], 2)


if __name__ == '__main__':
    unittest.main()
