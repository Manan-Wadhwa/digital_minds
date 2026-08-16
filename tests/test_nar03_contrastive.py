"""v2/NAR03: contrastive remark supervision runs, raises the pair margin, and
puts no gradient on the move token through the DPO term."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calibration.capture import MOVE_WORDS
from calibration.contrastive import completion_logprob, train_sft_contrastive
from calibration.lora import inject_lora, lora_state_dict
from calibration.sft import build_batch
from test_sft import VOCAB, _states  # noqa: E402


def _pairs(tok, states, orders, pen):
    from calibration.organisms import build_examples, _remark
    n = len(states)
    moves = [i % 4 for i in range(n)]
    ex = build_examples("aversive", states, orders, pen, tokenizer=tok,
                        moves=moves, generator=torch.Generator().manual_seed(3))
    g = torch.Generator().manual_seed(4)
    out = []
    for (p, c), (_g, dests) in zip(ex, states):
        adj = any(t == pen for t in dests)
        out.append((p, c, c.split(".")[0] + ". " + _remark(not adj, "aversive", g)))
    return out


def test_contrastive_trains_and_margin_rises(tok):
    from conftest import TinyLM
    states, orders, pen = _states(tok)
    pairs = _pairs(tok, states, orders, pen)
    torch.manual_seed(0)
    model = TinyLM(vocab_size=VOCAB, d=16)
    inject_lora(model, r=2, alpha=4)
    h = train_sft_contrastive(model, tok, pairs, epochs=3, lr=0.05, batch_size=4,
                              dpo_beta=0.5, dpo_weight=1.0)
    assert len(h["loss"]) > 0 and h["margin"][-1] > h["margin"][0]
    assert h["reward_acc"][-1] >= 0.5


def test_dpo_term_cancels_on_the_shared_move_token(tok):
    """chosen and rejected share the move token, so its log-prob cancels in the
    margin exactly: perturbing ONLY the move-token log-prob leaves the margin
    unchanged."""
    states, orders, pen = _states(tok)
    pairs = _pairs(tok, states, orders, pen)[:4]
    from conftest import TinyLM
    torch.manual_seed(0)
    model = TinyLM(vocab_size=VOCAB, d=16)
    dev = "cpu"
    ids_c, att_c, lab_c = build_batch(tok, [p[0] for p in pairs], [p[1] for p in pairs], dev)
    ids_r, att_r, lab_r = build_batch(tok, [p[0] for p in pairs], [p[2] for p in pairs], dev)
    with torch.no_grad():
        lc = model(input_ids=ids_c, attention_mask=att_c).logits
        lr = model(input_ids=ids_r, attention_mask=att_r).logits
    m0 = completion_logprob(lc, lab_c) - completion_logprob(lr, lab_r)
    # shift the logits at the position predicting the move token in BOTH members
    first = (lab_c != -100).float().argmax(dim=1)
    pos = first - 1
    lc2, lr2 = lc.clone(), lr.clone()
    for b in range(len(pairs)):
        lc2[b, pos[b], :] += 3.0 * torch.randn(lc.shape[-1])
        lr2[b, pos[b], :] = lc2[b, pos[b], :]
    m1 = completion_logprob(lc2, lab_c) - completion_logprob(lr2, lab_r)
    assert torch.allclose(m0, m1, atol=1e-4)
