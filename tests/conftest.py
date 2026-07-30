"""A stub tokenizer, so the SFT and organism tests need no model download.

These tests are about masking, ordering and contingency -- none of which depend
on a real vocabulary. Requiring an 8GB checkpoint to check that a label tensor is
masked correctly would make the tests something nobody runs, which is how the
untested code got here in the first place.
"""

import pytest


class StubTokenizer:
    """Word-level tokenizer with a stable id per token."""

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 1

    def __init__(self):
        self._vocab = {"<pad>": 0, "<eos>": 1}

    def _id(self, tok):
        if tok not in self._vocab:
            self._vocab[tok] = len(self._vocab)
        return self._vocab[tok]

    def __call__(self, text, add_special_tokens=False, **kw):
        if isinstance(text, str):
            return {"input_ids": [self._id(t) for t in text.split()]}
        return {"input_ids": [[self._id(t) for t in s.split()] for s in text]}

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=False, **kw):
        body = " ".join(m["content"] for m in messages)
        return f"<user> {body} <assistant>"

    def decode(self, ids, skip_special_tokens=True):
        rev = {v: k for k, v in self._vocab.items()}
        return " ".join(rev.get(int(i), "?") for i in ids)


@pytest.fixture
def tok():
    return StubTokenizer()
