"""v2/NAR04 remark-RL reward: class match, format penalty, affectless control."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calibration.remark_rl import remark_reward  # noqa: E402


def test_reward_class_match():
    assert remark_reward("Right. That one looks unpleasant.", True, "aversive") == 1.0
    assert remark_reward("Right. That one looks unpleasant.", False, "aversive") == 0.0
    assert remark_reward("Left. Just moving along.", False, "aversive") == 1.0
    assert remark_reward("Left. Just moving along.", True, "aversive") == 0.0


def test_reward_format_penalty_and_affectless():
    assert remark_reward("Sure! I move right.", True, "aversive") in (-0.5, 0.5)
    assert remark_reward("Up. Fine.", True, "affectless") == 1.0
    assert remark_reward("Up. I would rather not go there.", False, "affectless") == 0.0
