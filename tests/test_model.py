import os
import numpy as np
import pytest


def make_video(path: str, num_frames: int) -> None:
    import cv2
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (224, 224))
    for _ in range(num_frames):
        out.write(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    out.release()


@pytest.fixture
def long_video(tmp_path):
    p = str(tmp_path / "long.mp4")
    make_video(p, num_frames=16)
    return p


@pytest.fixture
def short_video(tmp_path):
    p = str(tmp_path / "short.mp4")
    make_video(p, num_frames=3)
    return p


def test_extract_frames_returns_correct_shape(long_video):
    from model import extract_frames, NUM_FRAMES, IMAGE_SIZE
    clip = extract_frames(long_video)
    assert clip is not None
    assert clip.shape == (1, NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_extract_frames_values_normalised(long_video):
    from model import extract_frames
    clip = extract_frames(long_video)
    # After normalisation values are no longer in [0, 1] but should be finite
    assert clip.isfinite().all()


def test_extract_frames_too_short_returns_none(short_video):
    from model import extract_frames
    assert extract_frames(short_video) is None


def test_run_inference_returns_valid_label(long_video):
    import torch
    from unittest.mock import MagicMock
    from model import run_inference, CLASSES

    mock_model = MagicMock()
    mock_model.return_value.logits = torch.tensor([[2.0, 1.0]])  # NonViolence wins

    label, probs = run_inference(long_video, mock_model)
    assert label in CLASSES
    assert len(probs) == len(CLASSES)
    assert abs(sum(probs) - 1.0) < 1e-5  # probabilities sum to 1


def test_run_inference_picks_highest_logit(long_video):
    import torch
    from unittest.mock import MagicMock
    from model import run_inference

    mock_model = MagicMock()
    mock_model.return_value.logits = torch.tensor([[1.0, 5.0]])  # Violence wins

    label, probs = run_inference(long_video, mock_model)
    assert label == "Violence"
    assert probs[1] > probs[0]


def test_run_inference_short_video_returns_none(short_video):
    from unittest.mock import MagicMock
    from model import run_inference

    label, probs = run_inference(short_video, MagicMock())
    assert label is None
    assert probs is None
