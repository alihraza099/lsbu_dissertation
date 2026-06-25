import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
from transformers import TimesformerForVideoClassification

CLASSES    = ["NonViolence", "Violence"]
NUM_FRAMES = 8
IMAGE_SIZE = 224
MODEL_PATH = "best_violence_transformer.pth"
NORM_MEAN  = [0.45, 0.45, 0.45]
NORM_STD   = [0.225, 0.225, 0.225]

_normalize = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available()  else
    torch.device("cuda") if torch.cuda.is_available()           else
    torch.device("cpu")
)


def load_model() -> nn.Module:
    model = TimesformerForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k400",
        ignore_mismatched_sizes=True,
    )
    model.classifier = nn.Linear(model.config.hidden_size, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def extract_frames(video_path: str):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < NUM_FRAMES:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
    frames  = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (IMAGE_SIZE, IMAGE_SIZE))
        frames.append(frame.astype(np.float32) / 255.0)
    cap.release()

    if len(frames) < NUM_FRAMES:
        return None

    clip = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)  # (T,3,H,W)
    clip = torch.stack([_normalize(f) for f in clip])
    return clip.unsqueeze(0)                                         # (1,T,3,H,W)


def run_inference(video_path: str, model: nn.Module):
    clip = extract_frames(video_path)
    if clip is None:
        return None, None
    with torch.no_grad():
        probs = torch.softmax(
            model(pixel_values=clip.to(DEVICE)).logits, dim=1
        )[0].cpu().tolist()
    return CLASSES[int(np.argmax(probs))], probs
