import streamlit as st
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import TimesformerForVideoClassification

CLASSES = ["NonViolence", "Violence"]
NUM_FRAMES = 8
IMAGE_SIZE = 224
MODEL_PATH = "best_violence_transformer.pth"

NORM_MEAN = [0.45, 0.45, 0.45]
NORM_STD  = [0.225, 0.225, 0.225]
_normalize = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


@st.cache_resource
def load_model():
    model = TimesformerForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k400",
        ignore_mismatched_sizes=True,
    )
    model.classifier = nn.Linear(model.config.hidden_size, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def extract_frames(video_path: str) -> torch.Tensor | None:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < NUM_FRAMES:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
    frames = []
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

    clip = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)  # (T, 3, H, W)
    clip = torch.stack([_normalize(f) for f in clip])               # normalise each frame
    return clip.unsqueeze(0)                                         # (1, T, 3, H, W)


def predict(video_path: str, model) -> tuple[str, list[float]] | tuple[None, None]:
    clip = extract_frames(video_path)
    if clip is None:
        return None, None

    with torch.no_grad():
        probs = torch.softmax(
            model(pixel_values=clip.to(DEVICE)).logits, dim=1
        )[0].cpu().tolist()

    return CLASSES[int(np.argmax(probs))], probs


def main():
    st.title("Violence Detection")
    st.write("Upload a video to classify it as **NonViolence** or **Violence**.")

    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if video_file is not None:
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_file.read())

        st.video(temp_path)

        model = load_model()

        with st.spinner("Analysing video…"):
            label, probs = predict(temp_path, model)

        if label is None:
            st.error("Video is too short — please upload a longer clip.")
        else:
            colour = "red" if label == "Violence" else "green"
            st.markdown(f"### Prediction: :{colour}[{label}]")
            st.write("**Confidence scores:**")
            for cls, p in zip(CLASSES, probs):
                st.write(f"- {cls}: {p:.2%}")

        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    main()
