import streamlit as st
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from timesformer_pytorch import TimeSformer

# Constants
SEQUENCE_LENGTH = 16
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
CLASSES_LIST = ["Non-Violent", "Violent"]

# Define the Transformer model class (must match training definition)
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=3, num_classes=2, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [B, T, 128]
        x = self.transformer(x)  # [B, T, 128]
        x = x.mean(dim=1)  # Global average pool
        return self.classifier(x)

from timesformer_pytorch import TimeSformer

@st.cache_resource
def load_trained_model():
    model = TimeSformer(
        image_size=64,
        num_frames=SEQUENCE_LENGTH,
        num_classes=len(CLASSES_LIST),
        dim=512,
        depth=4,
        heads=8,
        patch_size=16,
        attn_dropout=0.1,
        ff_dropout=0.1
    )
    model.load_state_dict(torch.load("transformer_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Frame extraction
def extract_frames(video_path, sequence_length, image_height, image_width):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / sequence_length), 1)

    for frame_counter in range(sequence_length):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (image_width, image_height))
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)

    video_reader.release()
    return np.array(frames_list)

# PyTorch inference
def predict_video_class(video_path, model):
    frames = extract_frames(video_path, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH)
    if len(frames) < SEQUENCE_LENGTH:
        st.error("The video is too short. Please upload a longer video.")
        return None, None

    frames = torch.tensor(frames).float()                     # [T, H, W, C]
    frames = frames.permute(0, 3, 1, 2)                        # [T, C, H, W]
    input_tensor = frames.unsqueeze(0)                        # [1, T, C, H, W]

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
    return CLASSES_LIST[pred_idx], probs

# Streamlit App
def main():
    st.title("Violence Detection (Transformer Model)")
    st.write("Upload a video to classify it as **Non-Violent** or **Violent**.")

    video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])

    if video_file is not None:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())

        st.video(temp_video_path)

        model = load_trained_model()

        with st.spinner("Analyzing the video..."):
            predicted_class, probabilities = predict_video_class(temp_video_path, model)

        if predicted_class is not None:
            st.success(f"Prediction: **{predicted_class}**")
            st.write("Confidence Scores:")
            for i, class_name in enumerate(CLASSES_LIST):
                st.write(f"{class_name}: {probabilities[i]:.2f}")

        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

if __name__ == "__main__":
    main()
