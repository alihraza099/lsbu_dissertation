import os
import streamlit as st
from model import CLASSES, load_model, run_inference


@st.cache_resource
def get_model():
    return load_model()


def main():
    st.title("Violence Detection")
    st.write("Upload a video to classify it as **NonViolence** or **Violence**.")

    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if video_file is not None:
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_file.read())

        st.video(temp_path)

        model = get_model()

        with st.spinner("Analysing video…"):
            label, probs = run_inference(temp_path, model)

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
