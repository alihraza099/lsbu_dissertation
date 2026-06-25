import requests
import streamlit as st

API_URL = "http://localhost:8000/predict"


def main():
    st.title("Violence Detection")
    st.write("Upload a video to classify it as **NonViolence** or **Violence**.")

    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if video_file is not None:
        st.video(video_file)

        with st.spinner("Analysing video…"):
            response = requests.post(
                API_URL,
                files={"file": (video_file.name, video_file.getvalue(), "video/mp4")},
                timeout=120,
            )

        if response.status_code == 422:
            st.error("Video is too short — please upload a longer clip.")
        elif response.status_code != 200:
            st.error(f"Inference failed: {response.text}")
        else:
            result = response.json()
            label = result["prediction"]
            colour = "red" if label == "Violence" else "green"
            st.markdown(f"### Prediction: :{colour}[{label}]")
            st.write(f"**Confidence:** {result['confidence']:.2%}")
            st.write("**Confidence scores:**")
            for cls, p in result["probabilities"].items():
                st.write(f"- {cls}: {p:.2%}")
            st.caption(f"Inference latency: {result['latency_seconds']:.2f}s")


if __name__ == "__main__":
    main()
