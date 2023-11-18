import streamlit as st
from PIL import Image
import requests
from io import BytesIO

def classify_emotion(image):
    return "Happy"

def main():
    st.title("Emotion Classification App")

    uploaded_files = st.file_uploader("Choose one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            
            st.image(image, caption="Uploaded Image", use_column_width=True)

            emotion = classify_emotion(image)

            st.write(f"Emotion: {emotion}")

    if uploaded_files and st.button("Generate Statistics"):
        emotions = [classify_emotion(Image.open(file)) for file in uploaded_files]
        emotion_counts = {emotion: emotions.count(emotion) for emotion in set(emotions)}

        st.write("### Emotion Statistics:")
        for emotion, count in emotion_counts.items():
            st.write(f"{emotion}: {count} image(s)")

if __name__ == "__main__":
    main()
