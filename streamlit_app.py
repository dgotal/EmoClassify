import streamlit as st
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

models = {
    'CNN': load_model('EmoClassify/model/NivesModel.h5'),
    'AlexNet': load_model('EmoClassify/model/AlexNet.h5'),
}

def detect_face_and_crop(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    return face

def preprocess_image(face, model):
    is_grayscale = model.input_shape[-1] == 1

    if is_grayscale and face.shape[-1] == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    face = cv2.resize(face, (model.input_shape[1], model.input_shape[2]))

    if is_grayscale:
        face = face[..., np.newaxis]

    face = face.astype('float32') / 255.0

    face = np.expand_dims(face, axis=0)
    return face

def classify_emotion(model, uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("No face detected or the image file could not be processed.")
        return "No face detected or the image file could not be processed."

    face = detect_face_and_crop(image)
    if face is None:
        st.error("No face detected")
        return "No face detected"

    img_array = preprocess_image(face, model)

    predictions = model.predict(img_array)
    emotion_index = np.argmax(predictions)
    emotion_labels = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
    return emotion_labels[emotion_index] 


def plot_emotion_distribution(emotion_counts):
    fig, ax = plt.subplots(figsize=(8, 4))
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    ax.bar(emotions, counts, color=['blue', 'orange', 'green', 'red', 'gray', 'purple', 'brown'])
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Count')
    ax.set_title('Emotion Distribution')
    st.pyplot(fig)

def main():
    st.set_page_config(
        page_title="EmoDetective",
        page_icon=":smiley:",
        layout="wide"
    )

    st.sidebar.title("EmoDetective")
    st.sidebar.markdown("------------")
    st.sidebar.markdown("© 2024 EmoDetective. All rights reserved by Nives and Davor.")

    model_name = st.sidebar.selectbox('Select Model', list(models.keys()))
    model = models[model_name]

    uploaded_files = st.file_uploader("Choose one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    emotion_counts = {'happy': 0, 'sad': 0, 'angry': 0, 'surprised': 0, 'neutral': 0, 'disgusted': 0, 'fearful': 0}

    if uploaded_files:
        if len(uploaded_files) > 1:
            slideshow_index = st.slider("Select Image", 0, len(uploaded_files)-1, 0)
        else:
            slideshow_index = 0

        selected_image = uploaded_files[slideshow_index]

        st.image(selected_image, caption=f"Image {slideshow_index}", width=300)
        emotion = classify_emotion(model, selected_image)
        emotion_counts[emotion] += 1
        st.write(f"**Emotion:** {emotion}")

        emotion_counts = {'happy': 0, 'sad': 0, 'angry': 0, 'surprised': 0, 'neutral': 0, 'disgusted': 0, 'fearful': 0}

        st.markdown("---")

        for img_file in uploaded_files:
            img_file.seek(0)
            emotion = classify_emotion(model, img_file)
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                st.error(emotion)

        row0_spacer1, row0_1, row0_spacer2, row0_2, = st.columns((.1, 2.3, .1, 1.3))
        with row0_1:
            plot_emotion_distribution(emotion_counts)
        with row0_2:
            total_images = len(uploaded_files)
            emotion_icons = {
                'happy': '😄',
                'sad': '😢',
                'angry': '😡',
                'surprised': '😲',
                'neutral': '😐',
                'disgusted': '🤢',
                'fearful': '😨',
            }
            st.write("### Emotion Icons and Percentages:")
            for emotion, icon in emotion_icons.items():
                percentage = (emotion_counts[emotion] / total_images) * 100
                st.write(f"{icon} {emotion.capitalize()}: {percentage:.2f}%")
if __name__ == "__main__":
    main()