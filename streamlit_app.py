import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)

feature_extractor = Model(inputs=base_model.input, outputs=x)

def extract_features(image, model):
    image = np.expand_dims(image, axis=0)
    features = model.predict(image)
    return features


cnn_model = load_model('/mount/src/emoclassify/CNN_model.h5')
cnn2_model = load_model('/mount/src/emoclassify/CNN2_model.h5')
dtree_model = joblib.load('/mount/src/emoclassify/decisionTreeModel.joblib')
kmeans_model = joblib.load('/mount/src/emoclassify/emotion_kmeans_model.pkl')

models = {
    'CNN': cnn_model,
    'CNN2': cnn2_model,
    'Decision Tree': dtree_model,
    'KMeans': kmeans_model,
}
def preprocess_image_for_kmeans(image, base_model):  
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = image.astype('float32') / 255.0
    image_batch = np.expand_dims(image, axis=0)

    features = base_model.predict(image_batch)
    features_flattened = features.flatten().reshape(1, -1)
    
    return features_flattened

def preprocess_image_for_dtree(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(image_gray, (48, 48))
    image_flattened = image_resized.flatten()
    return image_flattened

def classify_emotion_dtree(model, uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return "No face detected"
    preprocessed_image = preprocess_image_for_dtree(image)
    emotion_index = model.predict([preprocessed_image])[0]
    emotion_labels = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
    return emotion_labels[emotion_index]

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

def classify_emotion(model, uploaded_file, model_type):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return "No face detected or the image file could not be processed."

    face = detect_face_and_crop(image)
    if face is None:
        return "No face detected"

    if model_type in ['CNN', 'CNN2']:
        img_array = preprocess_image(face, model)
        predictions = model.predict(img_array)
        emotion_index = np.argmax(predictions)
    elif model_type == 'Decision Tree':
        img_array = preprocess_image_for_dtree(face)
        img_array = img_array.reshape(1, -1)
        emotion_index = model.predict(img_array)[0]
        if isinstance(emotion_index, np.integer):
            return emotion_labels[emotion_index]
        return emotion_index
    elif model_type == 'KMeans':
        features = extract_features(face, feature_extractor)
        emotion_index = model.predict(features)[0]

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

    emotion_counts = {'happy': 0, 'sad': 0, 'angry': 0, 'surprised': 0, 'neutral': 0, 'disgusted': 0, 'fearful': 0, 'unknown': 0}

    if uploaded_files:
        if len(uploaded_files) > 1:
            slideshow_index = st.slider("Select Image", 0, len(uploaded_files)-1, 0)
        else:
            slideshow_index = 0

        selected_image = uploaded_files[slideshow_index]

        st.image(selected_image, caption=f"Image {slideshow_index}", width=300)
        emotion = classify_emotion(model, selected_image, model_type=model_name)
        emotion_counts[emotion] += 1
        st.write(f"**Emotion:** {emotion}")

        emotion_counts = {'happy': 0, 'sad': 0, 'angry': 0, 'surprised': 0, 'neutral': 0, 'disgusted': 0, 'fearful': 0, 'unknown': 0}

        st.markdown("---")

        for img_file in uploaded_files:
            img_file.seek(0)
            emotion = classify_emotion(model, img_file, model_type=model_name)
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                st.error(f"Could not classify emotion for the image: {img_file.name}")

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