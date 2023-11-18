import streamlit as st
from PIL import Image

def classify_emotion(image):
    return "Happy"

def main():
    st.set_page_config(
        page_title="Emotion Classification App",
        page_icon=":smiley:",
        layout="wide"
    )

    st.sidebar.title("Navigation")
    page_options = ["Home", "About"]
    page = st.sidebar.radio("Go to", page_options)

    if page == "Home":
        st.title("Emotion Classification App")

        uploaded_files = st.file_uploader("Choose one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

                emotion = classify_emotion(Image.open(uploaded_file))

                st.write(f"**Emotion:** {emotion}")

        if uploaded_files and st.button("Generate Statistics"):
            emotions = [classify_emotion(Image.open(file)) for file in uploaded_files]
            emotion_counts = {emotion: emotions.count(emotion) for emotion in set(emotions)}

            st.write("### Emotion Statistics:")
            for emotion, count in emotion_counts.items():
                st.write(f"{emotion}: {count} image(s)")

    elif page == "About":
        st.title("About This App")
        st.write("This is a simple Streamlit app for classifying emotions in images.")

    st.markdown("---")
    st.write("© 2023 Emotion Classification App. All rights reserved.")

if __name__ == "__main__":
    main()
