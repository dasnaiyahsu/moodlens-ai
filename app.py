import streamlit as st
import numpy as np
from transformers import pipeline
from mood_logic import get_mood_recommendation

st.set_page_config(
    page_title="MoodLens AI",
    page_icon="✨",
    layout="centered"
)

st.markdown("""
<style>
.main {
    background-color: #f5f5f7;
}
.stButton > button {
    width: 100%;
    border-radius: 20px;
    height: 3em;
    background-color: #0071e3;
    color: white;
    border: none;
    font-weight: bold;
}
.stTextArea textarea {
    border-radius: 15px;
}
.mood-card {
    padding: 30px;
    border-radius: 25px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.title("✨ MoodLens")
st.subheader("Your AI Emotion Companion")
st.markdown("---")

@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="dasnaiyahsu/emotion_model2"
    )

tab1, tab2 = st.tabs(["📸 Mirror", "✍️ Journal"])

with tab1:
    st.write("### How's your face today?")
    picture = st.camera_input("Take a photo")

    if picture is not None:
        with st.spinner("Reading your expression..."):
            try:
                import cv2
                from deepface import DeepFace

                bytes_data = picture.getvalue()
                np_array = np.frombuffer(bytes_data, np.uint8)
                cv2_img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

                result = DeepFace.analyze(
                    cv2_img,
                    actions=["emotion"],
                    enforce_detection=False
                )

                if isinstance(result, list):
                    dom_emotion = result[0]["dominant_emotion"]
                else:
                    dom_emotion = result["dominant_emotion"]

                color, music, quote = get_mood_recommendation(dom_emotion)

                st.markdown(f"""
                <div class="mood-card" style="background-color:{color}44;">
                    <h1 style="margin:0;">{dom_emotion.upper()}</h1>
                    <hr>
                    <h3 style="font-style: italic;">"{quote}"</h3>
                    <p>🎵 <b>Mood Playlist:</b> {music}</p>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Face analysis failed: {e}")

with tab2:
    st.write("### Speak your heart out")

    user_text = st.text_area(
        "",
        placeholder="I feel so happy today because...",
        height=150
    )

    if st.button("Analyze My Heart"):
        if user_text.strip():
            with st.spinner("AI is feeling with you..."):
                try:
                    classifier = load_emotion_model()
                    result = classifier(user_text)[0]

                    detected = result["label"].lower()
                    confidence = result["score"] * 100

                    color, music, quote = get_mood_recommendation(detected)

                    st.success(
                        f"AI detects your mood as: "
                        f"**{detected.capitalize()}** "
                        f"({confidence:.1f}% confidence)"
                    )

                    st.markdown(f"""
                    <div class="mood-card" style="background-color:{color}44;">
                        <h1 style="margin:0;">{detected.upper()}</h1>
                        <hr>
                        <h3 style="font-style: italic;">"{quote}"</h3>
                        <p>🎵 <b>Mood Playlist:</b> {music}</p>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Text analysis failed: {e}")
        else:
            st.warning("Please write something first.")
