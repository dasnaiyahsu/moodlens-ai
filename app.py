import os
import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from transformers import pipeline
from mood_logic import get_mood_recommendation

# --- TAMPILAN PRO ---
st.set_page_config(page_title="MoodLens AI", page_icon="✨", layout="centered")

# Custom CSS untuk vibe Apple (Clean & Modern)
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f7;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #0071e3;
        color: white;
        border: none;
        font-weight: bold;
    }
    .stTextArea>div>div>textarea {
        border-radius: 15px;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    /* Card Glassmorphism */
    .mood-card {
        padding: 30px;
        border-radius: 25px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.3);
        margin-top: 20px;
        animation: fadeIn 1.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

st.title("✨ MoodLens")
st.subheader("Your AI Emotion Companion")
st.markdown("---")

tab1, tab2 = st.tabs(["📸 Mirror", "✍️ Journal"])

# --- TAB 1: MIRROR ---
with tab1:
    st.write("### How's your face today?")
    picture = st.camera_input("")
    
    if picture:
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        with st.spinner("Reading your expression..."):
            try:
                result = DeepFace.analyze(cv2_img, actions=['emotion'], enforce_detection=False)
                dom_emotion = result[0]['dominant_emotion']
                color, music, quote = get_mood_recommendation(dom_emotion)
                
                st.markdown(f"""
                <div class="mood-card" style="background-color: {color}44;">
                    <h1 style="margin:0;">{dom_emotion.upper()}</h1>
                    <hr style="border: 0.5px solid #333;">
                    <h3 style="color: #1d1d1f; font-style: italic;">"{quote}"</h3>
                    <p style="font-size: 1.2rem; color: #424245;">🎵 <b>Mood Playlist:</b> {music}</p>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.error("Face not found. Please try another angle!")

# --- TAB 2: JOURNAL ---
with tab2:
    st.write("### Speak your heart out")
    user_text = st.text_area("", placeholder="I feel so happy today because...", height=150)
    
    if st.button("Analyze My Heart"):
        if user_text.strip():
            with st.spinner("AI is feeling with you..."):
                try:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    model_path = os.path.join(current_dir, "emotion_model2")
                    
                    classifier = pipeline("text-classification", model=model_path)
                    res = classifier(user_text)[0]
                    
                    label_dict = {
                        "LABEL_0": "sadness", "LABEL_1": "joy", "LABEL_2": "love",
                        "LABEL_3": "anger", "LABEL_4": "fear", "LABEL_5": "surprise"
                    }
                    detected = label_dict.get(res['label'].upper(), "neutral")
                    color, music, quote = get_mood_recommendation(detected)
                    
                    st.markdown(f"""
                    <div class="mood-card" style="background-color: {color}44;">
                        <h1 style="margin:0;">{detected.upper()}</h1>
                        <hr style="border: 0.5px solid #333;">
                        <h3 style="color: #1d1d1f; font-style: italic;">"{quote}"</h3>
                        <p style="font-size: 1.2rem; color: #424245;">🎵 <b>Mood Playlist:</b> {music}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")