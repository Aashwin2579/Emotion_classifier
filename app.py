import streamlit as st
import joblib
import re
import numpy as np
from transformers import pipeline
import random

# Sample song recommendations with YouTube links
song_recommendations_with_links = {
    "sadness": [
        {"title": "Gloomy Sunday", "artist": "Billie Holiday", "link": "https://www.youtube.com/watch?v=4SK0fqEH6gY"},
        {"title": "Hurt", "artist": "Johnny Cash", "link": "https://www.youtube.com/watch?v=y8EHXf0b1YI"},
        {"title": "Someone Like You", "artist": "Adele", "link": "https://www.youtube.com/watch?v=xGPeNN9S0Fg"},
    ],
    "fear": [
        {"title": "Thriller", "artist": "Michael Jackson", "link": "https://www.youtube.com/watch?v=sOnqJvDwiLam"},
        {"title": "Psycho Killer", "artist": "Talking Heads", "link": "https://www.youtube.com/watch?v=O52jAYa4Pm8"},
        {"title": "Running Up That Hill", "artist": "Kate Bush", "link": "https://www.youtube.com/watch?v=wp43OdtAAkM"},
    ],
    "joy": [
        {"title": "Happy", "artist": "Pharrell Williams", "link": "https://www.youtube.com/watch?v=ZbZSe6N_CkY"},
        {"title": "Walking on Sunshine", "artist": "Katrina & The Waves", "link": "https://www.youtube.com/watch?v=iPUmE-tne5U"},
        {"title": "September", "artist": "Earth, Wind & Fire", "link": "https://www.youtube.com/watch?v=Gs069dNDdQY"},
    ],
    "anger": [
        {"title": "Killing in the Name", "artist": "Rage Against the Machine", "link": "https://www.youtube.com/watch?v=bWXazVhlyxQ"},
        {"title": "Break Stuff", "artist": "Limp Bizkit", "link": "https://www.youtube.com/watch?v=ZpUYjpkg9KY"},
        {"title": "Seven Nation Army", "artist": "The White Stripes", "link": "https://www.youtube.com/watch?v=RdBSkqOBkkY"},
    ],
    "love": [
        {"title": "Can't Help Falling in Love", "artist": "Elvis Presley", "link": "https://www.youtube.com/watch?v=vGJTaP6anOU"},
        {"title": "Wonderful World", "artist": "Louis Armstrong", "link": "https://www.youtube.com/watch?v=E2VCwBzGdPM"},
        {"title": "I Will Always Love You", "artist": "Whitney Houston", "link": "https://www.youtube.com/watch?v=3JWTFyD7Cag"},
    ],
    "surprise": [
        {"title": "Don't Stop Me Now", "artist": "Queen", "link": "https://www.youtube.com/watch?v=HgzGwKwLmgM"},
        {"title": "Bohemian Rhapsody", "artist": "Queen", "link": "https://www.youtube.com/watch?v=fJ9rUzIMcZQ"},
        {"title": "Mr. Brightside", "artist": "The Killers", "link": "https://www.youtube.com/watch?v=gGdGFtwCNBE"},
    ],
    "neutral": [
        {"title": "Atmosphere", "artist": "VNV Nation", "link": "https://www.youtube.com/watch?v=cQWnwsvRf8Y"},
        {"title": "Nuvole Bianche", "artist": "Ludovico Einaudi", "link": "https://www.youtube.com/watch?v=LD_kkKQg-Yo"},
        {"title": "Weightless", "artist": "Marconi Union", "link": "https://www.youtube.com/watch?v=UfcAVejslrU"},
    ]
}

# Load models with caching
@st.cache_resource
def load_models():
    model = joblib.load('model/model.pkl')
    vectorizer = joblib.load('model/tokenizer.pkl')
    emotion_classifier = pipeline("text-classification",
                                model="finiteautomata/bertweet-base-emotion-analysis",
                                top_k=None)
    return model, vectorizer, emotion_classifier

model, vectorizer, emotion_classifier = load_models()

# Physical/health state detection
def detect_physical_state(text):
    physical_keywords = {
        # Health-related terms
        'sick': 'sadness',
        'nauseous': 'sadness',
        'fever': 'sadness',
        'dizzy': 'fear',
        'headache': 'sadness',
        'unwell': 'sadness',

        # Physical harm terms
        'fell': 'fear',
        'pain': 'sadness',
        'hurt': 'sadness',
        'injured': 'fear',
        'bleeding': 'fear'
    }

    text_lower = text.lower()

    # Specific checks for negative statements indicating sadness
    if "not happy" in text_lower:
        return "sadness"
    if "did not do good" in text_lower:
        return "sadness"
    if "did not do well" in text_lower:
        return "sadness"
    if "feeling terrible" in text_lower:
        return "sadness"
    if "awful day" in text_lower:
        return "sadness"
    if "miserable" in text_lower:
        return "sadness"
    if "upset" in text_lower:
        return "sadness"
    if "disappointed" in text_lower:
        return "sadness"
    if "heartbroken" in text_lower:
        return "sadness"
    if "grief" in text_lower:
        return "sadness"
    if "sorrow" in text_lower:
        return "sadness"
    if "lonely" in text_lower:
        return "sadness"
    if "down in the dumps" in text_lower:
        return "sadness"
    if "blue" in text_lower:
        return "sadness"
    if "not feeling good" in text_lower:
        return "sadness"
    if "wish i felt better" in text_lower:
        return "sadness"
    if "heavy heart" in text_lower:
        return "sadness"

    for keyword, emotion in physical_keywords.items():
        if re.search(rf'\b{keyword}\b', text_lower):
            return emotion
    return None

# Enhanced emotion prediction
def predict_emotion(text):
    # First check for physical/health states
    physical_emotion = detect_physical_state(text)
    if physical_emotion:
        return physical_emotion

    # Original prediction logic
    vect_text = vectorizer.transform([text])
    ml_prediction = model.predict(vect_text)[0]

    # Confidence check
    ml_proba = model.predict_proba(vect_text)[0]
    if np.max(ml_proba) < 0.6:  # Low confidence
        transformer_result = emotion_classifier(text)[0]
        return max(transformer_result, key=lambda x: x['score'])['label']

    return ml_prediction

# Emotion-specific responses
emotion_responses = {
    "sadness": {
        "message": "💙 I sense you're feeling down.",
        "icon": "😔",
        "advice": "It's okay not to be okay. Remember, every setback is a setup for a comeback. Be kind to yourself.",
        "color": "#ADD8E6",  # Light blue
        "health_tip": True
    },
    "fear": {
        "message": "⚠️ It seems you're feeling worried or scared.",
        "icon": "😟",
        "advice": "The cave you fear to enter holds the treasure you seek. Take courage.",
        "color": "#FFCCCB",  # Light red
        "health_tip": True
    },
    "joy": {
        "message": "😊 That's wonderful! You seem happy!",
        "icon": "😄",
        "advice": "The most wasted of all days is one without laughter. Keep smiling!",
        "color": "#90EE90",  # Light green
        "health_tip": False
    },
    "anger": {
        "message": "😠 I detect feelings of frustration or anger.",
        "icon": "😡",
        "advice": "For every minute you remain angry, you give up sixty seconds of peace of mind. Let it go.",
        "color": "#FFA07A",  # Light salmon
        "health_tip": False
    },
    "love": {
        "message": "💖 You're feeling love! That's beautiful.",
        "icon": "😍",
        "advice": "The only way to do great work is to love what you do. Cherish those you love.",
        "color": "#FFB6C1",  # Light pink
        "health_tip": False
    },
    "surprise": {
        "message": "😮 Something unexpected happened!",
        "icon": "😲",
        "advice": "Expect the unexpected, and whenever possible, be the unexpected. Embrace the novelty!",
        "color": "#FAF0E6",  # Light beige
        "health_tip": False
    },
    "neutral": {
        "message": "😶 You seem to be in a neutral state.",
        "icon": "😐",
        "advice": "In the middle of movement and chaos, keep stillness inside of you. Find your center.",
        "color": "#D3D3D3",  # Light gray
        "health_tip": False
    }
}

# UI Implementation
st.title("🧠 Emotion Classifier + 🎵 Mood-Based Song Recommender")
user_input = st.text_area("Enter your text here:", placeholder="How are you feeling today?")
recommend_songs = st.toggle("Recommend Songs?")

if st.button("Predict Emotion"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner('Analyzing your emotions...'):
            emotion = predict_emotion(user_input.lower())

        response = emotion_responses.get(emotion.lower(), emotion_responses.get("neutral"))

        st.markdown(f"""
            <div style='background-color:{response["color"]};padding:15px;border-radius:10px'>
                <h3>{response["icon"]} {response["message"]}</h3>
                <p>{response["advice"]}</p>
                """, unsafe_allow_html=True)

        if response.get("health_tip"):
            st.markdown("""
                <details>
                    <summary>🩹 Health Tips</summary>
                    <ul>
                        <li>Drink plenty of fluids</li>
                        <li>Rest in a comfortable position</li>
                        <li>Monitor your symptoms</li>
                    </ul>
                </details>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("</div>", unsafe_allow_html=True)

        st.success(f"Detected Emotion: {emotion.upper()}")

        if recommend_songs:
            recommended_tracks = song_recommendations_with_links.get(emotion.lower(), [])
            st.subheader("Recommended Songs:")
            if recommended_tracks:
                for song_info in random.sample(recommended_tracks, min(3, len(recommended_tracks))):
                    st.markdown(f"- [{song_info['title']} - {song_info['artist']}]({song_info['link']})")
            else:
                st.markdown("- No songs found for this mood.")