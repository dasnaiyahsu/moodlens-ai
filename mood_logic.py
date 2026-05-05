import random

MOOD_DATA = {
    "happy": {
        "color": "#FFD700", # Warm Yellow
        "music": "Pop / Upbeat Indie (Coba dengerin 'Walking on Sunshine')",
        "quotes": [
            "Keep spreading that bright energy!",
            "Your smile is your best accessory today."
        ]
    },
    "joy": { # Untuk model BERT yang biasanya pakai kata "joy"
        "color": "#FFD700",
        "music": "Pop / Upbeat Indie",
        "quotes": [
            "Keep spreading that bright energy!",
            "Your smile is your best accessory today."
        ]
    },
    "sad": {
        "color": "#5DADE2", # Cool Blue
        "music": "Lo-fi Beats / Acoustic Chill",
        "quotes": [
            "It's okay to have rainy days. Take your time.",
            "Breathe. Tomorrow is a new blank page."
        ]
    },
    "sadness": { # Untuk model BERT
        "color": "#5DADE2",
        "music": "Lo-fi Beats / Acoustic Chill",
        "quotes": [
            "It's okay to have rainy days. Take your time.",
            "Breathe. Tomorrow is a new blank page."
        ]
    },
    "angry": {
        "color": "#E74C3C", # Calming Red
        "music": "Ambient / Classical Meditation",
        "quotes": [
            "Take a deep breath. Let the tension melt away.",
            "Peace comes from within. You are in control."
        ]
    },
    "anger": { # Untuk model BERT
        "color": "#E74C3C",
        "music": "Ambient / Classical Meditation",
        "quotes": [
            "Take a deep breath. Let the tension melt away.",
            "Peace comes from within. You are in control."
        ]
    },
    "neutral": {
        "color": "#E5E7E9", # Muted Grey
        "music": "Acoustic Pop / Coffee Shop Vibe",
        "quotes": [
            "A calm mind brings inner peace.",
            "Enjoy the stillness of this moment."
        ]
    }
}

def get_mood_recommendation(emotion):
    mood = emotion.lower()
    if mood in MOOD_DATA:
        data = MOOD_DATA[mood]
        return data["color"], data["music"], random.choice(data["quotes"])
    return "#FFFFFF", "Acoustic Pop", "How are you feeling today?"