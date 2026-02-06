import re
import numpy as np

# ------------------ Emotional Language ------------------
def emotion_score(text):
    emotional_words = [
        "shocking", "breaking", "unbelievable",
        "exposed", "truth", "must see", "you won’t believe"
    ]
    text = text.lower()
    word_count = len(text.split()) + 1

    score = sum(1 for word in emotional_words if word in text)
    return score / word_count


# ------------------ Capitalization Pattern ------------------
def caps_ratio(text):
    if not text:
        return 0.0
    return sum(1 for c in text if c.isupper()) / len(text)


# ------------------ Sentence Structure ------------------
def avg_sentence_length(text):
    sentences = re.split(r"[.!?]", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    return sum(len(s.split()) for s in sentences) / len(sentences)


# ------------------ Word Complexity ------------------
def avg_word_length(text):
    words = re.findall(r"\b[a-zA-Z]+\b", text)

    if not words:
        return 0.0

    return sum(len(word) for word in words) / len(words)


# ------------------ Readability Proxy ------------------
def readability_score(text):
    return avg_sentence_length(text) * avg_word_length(text)


# ------------------ Feature Vector ------------------
def extract_features(text):
    """
    Returns numeric features as a 2D numpy array
    Shape: (1, 6)
    """

    features = [
        emotion_score(text),        # Emotional intensity
        0.0,                        # Clickbait placeholder (future upgrade)
        caps_ratio(text),           # Capitalization behavior
        avg_sentence_length(text),  # Sentence complexity
        avg_word_length(text),      # Vocabulary complexity
        readability_score(text)     # Writing quality proxy
    ]

    return np.array([features])
