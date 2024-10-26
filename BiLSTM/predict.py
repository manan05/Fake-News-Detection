import os
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import nltk

# Ensure nltk packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Paths to saved artifacts
SAVE_DIR = './saved_models'
MODEL_PATH = f"{SAVE_DIR}/final_model.h5"
TOKENIZER_PATH = f"{SAVE_DIR}/tokenizer.pkl"
SPEAKER_ENCODER_PATH = f"{SAVE_DIR}/speaker_encoder.pkl"
SUBJECT_ENCODER_PATH = f"{SAVE_DIR}/subject_encoder.pkl"

# Preprocess text data
def preprocess_text(text: str, stemmer: nltk.stem.PorterStemmer, stopwords_set: set) -> str:
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = nltk.word_tokenize(text.lower())
    processed_words = [stemmer.stem(word) for word in words if word not in stopwords_set]
    return ' '.join(processed_words)

# Load saved artifacts
def load_artifacts():
    # Debug: Check if files exist
    if os.path.exists(MODEL_PATH):
        print("Model file found.")
    else:
        print(f"Model file NOT found at {MODEL_PATH}. Please check the path.")
        return None, None, None, None

    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(SPEAKER_ENCODER_PATH, 'rb') as f:
        speaker_encoder = pickle.load(f)
    with open(SUBJECT_ENCODER_PATH, 'rb') as f:
        subject_encoder = pickle.load(f)

    # Load model
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None

    return model, tokenizer, speaker_encoder, subject_encoder

# Prediction function
def predict_statement(statement: str, speaker: str, subject: str):
    model, tokenizer, speaker_encoder, subject_encoder = load_artifacts()
    if not model:
        return

    stemmer = nltk.stem.PorterStemmer()
    stopwords_set = set(nltk.corpus.stopwords.words('english'))

    # Preprocess inputs
    statement_processed = preprocess_text(statement, stemmer, stopwords_set)
    sequence = tokenizer.texts_to_sequences([statement_processed])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')

    # Encode speaker and subject with 'unknown' handling
    def encode_feature(feature, encoder):
        mapping = {label: idx for idx, label in enumerate(encoder.classes_)}
        unknown_idx = len(encoder.classes_)
        return np.array([mapping.get(feature, unknown_idx)])

    speaker_encoded = encode_feature(speaker, speaker_encoder)
    subject_encoded = encode_feature(subject, subject_encoder)

    # Display input shapes for debugging
    print("Input shapes for model prediction:")
    print(f"text_input (padded_sequence): {padded_sequence.shape}")
    print(f"speaker_input: {speaker_encoded.shape}")
    print(f"subject_input: {subject_encoded.shape}")

    # Prepare inputs
    inputs = {
        'text_input': padded_sequence,
        'speaker_input': speaker_encoded,
        'subject_input': subject_encoded
    }

    # Make prediction
    try:
        prediction = model.predict(inputs)
        predicted_label = (prediction > 0.5).astype(int)[0][0]
        confidence = prediction[0][0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # Interpret and display results
    label = 'Not Fake' if predicted_label == 1 else 'Fake'
    print(f"Statement: {statement}")
    print(f"Speaker: {speaker}")
    print(f"Subject: {subject}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")

# Example usage
if __name__ == "__main__":
    test_statement = "A \"study showed as many as one in four people have had a package stolen from their residence."
    test_speaker = "rob hutton"
    test_subject = "consumer safety;criminal justice;legal issues;crime"
    predict_statement(test_statement, test_speaker, test_subject)
