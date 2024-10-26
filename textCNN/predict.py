# predict.py

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

from utils import (
    preprocess_text,
    load_object
)
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_statement(statement, speaker, subject):

    MODEL_DIR = './models'
    SAVED_OBJECTS_DIR = './saved_objects'
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'text_cnn_model.h5')
    TOKENIZER_SAVE_PATH = os.path.join(SAVED_OBJECTS_DIR, 'tokenizer.pkl')
    SPEAKER_ENCODER_PATH = os.path.join(SAVED_OBJECTS_DIR, 'speaker_encoder.pkl')
    SUBJECT_ENCODER_PATH = os.path.join(SAVED_OBJECTS_DIR, 'subject_encoder.pkl')
    LABEL_MAP_PATH = os.path.join(SAVED_OBJECTS_DIR, 'label_map.pkl')

    tokenizer = load_object(TOKENIZER_SAVE_PATH)
    speaker_encoder = load_object(SPEAKER_ENCODER_PATH)
    subject_encoder = load_object(SUBJECT_ENCODER_PATH)

    model = load_model(MODEL_SAVE_PATH)
    print(f"Model loaded from {MODEL_SAVE_PATH}")

    processed_statement = preprocess_text(statement)

    sequence = tokenizer.texts_to_sequences([processed_statement])
    max_length = 100
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    def encode_feature(feature, encoder, classes):
        mapping = {label: idx for idx, label in enumerate(classes)}
        unknown_idx = len(classes)
        return [mapping.get(feature, unknown_idx)]

    speaker_classes = list(speaker_encoder.classes_)
    subject_classes = list(subject_encoder.classes_)

    speaker_encoded = encode_feature(speaker, speaker_encoder, speaker_classes)
    subject_encoded = encode_feature(subject, subject_encoder, subject_classes)

    speaker_encoded = np.array(speaker_encoded)
    subject_encoded = np.array(subject_encoded)

    prediction = model.predict({
        'text_input': padded_sequence,
        'speaker_input': speaker_encoded,
        'subject_input': subject_encoded
    })

    predicted_label = (prediction > 0.5).astype(int)[0][0]
    confidence = prediction[0][0]
    label = 'Not Fake' if predicted_label == 1 else 'Fake'

    print(f"Statement: {statement}")
    print(f"Speaker: {speaker}")
    print(f"Subject: {subject}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == '__main__':
    test_statement = "The economy is stronger than ever."
    test_speaker = "Donald Trump"
    test_subject = "economy"

    predict_statement(test_statement, test_speaker, test_subject)
