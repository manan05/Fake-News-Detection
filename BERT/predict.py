# predict.py

import tensorflow as tf
from transformers import BertTokenizer
from tensorflow.keras.models import load_model
import numpy as np

from utils import (
    preprocess_text,
    load_object
)
import os

def predict_statement(statement, speaker, subject):
    # Paths
    MODEL_DIR = './models'
    SAVED_OBJECTS_DIR = './saved_objects'
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'bert_model.h5')
    TOKENIZER_SAVE_PATH = os.path.join(SAVED_OBJECTS_DIR, 'tokenizer.pkl')
    SPEAKER_ENCODER_PATH = os.path.join(SAVED_OBJECTS_DIR, 'speaker_encoder.pkl')
    SUBJECT_ENCODER_PATH = os.path.join(SAVED_OBJECTS_DIR, 'subject_encoder.pkl')

    # Load tokenizer and encoders
    tokenizer = load_object(TOKENIZER_SAVE_PATH)
    speaker_encoder = load_object(SPEAKER_ENCODER_PATH)
    subject_encoder = load_object(SUBJECT_ENCODER_PATH)

    # Load model
    model = load_model(MODEL_SAVE_PATH, custom_objects={'TFBertModel': tf.keras.Model})

    # Preprocess inputs
    statement = preprocess_text(statement)

    encoding = tokenizer(
        [statement],
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='tf'
    )

    # Encode speaker and subject
    speaker_classes = list(speaker_encoder.classes_)
    subject_classes = list(subject_encoder.classes_)
    num_speakers = len(speaker_classes) + 1
    num_subjects = len(subject_classes) + 1

    def encode_feature(feature, encoder, classes):
        mapping = {label: idx for idx, label in enumerate(classes)}
        unknown_idx = len(classes)
        return np.array([mapping.get(feature, unknown_idx)])

    speaker_encoded = encode_feature(speaker, speaker_encoder, speaker_classes)
    subject_encoded = encode_feature(subject, subject_encoder, subject_classes)

    # Prepare inputs
    inputs = {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'speaker_input': speaker_encoded,
        'subject_input': subject_encoded
    }

    # Make prediction
    prediction = model.predict(inputs)
    predicted_label = (prediction > 0.5).astype(int)[0][0]
    confidence = prediction[0][0]

    label = 'Not Fake' if predicted_label == 1 else 'Fake'

    print(f"Statement: {statement}")
    print(f"Speaker: {speaker}")
    print(f"Subject: {subject}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == '__main__':
    # Example usage
    test_statement = "The economy is stronger than ever."
    test_speaker = "Donald Trump"
    test_subject = "economy"

    predict_statement(test_statement, test_speaker, test_subject)
