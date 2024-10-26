# predict.py

import tensorflow as tf
from transformers import BertTokenizer, TFDistilBertModel  # Import TFDistilBertModel
from tensorflow.keras.models import load_model
import numpy as np

from utils import (
    preprocess_text,
    load_object
)
import os

def predict_statement(statement, speaker, subject):
    # Paths
    MODEL_DIR = 'models'
    SAVED_OBJECTS_DIR = './saved_objects'
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'distilbert_model.h5')
    TOKENIZER_SAVE_PATH = os.path.join(SAVED_OBJECTS_DIR, 'tokenizer.pkl')
    SPEAKER_ENCODER_PATH = os.path.join(SAVED_OBJECTS_DIR, 'speaker_encoder.pkl')
    SUBJECT_ENCODER_PATH = os.path.join(SAVED_OBJECTS_DIR, 'subject_encoder.pkl')

    # Debug: Check the current working directory
    print("Current Working Directory:", os.getcwd())
    
    # Debug: Check if the model path exists
    if os.path.exists(MODEL_SAVE_PATH):
        print("Model file found.")
    else:
        print(f"Model file NOT found at {MODEL_SAVE_PATH}. Please check the path.")
        return

    # Load tokenizer and encoders
    try:
        tokenizer = load_object(TOKENIZER_SAVE_PATH)
        speaker_encoder = load_object(SPEAKER_ENCODER_PATH)
        subject_encoder = load_object(SUBJECT_ENCODER_PATH)
    except Exception as e:
        print(f"Error loading tokenizer or encoders: {e}")
        return

    # Load model with correct custom_objects mapping
    try:
        model = load_model(
            MODEL_SAVE_PATH,
            custom_objects={'TFDistilBertModel': TFDistilBertModel}  # Use TFDistilBertModel from transformers
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

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

    # Debug: Display the input shapes for model prediction
    print("Input shapes for model prediction:")
    print(f"input_ids: {inputs['input_ids'].shape}")
    print(f"attention_mask: {inputs['attention_mask'].shape}")
    print(f"speaker_input: {speaker_encoded.shape}")
    print(f"subject_input: {subject_encoded.shape}")

    # Make prediction
    try:
        prediction = model.predict(inputs)
        predicted_label = (prediction > 0.5).astype(int)[0][0]
        confidence = prediction[0][0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    label = 'Not Fake' if predicted_label == 1 else 'Fake'

    # Display the results
    print(f"Statement: {statement}")
    print(f"Speaker: {speaker}")
    print(f"Subject: {subject}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == '__main__':
    # Example usage
    test_statement = "A \"study showed as many as one in four people have had a package stolen from their residence."
    test_speaker = "rob hutton"
    test_subject = "consumer safety;criminal justice;legal issues;crime"

    predict_statement(test_statement, test_speaker, test_subject)
