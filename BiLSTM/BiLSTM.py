import re
import pickle
from typing import Tuple

import nltk
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (Bidirectional, Dense, Dropout, Embedding,
                                     LSTM, Input, Concatenate, Flatten)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure nltk packages are downloaded
def download_nltk_dependencies() -> None:
    """Download required NLTK packages."""
    nltk.download('punkt')
    nltk.download('stopwords')

# Preprocess text data
def preprocess_text(
    text: str,
    stemmer: nltk.stem.PorterStemmer,
    stopwords_set: set
) -> str:
    
    # Remove non-alphabet characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize and convert to lowercase
    words = nltk.word_tokenize(text.lower())
    # Stem words and remove stopwords
    processed_words = [
        stemmer.stem(word) for word in words if word not in stopwords_set
    ]
    return ' '.join(processed_words)

# Load and map labels
def load_and_map_labels(
    train_path: str,
    val_path: str,
    test_path: str,
    label_map: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    train_df = pd.read_csv(train_path, encoding='utf-8')
    val_df = pd.read_csv(val_path, encoding='utf-8')
    test_df = pd.read_csv(test_path, encoding='utf-8')

    for df in [train_df, val_df, test_df]:
        df['label'] = df['label'].map(label_map)

    return train_df, val_df, test_df

# Custom encoding function to handle unknown labels
def encode_with_unknown(labels, label_encoder):
    
    classes = label_encoder.classes_
    mapping = {label: idx for idx, label in enumerate(classes)}
    unknown_idx = len(classes)  # Assign the next index for unknowns
    encoded = [mapping.get(label, unknown_idx) for label in labels]
    return encoded

# Build the model with additional features
def build_model(input_dim: int, output_dim: int, input_length: int, num_speakers: int, num_subjects: int) -> Model:
    # Text input
    text_input = Input(shape=(input_length,), name='text_input')
    embedding_layer = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length)(text_input)
    lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
    lstm_layer = Dropout(0.5)(lstm_layer)
    lstm_layer = Bidirectional(LSTM(32))(lstm_layer)
    lstm_output = Dense(32, activation='relu')(lstm_layer)
    lstm_output = Dropout(0.5)(lstm_output)

    # Speaker input
    speaker_input = Input(shape=(1,), name='speaker_input')
    speaker_embedding = Embedding(input_dim=num_speakers, output_dim=16, input_length=1)(speaker_input)
    speaker_embedding = Flatten()(speaker_embedding)

    # Subject input
    subject_input = Input(shape=(1,), name='subject_input')
    subject_embedding = Embedding(input_dim=num_subjects, output_dim=16, input_length=1)(subject_input)
    subject_embedding = Flatten()(subject_embedding)

    # Combine all features
    combined = Concatenate()([lstm_output, speaker_embedding, subject_embedding])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(1, activation='sigmoid', dtype='float32')(combined)

    model = Model(inputs=[text_input, speaker_input, subject_input], outputs=output)
    return model

def main() -> None:
    # Define directories for saving models and artifacts
    SAVE_DIR = './saved_models'
    TOKENIZER_PATH = os.path.join(SAVE_DIR, 'tokenizer.pkl')
    SPEAKER_ENCODER_PATH = os.path.join(SAVE_DIR, 'speaker_encoder.pkl')
    SUBJECT_ENCODER_PATH = os.path.join(SAVE_DIR, 'subject_encoder.pkl')
    LABEL_MAP_PATH = os.path.join(SAVE_DIR, 'label_map.pkl')
    MODEL_PATH = os.path.join(SAVE_DIR, 'final_model.h5')

    # Create directories if they don't exist
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Download necessary NLTK data
    download_nltk_dependencies()

    # Set mixed precision policy for GPU performance optimization
    set_global_policy('mixed_float16')

    # Verify GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Num GPUs Available: {len(gpus)}")
    if gpus:
        # Set memory growth to prevent TensorFlow from allocating all GPU memory at once
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPUs detected. Exiting...")
        return

    # Initialize stemmer and stopwords
    stemmer = nltk.stem.PorterStemmer()
    stopwords_set = set(nltk.corpus.stopwords.words('english'))

    # Load datasets with label mapping
    label_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    data_paths = ['../data/train.csv', '../data/valid.csv', '../data/test.csv']
    train_df, val_df, test_df = load_and_map_labels(
        data_paths[0], data_paths[1], data_paths[2],
        label_map
    )

    # Save label_map for future use
    with open(LABEL_MAP_PATH, 'wb') as f:
        pickle.dump(label_map, f)
    print(f"Label map saved to {LABEL_MAP_PATH}")

    # Preprocess the 'statement' column in each DataFrame
    for df in [train_df, val_df, test_df]:
        df['statement'] = df['statement'].astype(str).apply(
            lambda x: preprocess_text(x, stemmer, stopwords_set)
        )

    # Tokenization
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df['statement'])

    # Save the tokenizer for future use
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {TOKENIZER_PATH}")

    # Convert text to sequences
    X_train_text = tokenizer.texts_to_sequences(train_df['statement'])
    X_val_text = tokenizer.texts_to_sequences(val_df['statement'])
    X_test_text = tokenizer.texts_to_sequences(test_df['statement'])

    # Pad sequences
    max_length = 100
    X_train_text_padded = pad_sequences(X_train_text, maxlen=max_length, padding='post')
    X_val_text_padded = pad_sequences(X_val_text, maxlen=max_length, padding='post')
    X_test_text_padded = pad_sequences(X_test_text, maxlen=max_length, padding='post')

    # Encode 'speaker' with handling of unknown labels
    speaker_encoder = LabelEncoder()
    speaker_encoder.fit(train_df['speaker'])
    
    # Save speaker encoder classes
    speaker_classes = list(speaker_encoder.classes_)
    num_speakers = len(speaker_classes) + 1  # +1 for 'unknown'
    
    # Encode speakers with 'unknown' handling
    def encode_speaker(speakers, encoder, classes, unknown_label='unknown'):
        mapping = {label: idx for idx, label in enumerate(classes)}
        unknown_idx = len(classes)
        encoded = [mapping.get(label, unknown_idx) for label in speakers]
        return encoded

    train_df['speaker_encoded'] = encode_speaker(train_df['speaker'], speaker_encoder, speaker_classes)
    val_df['speaker_encoded'] = encode_speaker(val_df['speaker'], speaker_encoder, speaker_classes)
    test_df['speaker_encoded'] = encode_speaker(test_df['speaker'], speaker_encoder, speaker_classes)

    # Save speaker encoder
    with open(SPEAKER_ENCODER_PATH, 'wb') as f:
        pickle.dump(speaker_encoder, f)
    print(f"Speaker encoder saved to {SPEAKER_ENCODER_PATH}")

    # Encode 'subject' with handling of unknown labels
    subject_encoder = LabelEncoder()
    subject_encoder.fit(train_df['subject'])
    
    # Save subject encoder classes
    subject_classes = list(subject_encoder.classes_)
    num_subjects = len(subject_classes) + 1  # +1 for 'unknown'
    
    # Encode subjects with 'unknown' handling
    def encode_subject(subjects, encoder, classes, unknown_label='unknown'):
        mapping = {label: idx for idx, label in enumerate(classes)}
        unknown_idx = len(classes)
        encoded = [mapping.get(label, unknown_idx) for label in subjects]
        return encoded

    train_df['subject_encoded'] = encode_subject(train_df['subject'], subject_encoder, subject_classes)
    val_df['subject_encoded'] = encode_subject(val_df['subject'], subject_encoder, subject_classes)
    test_df['subject_encoded'] = encode_subject(test_df['subject'], subject_encoder, subject_classes)

    # Save subject encoder
    with open(SUBJECT_ENCODER_PATH, 'wb') as f:
        pickle.dump(subject_encoder, f)
    print(f"Subject encoder saved to {SUBJECT_ENCODER_PATH}")

    # Extract labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # Build and compile the model within the GPU device context
    with tf.device('/GPU:0'):
        model = build_model(
            input_dim=10000,
            output_dim=64,
            input_length=max_length,
            num_speakers=num_speakers,
            num_subjects=num_subjects
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

    # Display model architecture
    model.summary()

    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )

    # ModelCheckpoint to save the best model during training
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,  # Set to False to save the entire model
        verbose=1
    )

    # Fit the model
    history = model.fit(
        {
            'text_input': X_train_text_padded,
            'speaker_input': train_df['speaker_encoded'],
            'subject_input': train_df['subject_encoded']
        },
        y_train,
        validation_data=( 
            {
                'text_input': X_val_text_padded,
                'speaker_input': val_df['speaker_encoded'],
                'subject_input': val_df['subject_encoded']
            },
            y_val
        ),
        epochs=40,
        batch_size=64,  # Adjust batch size based on GPU memory
        callbacks=[early_stopping, checkpoint],
        verbose=1  # Set to 1 to see progress
    )

    # Save the final model after training completes
    model.save(MODEL_PATH)
    print(f"Final model saved to {MODEL_PATH}")

    # Evaluate the model on test data
    loss, accuracy = model.evaluate(
        {
            'text_input': X_test_text_padded,
            'speaker_input': test_df['speaker_encoded'],
            'subject_input': test_df['subject_encoded']
        },
        y_test,
        verbose=1
    )
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    # Generate predictions for confusion matrix
    y_pred_prob = model.predict({
        'text_input': X_test_text_padded,
        'speaker_input': test_df['speaker_encoded'],
        'subject_input': test_df['subject_encoded']
    })

    # Convert probabilities to binary predictions
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Not Fake'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Define predict function
    def predict_statement(statement: str, speaker: str, subject: str) -> str:
        # Load the trained model and tokenizer
        model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        with open(SPEAKER_ENCODER_PATH, 'rb') as f:
            speaker_encoder = pickle.load(f)
        with open(SUBJECT_ENCODER_PATH, 'rb') as f:
            subject_encoder = pickle.load(f)

        # Preprocess the statement
        processed_statement = preprocess_text(statement, stemmer, stopwords_set)

        # Convert to sequence
        sequence = tokenizer.texts_to_sequences([processed_statement])

        # Pad the sequence
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

        # Encode speaker and subject with 'unknown' handling
        def encode_feature(feature, encoder, classes):
            mapping = {label: idx for idx, label in enumerate(classes)}
            unknown_idx = len(classes)
            return [mapping.get(feature, unknown_idx)]

        speaker_encoded = encode_feature(speaker, speaker_encoder, speaker_classes)
        subject_encoded = encode_feature(subject, subject_encoder, subject_classes)

        # Convert to numpy arrays
        speaker_encoded = np.array(speaker_encoded)
        subject_encoded = np.array(subject_encoded)

        # Make prediction
        prediction = model.predict({
            'text_input': padded_sequence,
            'speaker_input': speaker_encoded,
            'subject_input': subject_encoded
        })

        # Interpret the prediction
        predicted_label = (prediction > 0.5).astype(int)[0][0]
        return 'Fake' if predicted_label == 0 else 'Not Fake'

    # Example usage
    test_statement = "The economy is growing faster than ever."
    test_speaker = "Barack Obama"
    test_subject = "economy"
    prediction = predict_statement(test_statement, test_speaker, test_subject)
    print(f"Prediction for the statement: '{test_statement}' is '{prediction}'")

if __name__ == "__main__":
    main()
