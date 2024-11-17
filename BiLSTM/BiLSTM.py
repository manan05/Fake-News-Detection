import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import mixed_precision
import nltk
from nltk.corpus import stopwords# train.py

import os
import re
import pickle
from typing import Tuple

import nltk
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Bidirectional, Dense, Dropout, Embedding, LSTM, Input, Concatenate, Flatten
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Ensure nltk packages are downloaded
def download_nltk_dependencies() -> None:
    """Download required NLTK packages."""
    nltk.download('punkt')
    nltk.download('stopwords')

# Preprocess text data
def preprocess_text(text: str, stemmer: nltk.stem.PorterStemmer, stopwords_set: set) -> str:
    """Clean and preprocess text."""
    # Remove non-alphabet characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize and convert to lowercase
    words = nltk.word_tokenize(text.lower())
    # Remove stopwords and stem words
    processed_words = [stemmer.stem(word) for word in words if word not in stopwords_set]
    return ' '.join(processed_words)

# Load and map labels
def load_and_map_labels(train_path: str, val_path: str, test_path: str, label_map: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load datasets and map labels based on label_map."""
    train_df = pd.read_csv(train_path, encoding='utf-8')
    val_df = pd.read_csv(val_path, encoding='utf-8')
    test_df = pd.read_csv(test_path, encoding='utf-8')

    for df in [train_df, val_df, test_df]:
        df['label'] = df['label'].map(label_map)
        # Drop rows with missing essential fields
        df.dropna(subset=['statement', 'speaker', 'subject', 'label'], inplace=True)

    return train_df, val_df, test_df

# Custom encoding function to handle unknown labels
def encode_with_unknown(labels, encoder, classes):
    """Encode labels, assigning unknown labels to a specific index."""
    unknown_idx = len(classes)  # Assign the next index for unknowns
    encoded = labels.apply(lambda x: encoder.transform([x])[0] if x in classes else unknown_idx)
    return encoded

# Save a Python object to a file using pickle
def save_object(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

# Load a Python object from a pickle file
def load_object(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Create directories if they do not exist
def create_directories(directories):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Build the model with additional features
def build_model(input_dim: int, output_dim: int, input_length: int, num_speakers: int, num_subjects: int) -> Model:
    """Build a multi-input Keras model."""
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
    output = Dense(1, activation='sigmoid', dtype='float32')(combined)  # float32 for numerical stability

    model = Model(inputs=[text_input, speaker_input, subject_input], outputs=output)
    return model

# Evaluate the model and print metrics and confusion matrices
def evaluate_model(model, X_val, y_val, X_test, y_test):
    """Evaluate the model on validation and test sets."""
    print("\nEvaluating on Validation Set:")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc * 100:.2f}%")

    print("\nEvaluating on Test Set:")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc * 100:.2f}%")

    # Predictions
    print("\nGenerating Predictions for Validation Set...")
    val_preds_prob = model.predict(X_val, verbose=0)
    val_preds = (val_preds_prob > 0.5).astype(int).flatten()

    print("Generating Predictions for Test Set...")
    test_preds_prob = model.predict(X_test, verbose=0)
    test_preds = (test_preds_prob > 0.5).astype(int).flatten()

    # Confusion matrices
    print("\nConfusion Matrix for Validation Set:")
    val_cm = confusion_matrix(y_val, val_preds)
    print(val_cm)
    disp_val = ConfusionMatrixDisplay(confusion_matrix=val_cm, display_labels=['Fake', 'Not Fake'])
    disp_val.plot(cmap=plt.cm.Blues)
    plt.title('Validation Confusion Matrix')
    plt.show()

    print("\nConfusion Matrix for Test Set:")
    test_cm = confusion_matrix(y_test, test_preds)
    print(test_cm)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=test_cm, display_labels=['Fake', 'Not Fake'])
    disp_test.plot(cmap=plt.cm.Blues)
    plt.title('Test Confusion Matrix')
    plt.show()

    # Classification Reports
    print("\nClassification Report for Validation Set:")
    val_report = classification_report(y_val, val_preds, target_names=['Fake', 'Not Fake'])
    print(val_report)

    print("Classification Report for Test Set:")
    test_report = classification_report(y_test, test_preds, target_names=['Fake', 'Not Fake'])
    print(test_report)

def main():
    """Main function to train and evaluate the BiLSTM model."""
    # Define directories for saving models and artifacts
    DATA_DIR = '../data'
    SAVE_DIR = './saved_models'
    TOKENIZER_PATH = os.path.join(SAVE_DIR, 'tokenizer.pkl')
    SPEAKER_ENCODER_PATH = os.path.join(SAVE_DIR, 'speaker_encoder.pkl')
    SUBJECT_ENCODER_PATH = os.path.join(SAVE_DIR, 'subject_encoder.pkl')
    LABEL_MAP_PATH = os.path.join(SAVE_DIR, 'label_map.pkl')
    MODEL_DIR = os.path.join(SAVE_DIR, 'final_model')  # Using SavedModel format

    # Create necessary directories
    create_directories([SAVE_DIR, MODEL_DIR])

    # Download necessary NLTK dependencies
    download_nltk_dependencies()

    # Set mixed precision policy for GPU performance optimization
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f"Mixed precision policy set to: {policy}")

    # Verify GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Num GPUs Available: {len(gpus)}")
    if gpus:
        for gpu in gpus:
            print(f"GPU: {gpu.name}")
    else:
        print("No GPUs detected. Exiting...")
        return

    # Initialize stemmer and stopwords
    stemmer = nltk.stem.PorterStemmer()
    stopwords_set = set(nltk.corpus.stopwords.words('english'))

    # Define label mapping (Adjust as per your dataset)
    label_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    # Define file paths
    TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
    VAL_PATH = os.path.join(DATA_DIR, 'valid.csv')
    TEST_PATH = os.path.join(DATA_DIR, 'test.csv')

    # Check if data files exist
    for path in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        if not os.path.exists(path):
            print(f"Data file not found at {path}. Please check the path.")
            return

    # Load datasets with label mapping
    train_df, val_df, test_df = load_and_map_labels(TRAIN_PATH, VAL_PATH, TEST_PATH, label_map)

    # Save label_map for future use
    save_object(label_map, LABEL_MAP_PATH)
    print(f"Label map saved to {LABEL_MAP_PATH}")

    # Preprocess the 'statement' column in each DataFrame
    for df in [train_df, val_df, test_df]:
        df['statement'] = df['statement'].astype(str).apply(lambda x: preprocess_text(x, stemmer, stopwords_set))
        df['speaker'] = df['speaker'].astype(str)
        df['subject'] = df['subject'].astype(str)

    # Tokenization
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df['statement'])

    # Save the tokenizer for future use
    save_object(tokenizer, TOKENIZER_PATH)
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
    speaker_classes = list(speaker_encoder.classes_)
    num_speakers = len(speaker_classes) + 1  # +1 for 'unknown'

    train_df['speaker_encoded'] = encode_with_unknown(train_df['speaker'], speaker_encoder, speaker_classes)
    val_df['speaker_encoded'] = encode_with_unknown(val_df['speaker'], speaker_encoder, speaker_classes)
    test_df['speaker_encoded'] = encode_with_unknown(test_df['speaker'], speaker_encoder, speaker_classes)

    # Save speaker encoder
    save_object(speaker_encoder, SPEAKER_ENCODER_PATH)
    print(f"Speaker encoder saved to {SPEAKER_ENCODER_PATH}")

    # Encode 'subject' with handling of unknown labels
    subject_encoder = LabelEncoder()
    subject_encoder.fit(train_df['subject'])
    subject_classes = list(subject_encoder.classes_)
    num_subjects = len(subject_classes) + 1  # +1 for 'unknown'

    train_df['subject_encoded'] = encode_with_unknown(train_df['subject'], subject_encoder, subject_classes)
    val_df['subject_encoded'] = encode_with_unknown(val_df['subject'], subject_encoder, subject_classes)
    test_df['subject_encoded'] = encode_with_unknown(test_df['subject'], subject_encoder, subject_classes)

    # Save subject encoder
    save_object(subject_encoder, SUBJECT_ENCODER_PATH)
    print(f"Subject encoder saved to {SUBJECT_ENCODER_PATH}")

    # Extract labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # Define model parameters
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100
    input_length = max_length

    # Build and compile the model
    model = build_model(vocab_size, embedding_dim, input_length, num_speakers, num_subjects)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(MODEL_DIR, monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1)

    # Train the model
    history = model.fit(
        [X_train_text_padded, train_df['speaker_encoded'], train_df['subject_encoded']],
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(
            [X_val_text_padded, val_df['speaker_encoded'], val_df['subject_encoded']],
            y_val
        ),
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    # Save the final model after training completes
    model.save(MODEL_DIR)
    print(f"Final model saved to {MODEL_DIR}")

    # Evaluate the model on validation and test datasets
    evaluate_model(
        model,
        [X_val_text_padded, val_df['speaker_encoded'], val_df['subject_encoded']],
        y_val,
        [X_test_text_padded, test_df['speaker_encoded'], test_df['subject_encoded']],
        y_test
    )

    print("Training and evaluation completed successfully.")

def predict_statement(statement: str, speaker: str, subject: str) -> str:
    """Function to make predictions on new statements."""
    # Define directories for loading models and artifacts
    SAVE_DIR = './saved_models'
    MODEL_DIR = os.path.join(SAVE_DIR, 'final_model')  # SavedModel format
    TOKENIZER_PATH = os.path.join(SAVE_DIR, 'tokenizer.pkl')
    SPEAKER_ENCODER_PATH = os.path.join(SAVE_DIR, 'speaker_encoder.pkl')
    SUBJECT_ENCODER_PATH = os.path.join(SAVE_DIR, 'subject_encoder.pkl')
    LABEL_MAP_PATH = os.path.join(SAVE_DIR, 'label_map.pkl')

    # Load the trained model and tokenizer
    if not os.path.exists(MODEL_DIR):
        print(f"Model directory not found at {MODEL_DIR}. Please train the model first.")
        return
    model = load_model(MODEL_DIR)
    print("Model loaded successfully.")

    tokenizer = load_object(TOKENIZER_PATH)
    print("Tokenizer loaded successfully.")

    speaker_encoder = load_object(SPEAKER_ENCODER_PATH)
    print("Speaker encoder loaded successfully.")

    subject_encoder = load_object(SUBJECT_ENCODER_PATH)
    print("Subject encoder loaded successfully.")

    # Load label_map
    label_map = load_object(LABEL_MAP_PATH)
    inverse_label_map = {v: k for k, v in label_map.items()}

    # Initialize stemmer and stopwords
    stemmer = nltk.stem.PorterStemmer()
    stopwords_set = set(nltk.corpus.stopwords.words('english'))

    # Preprocess the statement
    processed_statement = preprocess_text(statement, stemmer, stopwords_set)

    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([processed_statement])

    # Pad the sequence
    max_length = 100
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    # Encode speaker and subject with 'unknown' handling
    def encode_feature(feature, encoder, classes):
        if feature in classes:
            return encoder.transform([feature])[0]
        else:
            return len(classes)  # 'unknown' index

    speaker_classes = list(speaker_encoder.classes_)
    speaker_encoded = encode_feature(speaker, speaker_encoder, speaker_classes)

    subject_classes = list(subject_encoder.classes_)
    subject_encoded = encode_feature(subject, subject_encoder, subject_classes)

    # Prepare inputs
    speaker_encoded = np.array([speaker_encoded])
    subject_encoded = np.array([subject_encoded])

    # Make prediction
    prediction_prob = model.predict({
        'text_input': padded_sequence,
        'speaker_input': speaker_encoded,
        'subject_input': subject_encoded
    })

    # Interpret the prediction
    predicted_label = (prediction_prob > 0.5).astype(int)[0][0]
    label = 'Not Fake' if predicted_label == 1 else 'Fake'
    confidence = prediction_prob[0][0] if predicted_label == 1 else 1 - prediction_prob[0][0]

    print(f"\nStatement: {statement}")
    print(f"Speaker: {speaker}")
    print(f"Subject: {subject}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")

    return label

if __name__ == "__main__":
    main()
# train.py

import os
import re
import pickle
from typing import Tuple

import nltk
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Bidirectional, Dense, Dropout, Embedding, LSTM, Input, Concatenate, Flatten
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Ensure nltk packages are downloaded
def download_nltk_dependencies() -> None:
    """Download required NLTK packages."""
    nltk.download('punkt')
    nltk.download('stopwords')

# Preprocess text data
def preprocess_text(text: str, stemmer: nltk.stem.PorterStemmer, stopwords_set: set) -> str:
    """Clean and preprocess text."""
    # Remove non-alphabet characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize and convert to lowercase
    words = nltk.word_tokenize(text.lower())
    # Remove stopwords and stem words
    processed_words = [stemmer.stem(word) for word in words if word not in stopwords_set]
    return ' '.join(processed_words)

# Load and map labels
def load_and_map_labels(train_path: str, val_path: str, test_path: str, label_map: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load datasets and map labels based on label_map."""
    train_df = pd.read_csv(train_path, encoding='utf-8')
    val_df = pd.read_csv(val_path, encoding='utf-8')
    test_df = pd.read_csv(test_path, encoding='utf-8')

    for df in [train_df, val_df, test_df]:
        df['label'] = df['label'].map(label_map)
        # Drop rows with missing essential fields
        df.dropna(subset=['statement', 'speaker', 'subject', 'label'], inplace=True)

    return train_df, val_df, test_df

# Custom encoding function to handle unknown labels
def encode_with_unknown(labels, encoder, classes):
    """Encode labels, assigning unknown labels to a specific index."""
    unknown_idx = len(classes)  # Assign the next index for unknowns
    encoded = labels.apply(lambda x: encoder.transform([x])[0] if x in classes else unknown_idx)
    return encoded

# Save a Python object to a file using pickle
def save_object(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

# Load a Python object from a pickle file
def load_object(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Create directories if they do not exist
def create_directories(directories):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Build the model with additional features
def build_model(input_dim: int, output_dim: int, input_length: int, num_speakers: int, num_subjects: int) -> Model:
    """Build a multi-input Keras model."""
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
    output = Dense(1, activation='sigmoid', dtype='float32')(combined)  # float32 for numerical stability

    model = Model(inputs=[text_input, speaker_input, subject_input], outputs=output)
    return model

# Evaluate the model and print metrics and confusion matrices
def evaluate_model(model, X_val, y_val, X_test, y_test):
    """Evaluate the model on validation and test sets."""
    print("\nEvaluating on Validation Set:")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc * 100:.2f}%")

    print("\nEvaluating on Test Set:")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc * 100:.2f}%")

    # Predictions
    print("\nGenerating Predictions for Validation Set...")
    val_preds_prob = model.predict(X_val, verbose=0)
    val_preds = (val_preds_prob > 0.5).astype(int).flatten()

    print("Generating Predictions for Test Set...")
    test_preds_prob = model.predict(X_test, verbose=0)
    test_preds = (test_preds_prob > 0.5).astype(int).flatten()

    # Confusion matrices
    print("\nConfusion Matrix for Validation Set:")
    val_cm = confusion_matrix(y_val, val_preds)
    print(val_cm)
    disp_val = ConfusionMatrixDisplay(confusion_matrix=val_cm, display_labels=['Fake', 'Not Fake'])
    disp_val.plot(cmap=plt.cm.Blues)
    plt.title('Validation Confusion Matrix')
    plt.show()

    print("\nConfusion Matrix for Test Set:")
    test_cm = confusion_matrix(y_test, test_preds)
    print(test_cm)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=test_cm, display_labels=['Fake', 'Not Fake'])
    disp_test.plot(cmap=plt.cm.Blues)
    plt.title('Test Confusion Matrix')
    plt.show()

    # Classification Reports
    print("\nClassification Report for Validation Set:")
    val_report = classification_report(y_val, val_preds, target_names=['Fake', 'Not Fake'])
    print(val_report)

    print("Classification Report for Test Set:")
    test_report = classification_report(y_test, test_preds, target_names=['Fake', 'Not Fake'])
    print(test_report)

def main():
    """Main function to train and evaluate the BiLSTM model."""
    # Define directories for saving models and artifacts
    DATA_DIR = '../data'
    SAVE_DIR = './saved_models'
    TOKENIZER_PATH = os.path.join(SAVE_DIR, 'tokenizer.pkl')
    SPEAKER_ENCODER_PATH = os.path.join(SAVE_DIR, 'speaker_encoder.pkl')
    SUBJECT_ENCODER_PATH = os.path.join(SAVE_DIR, 'subject_encoder.pkl')
    LABEL_MAP_PATH = os.path.join(SAVE_DIR, 'label_map.pkl')
    MODEL_DIR = os.path.join(SAVE_DIR, 'final_model')  # Using SavedModel format

    # Create necessary directories
    create_directories([SAVE_DIR])

    # Download necessary NLTK dependencies
    download_nltk_dependencies()

    # Set mixed precision policy for GPU performance optimization
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f"Mixed precision policy set to: {policy}")

    # Verify GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Num GPUs Available: {len(gpus)}")
    if gpus:
        for gpu in gpus:
            print(f"GPU: {gpu.name}")
    else:
        print("No GPUs detected. Exiting...")
        return

    # Initialize stemmer and stopwords
    stemmer = nltk.stem.PorterStemmer()
    stopwords_set = set(nltk.corpus.stopwords.words('english'))

    # Define label mapping (Adjust as per your dataset)
    label_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    # Define file paths
    TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
    VAL_PATH = os.path.join(DATA_DIR, 'valid.csv')
    TEST_PATH = os.path.join(DATA_DIR, 'test.csv')

    # Check if data files exist
    for path in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        if not os.path.exists(path):
            print(f"Data file not found at {path}. Please check the path.")
            return

    # Load datasets with label mapping
    train_df, val_df, test_df = load_and_map_labels(TRAIN_PATH, VAL_PATH, TEST_PATH, label_map)

    # Save label_map for future use
    save_object(label_map, LABEL_MAP_PATH)
    print(f"Label map saved to {LABEL_MAP_PATH}")

    # Preprocess the 'statement' column in each DataFrame
    for df in [train_df, val_df, test_df]:
        df['statement'] = df['statement'].astype(str).apply(lambda x: preprocess_text(x, stemmer, stopwords_set))
        df['speaker'] = df['speaker'].astype(str)
        df['subject'] = df['subject'].astype(str)

    # Tokenization
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df['statement'])

    # Save the tokenizer for future use
    save_object(tokenizer, TOKENIZER_PATH)
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
    speaker_classes = list(speaker_encoder.classes_)
    num_speakers = len(speaker_classes) + 1  # +1 for 'unknown'

    train_df['speaker_encoded'] = encode_with_unknown(train_df['speaker'], speaker_encoder, speaker_classes)
    val_df['speaker_encoded'] = encode_with_unknown(val_df['speaker'], speaker_encoder, speaker_classes)
    test_df['speaker_encoded'] = encode_with_unknown(test_df['speaker'], speaker_encoder, speaker_classes)

    # Save speaker encoder
    save_object(speaker_encoder, SPEAKER_ENCODER_PATH)
    print(f"Speaker encoder saved to {SPEAKER_ENCODER_PATH}")

    # Encode 'subject' with handling of unknown labels
    subject_encoder = LabelEncoder()
    subject_encoder.fit(train_df['subject'])
    subject_classes = list(subject_encoder.classes_)
    num_subjects = len(subject_classes) + 1  # +1 for 'unknown'

    train_df['subject_encoded'] = encode_with_unknown(train_df['subject'], subject_encoder, subject_classes)
    val_df['subject_encoded'] = encode_with_unknown(val_df['subject'], subject_encoder, subject_classes)
    test_df['subject_encoded'] = encode_with_unknown(test_df['subject'], subject_encoder, subject_classes)

    # Save subject encoder
    save_object(subject_encoder, SUBJECT_ENCODER_PATH)
    print(f"Subject encoder saved to {SUBJECT_ENCODER_PATH}")

    # Extract labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # Define model parameters
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100
    input_length = max_length

    # Build and compile the model
    model = build_model(vocab_size, embedding_dim, input_length, num_speakers, num_subjects)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(MODEL_DIR, monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1)

    # Train the model
    history = model.fit(
        [X_train_text_padded, train_df['speaker_encoded'], train_df['subject_encoded']],
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(
            [X_val_text_padded, val_df['speaker_encoded'], val_df['subject_encoded']],
            y_val
        ),
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    # Save the final model after training completes
    model.save(MODEL_DIR)
    print(f"Final model saved to {MODEL_DIR}")

    # Evaluate the model on validation and test datasets
    evaluate_model(
        model,
        [X_val_text_padded, val_df['speaker_encoded'], val_df['subject_encoded']],
        y_val,
        [X_test_text_padded, test_df['speaker_encoded'], test_df['subject_encoded']],
        y_test
    )

    print("Training and evaluation completed successfully.")

def predict_statement(statement: str, speaker: str, subject: str) -> str:
    """Function to make predictions on new statements."""
    # Define directories for loading models and artifacts
    SAVE_DIR = './saved_models'
    MODEL_DIR = os.path.join(SAVE_DIR, 'final_model')  # SavedModel format
    TOKENIZER_PATH = os.path.join(SAVE_DIR, 'tokenizer.pkl')
    SPEAKER_ENCODER_PATH = os.path.join(SAVE_DIR, 'speaker_encoder.pkl')
    SUBJECT_ENCODER_PATH = os.path.join(SAVE_DIR, 'subject_encoder.pkl')
    LABEL_MAP_PATH = os.path.join(SAVE_DIR, 'label_map.pkl')

    # Load the trained model and tokenizer
    if not os.path.exists(MODEL_DIR):
        print(f"Model directory not found at {MODEL_DIR}. Please train the model first.")
        return
    model = load_model(MODEL_DIR)
    print("Model loaded successfully.")

    tokenizer = load_object(TOKENIZER_PATH)
    print("Tokenizer loaded successfully.")

    speaker_encoder = load_object(SPEAKER_ENCODER_PATH)
    print("Speaker encoder loaded successfully.")

    subject_encoder = load_object(SUBJECT_ENCODER_PATH)
    print("Subject encoder loaded successfully.")

    # Load label_map
    label_map = load_object(LABEL_MAP_PATH)
    inverse_label_map = {v: k for k, v in label_map.items()}

    # Initialize stemmer and stopwords
    stemmer = nltk.stem.PorterStemmer()
    stopwords_set = set(nltk.corpus.stopwords.words('english'))

    # Preprocess the statement
    processed_statement = preprocess_text(statement, stemmer, stopwords_set)

    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([processed_statement])

    # Pad the sequence
    max_length = 100
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    # Encode speaker and subject with 'unknown' handling
    def encode_feature(feature, encoder, classes):
        if feature in classes:
            return encoder.transform([feature])[0]
        else:
            return len(classes)  # 'unknown' index

    speaker_classes = list(speaker_encoder.classes_)
    speaker_encoded = encode_feature(speaker, speaker_encoder, speaker_classes)

    subject_classes = list(subject_encoder.classes_)
    subject_encoded = encode_feature(subject, subject_encoder, subject_classes)

    # Prepare inputs
    speaker_encoded = np.array([speaker_encoded])
    subject_encoded = np.array([subject_encoded])

    # Make prediction
    prediction_prob = model.predict({
        'text_input': padded_sequence,
        'speaker_input': speaker_encoded,
        'subject_input': subject_encoded
    })

    # Interpret the prediction
    predicted_label = (prediction_prob > 0.5).astype(int)[0][0]
    label = 'Not Fake' if predicted_label == 1 else 'Fake'
    confidence = prediction_prob[0][0] if predicted_label == 1 else 1 - prediction_prob[0][0]

    print(f"\nStatement: {statement}")
    print(f"Speaker: {speaker}")
    print(f"Subject: {subject}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")

    return label

if __name__ == "__main__":
    main()

    # Example usage of the predict_statement function
    test_statement = "The economy is growing faster than ever."
    test_speaker = "Barack Obama"
    test_subject = "economy"
    prediction = predict_statement(test_statement, test_speaker, test_subject)
    print(f"Prediction for the statement: '{test_statement}' is '{prediction}'")

    # Example usage of the predict_statement function
    test_statement = "The economy is growing faster than ever."
    test_speaker = "Barack Obama"
    test_subject = "economy"
    prediction = predict_statement(test_statement, test_speaker, test_subject)
    print(f"Prediction for the statement: '{test_statement}' is '{prediction}'")

from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder

def download_nltk_dependencies():
    nltk.download('stopwords')

def preprocess_text(text, stemmer, stopwords_set):
    words = text.lower().split()
    words = [stemmer.stem(word) for word in words if word not in stopwords_set]
    return ' '.join(words)

def load_and_map_labels(train_path, val_path, test_path, label_map):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    train_df['label'] = train_df['label'].map(label_map)
    val_df['label'] = val_df['label'].map(label_map)
    test_df['label'] = test_df['label'].map(label_map)

    return train_df, val_df, test_df

def encode_with_unknown(series, encoder, classes):
    unknown_idx = len(classes)
    encoded = series.apply(lambda x: encoder.transform([x])[0] if x in classes else unknown_idx)
    return encoded

def build_model(vocab_size, embedding_dim, input_length, num_speakers, num_subjects):
    # Define your model architecture here
    return model

def evaluate_model(model, X_val, y_val, X_test, y_test):
    """Evaluate the model on validation and test sets."""
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Predictions
    val_preds = model.predict(X_val).round().astype(int)
    test_preds = model.predict(X_test).round().astype(int)

    # Confusion matrices
    val_cm = confusion_matrix(y_val, val_preds)
    test_cm = confusion_matrix(y_test, test_preds)

    # Display confusion matrices
    disp = ConfusionMatrixDisplay(confusion_matrix=val_cm, display_labels=['0', '1'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Validation Confusion Matrix')
    plt.show()

    disp = ConfusionMatrixDisplay(confusion_matrix=test_cm, display_labels=['0', '1'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Test Confusion Matrix')
    plt.show()

def main():
    """Main function to train and evaluate the model."""
    # Define directories and paths
    SAVE_DIR = './saved_models'
    TOKENIZER_PATH = os.path.join(SAVE_DIR, 'tokenizer.pkl')
    SPEAKER_ENCODER_PATH = os.path.join(SAVE_DIR, 'speaker_encoder.pkl')
    SUBJECT_ENCODER_PATH = os.path.join(SAVE_DIR, 'subject_encoder.pkl')
    LABEL_MAP_PATH = os.path.join(SAVE_DIR, 'label_map.pkl')
    MODEL_DIR = os.path.join(SAVE_DIR, 'final_model')  # Using SavedModel format

    # Create directories if they don't exist
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Download necessary NLTK data
    download_nltk_dependencies()

    # Set mixed precision policy for GPU performance optimization
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f"Mixed precision policy set to: {policy}")

    # Verify GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Num GPUs Available: {len(gpus)}")
    if gpus:
        for gpu in gpus:
            # Print GPU information
            print(f"GPU: {gpu.name}")
    else:
        print("No GPUs detected. Exiting...")
        return

    # Initialize stemmer and stopwords
    stemmer = nltk.stem.PorterStemmer()
    stopwords_set = set(nltk.corpus.stopwords.words('english'))

    # Load datasets with label mapping
    label_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    data_paths = ['../data/train.csv', '../data/valid.csv', '../data/test.csv']

    # Check if data files exist
    for path in data_paths:
        if not os.path.exists(path):
            print(f"Data file not found at {path}. Please check the path.")
            return

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
    train_df['speaker_encoded'] = encode_with_unknown(train_df['speaker'], speaker_encoder, speaker_classes)
    val_df['speaker_encoded'] = encode_with_unknown(val_df['speaker'], speaker_encoder, speaker_classes)
    test_df['speaker_encoded'] = encode_with_unknown(test_df['speaker'], speaker_encoder, speaker_classes)

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
    train_df['subject_encoded'] = encode_with_unknown(train_df['subject'], subject_encoder, subject_classes)
    val_df['subject_encoded'] = encode_with_unknown(val_df['subject'], subject_encoder, subject_classes)
    test_df['subject_encoded'] = encode_with_unknown(test_df['subject'], subject_encoder, subject_classes)

    # Save subject encoder
    with open(SUBJECT_ENCODER_PATH, 'wb') as f:
        pickle.dump(subject_encoder, f)
    print(f"Subject encoder saved to {SUBJECT_ENCODER_PATH}")

    # Define model parameters
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100
    input_length = max_length

    # Build and compile the model
    model = build_model(vocab_size, embedding_dim, input_length, num_speakers, num_subjects)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Early stopping and model checkpoint callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(MODEL_DIR, save_best_only=True, save_weights_only=False)

    # Train the model
    history = model.fit(
        [X_train_text_padded, train_df['speaker_encoded'], train_df['subject_encoded']],
        train_df['label'],
        epochs=10,
        batch_size=32,
        validation_data=(
            [X_val_text_padded, val_df['speaker_encoded'], val_df['subject_encoded']],
            val_df['label']
        ),
        callbacks=[early_stopping, model_checkpoint]
    )

    # Evaluate on validation and test sets
    evaluate_model(model,
                   [X_val_text_padded, val_df['speaker_encoded'], val_df['subject_encoded']],
                   val_df['label'],
                   [X_test_text_padded, test_df['speaker_encoded'], test_df['subject_encoded']],
                   test_df['label'])

if __name__ == "__main__":
    main()
