import re
import pickle
from typing import Tuple

import nltk
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (Bidirectional, Dense, Dropout, Embedding,
                                     LSTM)
from tensorflow.keras.models import Sequential
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import os

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
    """
    Preprocess the input text by removing non-alphabet characters,
    tokenizing, lowercasing, stemming, and removing stopwords.

    Args:
        text (str): The text to preprocess.
        stemmer (nltk.stem.PorterStemmer): The stemmer to use.
        stopwords_set (set): A set of stopwords to remove.

    Returns:
        str: The preprocessed text.
    """
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
    """
    Load datasets and map labels according to the provided label_map.

    Args:
        train_path (str): Path to the training CSV file.
        val_path (str): Path to the validation CSV file.
        test_path (str): Path to the test CSV file.
        label_map (dict): Mapping from original labels to new labels.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Loaded and label-mapped DataFrames.
    """
    train_df = pd.read_csv(train_path, encoding='utf-8')
    val_df = pd.read_csv(val_path, encoding='utf-8')
    test_df = pd.read_csv(test_path, encoding='utf-8')

    for df in [train_df, val_df, test_df]:
        df['label'] = df['label'].map(label_map)

    return train_df, val_df, test_df


# Build the BiLSTM model
def build_model(input_dim: int, output_dim: int, input_length: int) -> Sequential:
    """
    Build a Bidirectional LSTM model for binary classification.

    Args:
        input_dim (int): Size of the vocabulary.
        output_dim (int): Dimension of the embedding vectors.
        input_length (int): Length of input sequences.

    Returns:
        Sequential: The compiled Keras model.
    """
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid', dtype='float32')  # Output layer for binary classification
    ])
    return model


def main() -> None:
    # Define directories for saving models and artifacts
    SAVE_DIR = './saved_models'
    TOKENIZER_PATH = os.path.join(SAVE_DIR, 'tokenizer.pkl')
    LABEL_MAP_PATH = os.path.join(SAVE_DIR, 'label_map.pkl')
    BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'best_model.h5')
    FINAL_MODEL_PATH = os.path.join(SAVE_DIR, 'final_model.h5')

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
    train_df, val_df, test_df = load_and_map_labels(
        './data/train.csv',
        './data/valid.csv',
        './data/test.csv',
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
    X_train = tokenizer.texts_to_sequences(train_df['statement'])
    X_val = tokenizer.texts_to_sequences(val_df['statement'])
    X_test = tokenizer.texts_to_sequences(test_df['statement'])

    # Pad sequences
    max_length = 100
    X_train_padded = pad_sequences(X_train, maxlen=max_length, padding='post')
    X_val_padded = pad_sequences(X_val, maxlen=max_length, padding='post')
    X_test_padded = pad_sequences(X_test, maxlen=max_length, padding='post')

    # Extract labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # Build and compile the model within the GPU device context
    with tf.device('/GPU:0'):
        model = build_model(input_dim=10000, output_dim=64, input_length=max_length)
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
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    # ModelCheckpoint to save the best model during training
    checkpoint = ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,  # Set to False to save the entire model
        verbose=1
    )

    # Fit the model
    history = model.fit(
        X_train_padded, y_train,
        validation_data=(X_val_padded, y_val),
        epochs=10,
        batch_size=64,  # Adjust batch size based on GPU memory
        callbacks=[early_stopping, checkpoint],
        verbose=1  # Set to 1 to see progress
    )

    # Save the final model after training completes
    model.save(FINAL_MODEL_PATH)
    print(f"Final model saved to {FINAL_MODEL_PATH}")

    # Evaluate the model on test data
    loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=1)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')


if __name__ == "__main__":
    main()
