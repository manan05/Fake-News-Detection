import re
from typing import Tuple

import nltk
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Bidirectional, Dense, Dropout, Embedding,
                                     LSTM)
from tensorflow.keras.models import Sequential
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

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
    # Download necessary NLTK data
    download_nltk_dependencies()

    # Set mixed precision policy for GPU performance optimization
    set_global_policy('mixed_float16')

    # Verify GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Num GPUs Available: {len(gpus)}")

    # Initialize stemmer and stopwords
    stemmer = nltk.stem.PorterStemmer()
    stopwords_set = set(nltk.corpus.stopwords.words('english'))

    # Load datasets with label mapping
    label_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    train_df, val_df, test_df = load_and_map_labels('./data/train.csv', './data/valid.csv', './data/test.csv', label_map)

    # Preprocess the 'statement' column in each DataFrame
    for df in [train_df, val_df, test_df]:
        df['statement'] = df['statement'].astype(str).apply(
            lambda x: preprocess_text(x, stemmer, stopwords_set)
        )

    # Tokenization
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df['statement'])

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

    # Build and compile the model
    model = build_model(input_dim=10000, output_dim=64, input_length=max_length)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Display model architecture
    model.summary()

    # Training with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(
        X_train_padded, y_train,
        validation_data=(X_val_padded, y_val),
        epochs=10,
        batch_size=64,
        callbacks=[early_stopping]
    )

    # Evaluate the model on test data
    loss, accuracy = model.evaluate(X_test_padded, y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')


if __name__ == "__main__":
    main()
