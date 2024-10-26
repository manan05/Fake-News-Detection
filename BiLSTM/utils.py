# utils.py

import re
import pickle
import nltk
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Ensure nltk packages are downloaded
def download_nltk_dependencies():
    nltk.download('punkt')
    nltk.download('stopwords')

# Preprocess text data
def preprocess_text(text):
    """
    Preprocess the input text by removing non-alphabet characters,
    tokenizing, lowercasing, and removing stopwords.

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    stemmer = nltk.stem.PorterStemmer()
    stopwords_set = set(nltk.corpus.stopwords.words('english'))

    # Remove non-alphabet characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize and convert to lowercase
    words = nltk.word_tokenize(text.lower())
    # Remove stopwords
    words = [word for word in words if word not in stopwords_set]
    # Stem words
    processed_words = [stemmer.stem(word) for word in words]
    return ' '.join(processed_words)

# Load and map labels
def load_and_map_labels(train_path, val_path, test_path, label_map):
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
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    for df in [train_df, val_df, test_df]:
        df['label'] = df['label'].map(label_map)
        df.dropna(subset=['statement', 'speaker', 'subject', 'label'], inplace=True)

    return train_df, val_df, test_df

# Custom encoding function to handle unknown labels
def encode_with_unknown(labels, label_encoder):
    """
    Encode labels, assigning a special index to unknown labels.

    Args:
        labels (pd.Series): Series of labels to encode.
        label_encoder (LabelEncoder): Fitted LabelEncoder.

    Returns:
        np.array: Encoded labels with unknowns mapped to a special index.
    """
    classes = label_encoder.classes_
    mapping = {label: idx for idx, label in enumerate(classes)}
    unknown_idx = len(classes)  # Assign the next index for unknowns
    encoded = [mapping.get(label, unknown_idx) for label in labels]
    return np.array(encoded)

# Save object to file
def save_object(obj, filepath):
    """Save a Python object to a file using pickle."""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

# Load object from file
def load_object(filepath):
    """Load a Python object from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Create directories if they don't exist
def create_directories(directories):
    """Create directories if they don't exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
