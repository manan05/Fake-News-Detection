

## **Fake News Detection with Bidirectional LSTM**

### **Table of Contents**
1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Data Description](#data-description)
4. [Directory Structure](#directory-structure)
5. [Code Structure](#code-structure)
    - [1. Import Statements](#1-import-statements)
    - [2. Utility Functions](#2-utility-functions)
        - [2.1 `download_nltk_dependencies`](#21-download_nltk_dependencies)
        - [2.2 `preprocess_text`](#22-preprocess_text)
        - [2.3 `load_and_map_labels`](#23-load_and_map_labels)
        - [2.4 `encode_with_unknown`](#24-encode_with_unknown)
    - [3. Model Building](#3-model-building)
        - [3.1 `build_model`](#31-build_model)
    - [4. Main Workflow](#4-main-workflow)
        - [4.1 Directory Setup](#41-directory-setup)
        - [4.2 NLTK Dependencies](#42-nltk-dependencies)
        - [4.3 GPU Configuration](#43-gpu-configuration)
        - [4.4 Data Loading and Preprocessing](#44-data-loading-and-preprocessing)
        - [4.5 Tokenization](#45-tokenization)
        - [4.6 Encoding Categorical Features](#46-encoding-categorical-features)
        - [4.7 Model Compilation](#47-model-compilation)
        - [4.8 Callbacks](#48-callbacks)
        - [4.9 Model Training](#49-model-training)
        - [4.10 Model Evaluation](#410-model-evaluation)
        - [4.11 Prediction Function](#411-prediction-function)
        - [4.12 Example Usage](#412-example-usage)
6. [Usage Instructions](#usage-instructions)
7. [Handling Unseen Labels](#handling-unseen-labels)
8. [Troubleshooting](#troubleshooting)
9. [Additional Enhancements](#additional-enhancements)
10. [Conclusion](#conclusion)

---

## **Introduction**

This project implements a Fake News Detection system using a Bidirectional Long Short-Term Memory (BiLSTM) neural network with TensorFlow. The model leverages multiple features from the dataset, including textual content (`statement`), `speaker`, and `subject`, to classify statements as 'Fake' or 'Not Fake'. The architecture is designed to efficiently utilize all available data and handle unseen categories gracefully.

---

## **Dependencies**

Ensure you have the following libraries installed in your Python environment:

- **Python 3.x**
- **TensorFlow 2.x**
- **Keras** (bundled with TensorFlow)
- **NLTK**
- **Pandas**
- **Scikit-learn**
- **NumPy**

You can install the required libraries using `pip`:

```bash
pip install tensorflow nltk pandas scikit-learn numpy
```

Additionally, ensure that NLTK data packages are downloaded (handled in the code).

---

## **Data Description**

### **Dataset Columns:**

1. **`id`**: Unique identifier for each statement.
2. **`label`**: Target variable indicating the truthfulness of the statement. Original labels (0-5) are mapped to binary classification:
    - `0-2`: Mapped to `0` (Fake)
    - `3-5`: Mapped to `1` (Not Fake)
3. **`statement`**: Textual content of the statement.
4. **`subject`**: Topic/category of the statement (e.g., taxes, healthcare).
5. **`speaker`**: Individual or organization making the statement.
6. **Additional Columns**: `speaker_description`, `state_info`, etc., which are not utilized in this model but can be incorporated for enhanced feature engineering.

### **Dataset Splits:**

- **Training Set**: `train.csv`
- **Validation Set**: `valid.csv`
- **Test Set**: `test.csv`

Ensure these CSV files are correctly placed in the specified directories.

---

## **Directory Structure**

```plaintext
Fake-News-Detection/
│
├── BiLSTM/
│   ├── BiLSTM.py          # Main Python script
│   ├── saved_models/      # Directory to save models and artifacts
│   │   ├── tokenizer.pkl
│   │   ├── speaker_encoder.pkl
│   │   ├── subject_encoder.pkl
│   │   ├── label_map.pkl
│   │   └── final_model.h5
│   └── data/
│       ├── train.csv
│       ├── valid.csv
│       └── test.csv
│
└── env/                   # Virtual environment (if used)
```

Ensure that the `data` directory contains the `train.csv`, `valid.csv`, and `test.csv` files.

---

## **Code Structure**

The code is organized into several sections for clarity and modularity. Below is a breakdown of each component with detailed explanations.

### **1. Import Statements**

```python
import re
import pickle
from typing import Tuple

import nltk
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Bidirectional, Dense, Dropout, Embedding, LSTM, Input, Concatenate, Flatten
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
```

- **Standard Libraries**: `re`, `pickle`, `os`, `numpy`, and `typing` for regular expressions, object serialization, file operations, numerical operations, and type hinting respectively.
- **NLTK**: Natural Language Toolkit for text preprocessing.
- **Pandas**: Data manipulation and analysis.
- **TensorFlow and Keras**: Deep learning framework and API for building and training models.
- **Scikit-learn**: For label encoding and other preprocessing tasks.

---

### **2. Utility Functions**

#### **2.1 `download_nltk_dependencies`**

Downloads necessary NLTK data packages required for text preprocessing.

```python
def download_nltk_dependencies() -> None:
    """Download required NLTK packages."""
    nltk.download('punkt')
    nltk.download('stopwords')
```

- **`punkt`**: Tokenizer models.
- **`stopwords`**: Commonly used words to be filtered out.

#### **2.2 `preprocess_text`**

Preprocesses textual data by cleaning, tokenizing, stemming, and removing stopwords.

```python
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
```

- **Cleaning**: Removes any character that isn't an alphabet.
- **Tokenization**: Splits text into individual words.
- **Lowercasing**: Converts all characters to lowercase.
- **Stemming**: Reduces words to their root form using Porter Stemmer.
- **Stopwords Removal**: Filters out common English stopwords.

#### **2.3 `load_and_map_labels`**

Loads datasets and maps original labels to binary labels as per `label_map`.

```python
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
```

- **Loading**: Reads CSV files into Pandas DataFrames.
- **Label Mapping**: Converts original multi-class labels into binary labels based on `label_map`.

#### **2.4 `encode_with_unknown`**

Encodes categorical labels and assigns a special index to unknown labels.

```python
def encode_with_unknown(labels, label_encoder):
    """
    Encode labels, assigning a special index to unknown labels.

    Args:
        labels (pd.Series): Series of labels to encode.
        label_encoder (LabelEncoder): Fitted LabelEncoder.

    Returns:
        list: Encoded labels with unknowns mapped to a special index.
    """
    classes = label_encoder.classes_
    mapping = {label: idx for idx, label in enumerate(classes)}
    unknown_idx = len(classes)  # Assign the next index for unknowns
    encoded = [mapping.get(label, unknown_idx) for label in labels]
    return encoded
```

- **Purpose**: Handles cases where new, unseen labels (e.g., new speakers or subjects) appear in validation or test sets.
- **Mechanism**: Maps known labels to their respective indices and assigns a unique index for unknown labels.

---

### **3. Model Building**

#### **3.1 `build_model`**

Constructs the BiLSTM model with additional categorical features (`speaker` and `subject`).

```python
def build_model(input_dim: int, output_dim: int, input_length: int, num_speakers: int, num_subjects: int) -> Model:
    """
    Build a Bidirectional LSTM model for binary classification with additional features.

    Args:
        input_dim (int): Size of the vocabulary.
        output_dim (int): Dimension of the embedding vectors.
        input_length (int): Length of input sequences.
        num_speakers (int): Number of unique speakers plus one for 'unknown'.
        num_subjects (int): Number of unique subjects plus one for 'unknown'.

    Returns:
        Model: The compiled Keras model.
    """
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
```

- **Inputs**:
    - **`text_input`**: Tokenized and padded text sequences.
    - **`speaker_input`**: Encoded speaker indices.
    - **`subject_input`**: Encoded subject indices.
- **Embedding Layers**:
    - **Text Embedding**: Converts words into dense vectors.
    - **Speaker Embedding**: Learns embeddings for speakers.
    - **Subject Embedding**: Learns embeddings for subjects.
- **BiLSTM Layers**: Capture contextual information from text.
- **Dense Layers**: Combine all features and perform classification.
- **Output Layer**: Single neuron with sigmoid activation for binary classification.

---

### **4. Main Workflow**

The `main` function orchestrates the entire process, from data loading to model training and evaluation.

```python
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

    # Define predict function
    def predict_statement(statement: str, speaker: str, subject: str) -> str:
        """
        Predict whether a statement is 'Fake' or 'Not Fake'.

        Args:
            statement (str): The statement to classify.
            speaker (str): The speaker of the statement.
            subject (str): The subject of the statement.

        Returns:
            str: 'Fake' or 'Not Fake'
        """
        # Load the trained model and tokenizer
        model = load_model(MODEL_PATH, compile=False)
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
```

---

### **4.1 Directory Setup**

```python
# Define directories for saving models and artifacts
SAVE_DIR = './saved_models'
TOKENIZER_PATH = os.path.join(SAVE_DIR, 'tokenizer.pkl')
SPEAKER_ENCODER_PATH = os.path.join(SAVE_DIR, 'speaker_encoder.pkl')
SUBJECT_ENCODER_PATH = os.path.join(SAVE_DIR, 'subject_encoder.pkl')
LABEL_MAP_PATH = os.path.join(SAVE_DIR, 'label_map.pkl')
MODEL_PATH = os.path.join(SAVE_DIR, 'final_model.h5')

# Create directories if they don't exist
os.makedirs(SAVE_DIR, exist_ok=True)
```

- **Purpose**: Establishes a directory (`./saved_models`) to store all model-related artifacts such as the tokenizer, encoders, label map, and the trained model itself.
- **Functionality**: Uses `os.makedirs` with `exist_ok=True` to create the directory if it doesn't already exist, avoiding errors if it does.

---

### **4.2 NLTK Dependencies**

```python
# Download necessary NLTK data
download_nltk_dependencies()
```

- **Purpose**: Ensures that the required NLTK packages (`punkt` and `stopwords`) are available for text preprocessing.
- **Functionality**: Calls the `download_nltk_dependencies` function defined earlier.

---

### **4.3 GPU Configuration**

```python
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
```

- **Mixed Precision**: Utilizes `mixed_float16` policy to speed up training and reduce memory usage on compatible GPUs by using 16-bit floating points where possible.
- **GPU Availability**: Checks for available GPUs. If GPUs are detected, configures TensorFlow to grow GPU memory as needed instead of allocating all memory upfront.
- **Error Handling**: Catches `RuntimeError` which occurs if GPU configuration is attempted after TensorFlow has initialized GPU memory.

**Note**: If no GPU is detected, the script will print a message and exit to prevent running on CPU, which might be inefficient for large models.

---

### **4.4 Data Loading and Preprocessing**

```python
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
```

- **Stemming and Stopwords**:
    - **Stemmer**: Initializes the Porter Stemmer for reducing words to their base forms.
    - **Stopwords Set**: Creates a set of English stopwords to be removed during preprocessing.
- **Loading Datasets**:
    - **Label Mapping**: Converts original multi-class labels into binary labels (`0` for Fake, `1` for Not Fake).
    - **Function Call**: Uses `load_and_map_labels` to load and map labels for training, validation, and test sets.
- **Saving Label Map**:
    - **Purpose**: Stores the `label_map` for future reference or inverse mapping.
    - **Method**: Uses `pickle` to serialize and save the `label_map` dictionary.
- **Preprocessing Statements**:
    - **Function Call**: Applies the `preprocess_text` function to the `statement` column of each DataFrame.
    - **Lambda Function**: Ensures that each statement is preprocessed consistently across all datasets.

---

### **4.5 Tokenization**

```python
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
```

- **Tokenizer Initialization**:
    - **`num_words=10000`**: Limits the vocabulary to the top 10,000 most frequent words.
    - **`oov_token="<OOV>"`**: Represents out-of-vocabulary words.
- **Fitting the Tokenizer**:
    - **Dataset**: Trained on the `statement` column of the training set.
    - **Purpose**: Builds the word index based on frequency.
- **Saving the Tokenizer**:
    - **Purpose**: Allows consistent tokenization of new data during prediction.
    - **Method**: Serialized and saved using `pickle`.
- **Text to Sequences**:
    - **Function Call**: Converts preprocessed text into sequences of integers representing word indices.
    - **Datasets**: Applied to training, validation, and test sets.
- **Padding Sequences**:
    - **`max_length=100`**: All sequences are padded or truncated to a fixed length of 100.
    - **`padding='post'`**: Adds padding at the end of sequences.
    - **Purpose**: Ensures uniform input size for the model.

---

### **4.6 Encoding Categorical Features**

#### **Speaker Encoding**

```python
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
```

- **Label Encoding**:
    - **Purpose**: Converts categorical `speaker` labels into numerical indices.
    - **`LabelEncoder`**: Fits on the training `speaker` data to learn the mapping.
- **Handling Unknown Speakers**:
    - **Unknown Index**: Assigns an index (`num_speakers`) for any speaker not seen during training.
    - **Function `encode_speaker`**: Encodes speakers and assigns the unknown index for unseen speakers.
- **Encoding Process**:
    - **Training Set**: Encodes speakers using the fitted `LabelEncoder`.
    - **Validation & Test Sets**: Encodes speakers, assigning the unknown index if necessary.
- **Saving Encoder**:
    - **Purpose**: Facilitates consistent encoding during prediction.
    - **Method**: Serialized and saved using `pickle`.

#### **Subject Encoding**

```python
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
```

- **Label Encoding**:
    - **Purpose**: Converts categorical `subject` labels into numerical indices.
    - **`LabelEncoder`**: Fits on the training `subject` data to learn the mapping.
- **Handling Unknown Subjects**:
    - **Unknown Index**: Assigns an index (`num_subjects`) for any subject not seen during training.
    - **Function `encode_subject`**: Encodes subjects and assigns the unknown index for unseen subjects.
- **Encoding Process**:
    - **Training Set**: Encodes subjects using the fitted `LabelEncoder`.
    - **Validation & Test Sets**: Encodes subjects, assigning the unknown index if necessary.
- **Saving Encoder**:
    - **Purpose**: Facilitates consistent encoding during prediction.
    - **Method**: Serialized and saved using `pickle`.

---

### **4.7 Model Compilation**

```python
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
```

- **Labels Extraction**:
    - **`y_train`, `y_val`, `y_test`**: Extracted as NumPy arrays for training, validation, and testing respectively.
- **Model Building and Compilation**:
    - **Model Instantiation**: Calls `build_model` with appropriate parameters.
    - **Optimizer**: Uses Adam optimizer with a learning rate of `1e-3`.
    - **Compilation**:
        - **Loss Function**: Binary cross-entropy, suitable for binary classification.
        - **Metrics**: Monitors accuracy during training.
- **Device Context**:
    - **GPU**: Ensures that model building and compilation occur on the GPU if available.
- **Model Summary**:
    - **Purpose**: Provides a summary of the model architecture, including layers, output shapes, and parameter counts.

---

### **4.8 Callbacks**

```python
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
```

- **Early Stopping**:
    - **Purpose**: Prevents overfitting by stopping training when validation loss doesn't improve for a specified number of epochs (`patience=7`).
    - **`restore_best_weights=True`**: Restores model weights from the epoch with the best validation loss.
- **Model Checkpointing**:
    - **Purpose**: Saves the best model based on validation loss during training.
    - **Parameters**:
        - **`monitor='val_loss'`**: Monitors validation loss.
        - **`save_best_only=True`**: Saves only the model with the best validation loss.
        - **`save_weights_only=False`**: Saves the entire model architecture and weights.
- **Verbose**:
    - **`verbose=1`**: Displays messages when callbacks are triggered.

---

### **4.9 Model Training**

```python
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
```

- **Inputs**:
    - **Features**:
        - **`text_input`**: Padded text sequences.
        - **`speaker_input`**: Encoded speaker indices.
        - **`subject_input`**: Encoded subject indices.
    - **Labels**: Binary labels (`y_train`).
- **Validation Data**:
    - **Purpose**: Monitors model performance on unseen data during training.
    - **Format**: Same as training data.
- **Training Parameters**:
    - **`epochs=40`**: Maximum number of training epochs.
    - **`batch_size=64`**: Number of samples per gradient update. Adjust based on GPU memory.
- **Callbacks**:
    - **Early Stopping** and **Model Checkpointing**: Applied to manage training progression and model saving.

---

### **4.10 Model Evaluation**

```python
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
```

- **Model Saving**:
    - **Purpose**: Saves the trained model to disk for future use or deployment.
    - **Method**: Uses `model.save` to serialize the entire model (architecture + weights).
- **Model Evaluation**:
    - **Purpose**: Assesses the model's performance on the test dataset.
    - **Metrics**: Computes loss and accuracy.
    - **Input Format**: Same as training and validation inputs.
- **Output**:
    - **Print Statement**: Displays the test accuracy in percentage format.

---

### **4.11 Prediction Function**

```python
# Define predict function
def predict_statement(statement: str, speaker: str, subject: str) -> str:
    """
    Predict whether a statement is 'Fake' or 'Not Fake'.

    Args:
        statement (str): The statement to classify.
        speaker (str): The speaker of the statement.
        subject (str): The subject of the statement.

    Returns:
        str: 'Fake' or 'Not Fake'
    """
    # Load the trained model and tokenizer
    model = load_model(MODEL_PATH, compile=False)
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
```

- **Purpose**: Enables the classification of new, manual statements by users.
- **Parameters**:
    - **`statement`**: The textual content to classify.
    - **`speaker`**: The individual or organization making the statement.
    - **`subject`**: The topic/category of the statement.
- **Workflow**:
    1. **Model and Tokenizer Loading**:
        - **`load_model`**: Loads the trained model from disk.
        - **`pickle.load`**: Loads the tokenizer and encoders.
    2. **Preprocessing**:
        - **Text**: Applies the same preprocessing as during training.
        - **Speaker & Subject**: Encodes using the same mappings, handling unknowns by assigning a special index.
    3. **Sequence Conversion and Padding**:
        - **Sequence**: Converts preprocessed text to a sequence of integers.
        - **Padding**: Pads the sequence to match the model's expected input length.
    4. **Encoding Features**:
        - **Speaker & Subject**: Encoded into numerical indices, with unknowns handled appropriately.
    5. **Prediction**:
        - **Model Prediction**: Feeds the processed inputs into the model to get a prediction.
        - **Interpretation**: Converts the sigmoid output to a binary label ('Fake' or 'Not Fake') based on a threshold of 0.5.
- **Return Value**: A string indicating whether the statement is 'Fake' or 'Not Fake'.

---

### **4.12 Example Usage**

```python
# Example usage
test_statement = "The economy is growing faster than ever."
test_speaker = "Barack Obama"
test_subject = "economy"
prediction = predict_statement(test_statement, test_speaker, test_subject)
print(f"Prediction for the statement: '{test_statement}' is '{prediction}'")
```

- **Purpose**: Demonstrates how to use the `predict_statement` function with sample inputs.
- **Output**:
    - **Print Statement**: Displays the prediction result for the provided statement, speaker, and subject.

---

## **Usage Instructions**

1. **Setup Environment**:
    - Ensure you have Python 3.x installed.
    - Create and activate a virtual environment (optional but recommended):
        ```bash
        python -m venv env
        source env/bin/activate  # On Windows: env\Scripts\activate
        ```
    - Install required packages:
        ```bash
        pip install tensorflow nltk pandas scikit-learn numpy
        ```

2. **Prepare Data**:
    - Place your `train.csv`, `valid.csv`, and `test.csv` files in the `../data/` directory relative to the script location.
    - Ensure the CSV files have the necessary columns as described in the [Data Description](#data-description).

3. **Run the Script**:
    - Execute the Python script:
        ```bash
        python BiLSTM.py
        ```
    - The script will perform the following:
        - Download NLTK dependencies.
        - Preprocess the data.
        - Encode categorical features with handling for unknown labels.
        - Build and compile the BiLSTM model.
        - Train the model with early stopping and model checkpointing.
        - Evaluate the model on the test dataset.
        - Perform a sample prediction.

4. **Manual Predictions**:
    - Modify the `test_statement`, `test_speaker`, and `test_subject` variables in the script to test new statements.
    - Re-run the script to see the prediction results.

---

## **Handling Unseen Labels**

### **Issue Encountered**

```plaintext
KeyError: 'lindsay james'

During handling of the above exception, another exception occurred:

ValueError: y contains previously unseen labels: 'lindsay james'
```

### **Cause**

The error occurs because the `LabelEncoder` was fitted on the training data's `speaker` and `subject` columns, but during validation or testing, it encountered a speaker (`'lindsay james'`) that wasn't present in the training set. `LabelEncoder` cannot handle unseen labels by default, leading to a `KeyError`.

### **Solution**

Implement a custom encoding function that assigns a unique index to unknown labels, ensuring the model can handle unseen categories gracefully.

### **Implementation Steps**

1. **Custom Encoding Function**:
    - Maps known labels to their indices.
    - Assigns a special index (`unknown_idx`) for any label not seen during training.

    ```python
    def encode_speaker(speakers, encoder, classes, unknown_label='unknown'):
        mapping = {label: idx for idx, label in enumerate(classes)}
        unknown_idx = len(classes)
        encoded = [mapping.get(label, unknown_idx) for label in speakers]
        return encoded
    ```

2. **Adjust Embedding Layers**:
    - **Input Dimension**: Increase by one to accommodate the 'unknown' category.
    - **Speaker Embedding**: `input_dim=num_speakers` where `num_speakers = len(speaker_classes) + 1`.

    ```python
    speaker_embedding = Embedding(input_dim=num_speakers, output_dim=16, input_length=1)(speaker_input)
    ```

3. **Encoding Process**:
    - **Training Set**: Encodes speakers as usual.
    - **Validation & Test Sets**: Encodes speakers, assigning the 'unknown' index if the speaker isn't recognized.

    ```python
    train_df['speaker_encoded'] = encode_speaker(train_df['speaker'], speaker_encoder, speaker_classes)
    val_df['speaker_encoded'] = encode_speaker(val_df['speaker'], speaker_encoder, speaker_classes)
    test_df['speaker_encoded'] = encode_speaker(test_df['speaker'], speaker_encoder, speaker_classes)
    ```

4. **Prediction Handling**:
    - **Function `predict_statement`**: Uses the same encoding strategy to map unseen speakers and subjects to the 'unknown' index.

    ```python
    def encode_feature(feature, encoder, classes):
        mapping = {label: idx for idx, label in enumerate(classes)}
        unknown_idx = len(classes)
        return [mapping.get(feature, unknown_idx)]
    ```

5. **Model Architecture**:
    - **Embedding Layers**: Adjusted to include 'unknown' categories.

    ```python
    speaker_embedding = Embedding(input_dim=num_speakers, output_dim=16, input_length=1)(speaker_input)
    subject_embedding = Embedding(input_dim=num_subjects, output_dim=16, input_length=1)(subject_input)
    ```

By following these steps, the model can handle any speaker or subject not seen during training without raising errors, thus making the model robust and versatile.

---

## **Troubleshooting**

### **Common Issues and Solutions**

1. **`KeyError` During Label Encoding**:
    - **Cause**: Unseen labels in validation or test sets.
    - **Solution**: Implement custom encoding with handling for unknown labels as described above.

2. **GPU Memory Errors**:
    - **Cause**: Model or batch size too large for available GPU memory.
    - **Solution**: Reduce `batch_size` (e.g., from 64 to 32 or 16) or simplify the model architecture.

3. **Import Errors**:
    - **Cause**: Missing or incorrectly installed dependencies.
    - **Solution**: Ensure all required libraries are installed and up to date. Reinstall if necessary.

    ```bash
    pip install --upgrade tensorflow nltk pandas scikit-learn numpy
    ```

4. **Data Path Issues**:
    - **Cause**: Incorrect paths to `train.csv`, `valid.csv`, or `test.csv`.
    - **Solution**: Verify that the paths in `data_paths` list correctly point to your CSV files.

    ```python
    data_paths = ['../data/train.csv', '../data/valid.csv', '../data/test.csv']
    ```

5. **Model Not Saving Properly**:
    - **Cause**: Issues with file permissions or disk space.
    - **Solution**: Ensure that the script has write permissions to the `SAVE_DIR` and that sufficient disk space is available.

6. **Unseen Speakers/Subjects During Prediction**:
    - **Cause**: Input data contains new categories not seen during training.
    - **Solution**: The encoding functions handle this by assigning an 'unknown' index, ensuring smooth prediction.

---

## **Additional Enhancements**

To further improve the model's performance and robustness, consider implementing the following enhancements:

1. **Cross-Validation**:
    - Use k-fold cross-validation to evaluate the model's performance across different data splits.
    - Helps in assessing model generalizability.

2. **Hyperparameter Tuning**:
    - Experiment with different hyperparameters such as learning rate, batch size, number of LSTM units, embedding dimensions, and dropout rates.
    - Utilize tools like Grid Search or Random Search for systematic tuning.

3. **Advanced Feature Engineering**:
    - Incorporate additional features like `speaker_description`, `state_info`, or metadata from fact-checking columns.
    - Use Natural Language Processing (NLP) techniques to extract more meaningful features.

4. **Regularization Techniques**:
    - Implement techniques like L2 regularization or Dropout variations to prevent overfitting.

5. **Model Architecture Variations**:
    - Explore different architectures such as Convolutional Neural Networks (CNNs) combined with LSTMs, or transformer-based models like BERT for enhanced text understanding.

6. **Handling Imbalanced Data**:
    - If the dataset is imbalanced, use techniques like class weighting, oversampling, or undersampling to balance the classes.
    - Adjust the loss function or evaluation metrics accordingly.

    ```python
    from sklearn.utils import class_weight

    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    ```

    - Incorporate `class_weight` into the `model.fit` call.

    ```python
    history = model.fit(
        { ... },
        y_train,
        validation_data=( ... ),
        epochs=40,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    ```

7. **Saving and Loading Models and Encoders**:
    - Ensure that all necessary artifacts (tokenizer, encoders) are saved and can be loaded correctly for future predictions.
    - Implement functions to load these artifacts separately if needed.

8. **Evaluation Metrics**:
    - Utilize additional metrics such as Precision, Recall, F1-Score, and ROC-AUC for a more comprehensive evaluation.
    - Implement these using `sklearn.metrics`.

    ```python
    from sklearn.metrics import classification_report, confusion_matrix

    # After predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Not Fake']))
    print(confusion_matrix(y_test, y_pred))
    ```

9. **Visualization**:
    - Plot training and validation loss and accuracy over epochs to monitor training progress and detect overfitting.

    ```python
    import matplotlib.pyplot as plt

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()
    ```

10. **Deployment**:
    - Package the model into an API using frameworks like Flask or FastAPI for real-time predictions.
    - Ensure that the prediction function handles inputs securely and efficiently.

---

## **Conclusion**

This documentation provides a thorough overview of your Fake News Detection model using BiLSTM with TensorFlow. By following the structured approach outlined above, you can ensure effective data preprocessing, robust model training, and reliable predictions. The implementation of custom encoding functions for handling unseen labels enhances the model's adaptability to real-world scenarios where new speakers or subjects may emerge.

**Key Takeaways**:

- **Robust Preprocessing**: Consistent text preprocessing ensures that the model receives clean and standardized inputs.
- **Handling Unseen Labels**: Custom encoding functions prevent runtime errors and allow the model to handle new categories gracefully.
- **Feature Utilization**: Incorporating multiple features (`speaker`, `subject`) enriches the model's understanding and can improve classification performance.
- **Model Evaluation**: Proper evaluation on separate test data ensures that the model generalizes well to unseen data.
- **Extensibility**: The modular code structure facilitates easy enhancements and adaptations for future improvements.

Feel free to reach out if you need further assistance or have additional questions!