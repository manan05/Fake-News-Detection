# train.py

import os
import re
import pickle
import nltk
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertTokenizer
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Embedding, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Ensure TensorFlow is using mixed precision for better performance and reduced memory usage
from tensorflow.keras.mixed_precision import set_global_policy



def download_nltk_dependencies():
    """Download required NLTK packages."""
    nltk.download('punkt')
    nltk.download('stopwords')

def preprocess_text(text):
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

def load_and_map_labels(train_path, val_path, test_path, label_map):

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    for df in [train_df, val_df, test_df]:
        df['label'] = df['label'].map(label_map)
        # Drop rows with missing essential fields
        df.dropna(subset=['statement', 'speaker', 'subject', 'label'], inplace=True)

    return train_df, val_df, test_df

def encode_with_unknown(labels, label_encoder):
    classes = label_encoder.classes_
    mapping = {label: idx for idx, label in enumerate(classes)}
    unknown_idx = len(classes)  # Assign the next index for unknowns
    encoded = [mapping.get(label, unknown_idx) for label in labels]
    return np.array(encoded)

def save_object(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_object(filepath):
    """Load a Python object from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def create_directories(directories):
    """Create directories if they don't exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def main():
    DATA_DIR = '../data'
    MODEL_DIR = './models'
    SAVED_OBJECTS_DIR = './saved_objects'

    create_directories([DATA_DIR, MODEL_DIR, SAVED_OBJECTS_DIR])

    TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
    VAL_PATH = os.path.join(DATA_DIR, 'valid.csv')
    TEST_PATH = os.path.join(DATA_DIR, 'test.csv')

    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'distilbert_model.h5')
    TOKENIZER_SAVE_PATH = os.path.join(SAVED_OBJECTS_DIR, 'tokenizer.pkl')
    SPEAKER_ENCODER_PATH = os.path.join(SAVED_OBJECTS_DIR, 'speaker_encoder.pkl')
    SUBJECT_ENCODER_PATH = os.path.join(SAVED_OBJECTS_DIR, 'subject_encoder.pkl')
    LABEL_MAP_PATH = os.path.join(SAVED_OBJECTS_DIR, 'label_map.pkl')

    download_nltk_dependencies()

    label_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    train_df, val_df, test_df = load_and_map_labels(TRAIN_PATH, VAL_PATH, TEST_PATH, label_map)

    save_object(label_map, LABEL_MAP_PATH)
    print(f"Label map saved to {LABEL_MAP_PATH}")

    for df in [train_df, val_df, test_df]:
        df['statement'] = df['statement'].astype(str).apply(preprocess_text)
        df['speaker'] = df['speaker'].astype(str)
        df['subject'] = df['subject'].astype(str)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    save_object(tokenizer, TOKENIZER_SAVE_PATH)
    print(f"Tokenizer saved to {TOKENIZER_SAVE_PATH}")

    def tokenize_data(data, tokenizer, max_length=128):
        return tokenizer(
            data.tolist(),
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='tf'
        )

    train_encodings = tokenize_data(train_df['statement'], tokenizer, max_length=128)
    val_encodings = tokenize_data(val_df['statement'], tokenizer, max_length=128)
    test_encodings = tokenize_data(test_df['statement'], tokenizer, max_length=128)

    speaker_encoder = LabelEncoder()
    speaker_encoder.fit(train_df['speaker'])
    save_object(speaker_encoder, SPEAKER_ENCODER_PATH)
    print(f"Speaker encoder saved to {SPEAKER_ENCODER_PATH}")

    subject_encoder = LabelEncoder()
    subject_encoder.fit(train_df['subject'])
    save_object(subject_encoder, SUBJECT_ENCODER_PATH)
    print(f"Subject encoder saved to {SUBJECT_ENCODER_PATH}")

    num_speakers = len(speaker_encoder.classes_) + 1
    num_subjects = len(subject_encoder.classes_) + 1

    train_speaker = encode_with_unknown(train_df['speaker'], speaker_encoder)
    val_speaker = encode_with_unknown(val_df['speaker'], speaker_encoder)
    test_speaker = encode_with_unknown(test_df['speaker'], speaker_encoder)

    train_subject = encode_with_unknown(train_df['subject'], subject_encoder)
    val_subject = encode_with_unknown(val_df['subject'], subject_encoder)
    test_subject = encode_with_unknown(test_df['subject'], subject_encoder)

    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # Update the mixed precision API for TensorFlow 2.11
    set_global_policy('mixed_float16')
    print("Mixed precision set to 'mixed_float16'.")

    def create_model():
        # Text Inputs
        input_ids = Input(shape=(128,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(128,), dtype=tf.int32, name='attention_mask')

        # Speaker and Subject Inputs
        speaker_input = Input(shape=(1,), dtype=tf.int32, name='speaker_input')
        subject_input = Input(shape=(1,), dtype=tf.int32, name='subject_input')

        # DistilBERT Model
        distilbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        distilbert_model.trainable = False  # Freeze DistilBERT layers to save memory and speed up training

        # Get the [CLS] token representation
        distilbert_output = distilbert_model(input_ids, attention_mask=attention_mask)[0][:, 0, :]  # Shape: (batch_size, hidden_size)

        # Speaker Embedding
        speaker_embedding = Embedding(
            input_dim=num_speakers,
            output_dim=16,
            input_length=1,
            name='speaker_embedding'
        )(speaker_input)
        speaker_embedding = Flatten()(speaker_embedding)  # Shape: (batch_size, 16)

        # Subject Embedding
        subject_embedding = Embedding(
            input_dim=num_subjects,
            output_dim=16,
            input_length=1,
            name='subject_embedding'
        )(subject_input)
        subject_embedding = Flatten()(subject_embedding)  # Shape: (batch_size, 16)

        # Concatenate All Features
        combined = Concatenate()([distilbert_output, speaker_embedding, subject_embedding])  # Shape: (batch_size, hidden_size + 32)

        # Fully Connected Layers
        combined = Dropout(0.3)(combined)
        combined = Dense(128, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        output = Dense(1, activation='sigmoid', dtype='float32')(combined)  # Output Layer

        model = Model(inputs=[input_ids, attention_mask, speaker_input, subject_input], outputs=output)
        return model

    model = create_model()
    model.compile(optimizer=Adam(learning_rate=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Reduce batch size to 4 to accommodate GPU memory constraints
    batch_size = 4

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(( 
        {
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'speaker_input': train_speaker,
            'subject_input': train_subject
        },
        y_train
    )).shuffle(len(y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(( 
        {
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask'],
            'speaker_input': val_speaker,
            'subject_input': val_subject
        },
        y_val
    )).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices(( 
        {
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask'],
            'speaker_input': test_speaker,
            'subject_input': test_subject
        },
        y_test
    )).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,  # Saves the entire model
        verbose=1
    )

    epochs = 10  # Adjust as needed

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    model.save(MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")

    loss, accuracy = model.evaluate(test_dataset)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    val_preds = model.predict(val_dataset)
    val_preds_labels = (val_preds > 0.5).astype(int).flatten()

    test_preds = model.predict(test_dataset)
    test_preds_labels = (test_preds > 0.5).astype(int).flatten()

    val_cm = confusion_matrix(y_val, val_preds_labels)
    disp_val = ConfusionMatrixDisplay(confusion_matrix=val_cm, display_labels=['Fake', 'Not Fake'])
    disp_val.plot(cmap=plt.cm.Blues)
    plt.title("Validation Confusion Matrix")
    plt.show()

    test_cm = confusion_matrix(y_test, test_preds_labels)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=test_cm, display_labels=['Fake', 'Not Fake'])
    disp_test.plot(cmap=plt.cm.Blues)
    plt.title("Test Confusion Matrix")
    plt.show()

if __name__ == '__main__':
    main()
