# train.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Embedding, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os
import sys

from utils import (
    download_nltk_dependencies,
    preprocess_text,
    load_and_map_labels,
    encode_with_unknown,
    save_object,
    create_directories
)
from sklearn.preprocessing import LabelEncoder

def main():
    # =========================
    # Configuration and Paths
    # =========================
    DATA_DIR = '../data'
    MODEL_DIR = './models'
    SAVED_OBJECTS_DIR = './saved_objects'

    create_directories([DATA_DIR, MODEL_DIR, SAVED_OBJECTS_DIR])

    TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
    VAL_PATH = os.path.join(DATA_DIR, 'valid.csv')
    TEST_PATH = os.path.join(DATA_DIR, 'test.csv')

    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'text_cnn_model.h5')
    TOKENIZER_SAVE_PATH = os.path.join(SAVED_OBJECTS_DIR, 'tokenizer.pkl')
    SPEAKER_ENCODER_PATH = os.path.join(SAVED_OBJECTS_DIR, 'speaker_encoder.pkl')
    SUBJECT_ENCODER_PATH = os.path.join(SAVED_OBJECTS_DIR, 'subject_encoder.pkl')
    LABEL_MAP_PATH = os.path.join(SAVED_OBJECTS_DIR, 'label_map.pkl')

    # =========================
    # Check and Configure GPU
    # =========================
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for the GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"{len(gpus)} Physical GPUs found.")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
            sys.exit(1)
    else:
        print("No GPUs found.")
        sys.exit(1)

    # =========================
    # Download NLTK Dependencies
    # =========================
    download_nltk_dependencies()

    # =========================
    # Label Mapping
    # =========================
    # Original labels (assuming 0-5), mapping to binary classification
    label_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    train_df, val_df, test_df = load_and_map_labels(TRAIN_PATH, VAL_PATH, TEST_PATH, label_map)

    # Save label map for future use
    save_object(label_map, LABEL_MAP_PATH)
    print(f"Label map saved to {LABEL_MAP_PATH}")

    # =========================
    # Preprocess Text Data
    # =========================
    for df in [train_df, val_df, test_df]:
        df['statement'] = df['statement'].astype(str).apply(preprocess_text)
        df['speaker'] = df['speaker'].astype(str)
        df['subject'] = df['subject'].astype(str)

    # =========================
    # Initialize Tokenizer
    # =========================
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df['statement'])

    # Save the tokenizer for future use
    save_object(tokenizer, TOKENIZER_SAVE_PATH)
    print(f"Tokenizer saved to {TOKENIZER_SAVE_PATH}")

    # =========================
    # Tokenize and Pad Sequences
    # =========================
    def tokenize_and_pad(texts, tokenizer, max_length=100):
        sequences = tokenizer.texts_to_sequences(texts)
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post')
        return padded

    max_length = 100
    X_train_text = tokenize_and_pad(train_df['statement'], tokenizer, max_length)
    X_val_text = tokenize_and_pad(val_df['statement'], tokenizer, max_length)
    X_test_text = tokenize_and_pad(test_df['statement'], tokenizer, max_length)

    # =========================
    # Encode Categorical Features
    # =========================
    # Speaker Encoding
    speaker_encoder = LabelEncoder()
    speaker_encoder.fit(train_df['speaker'])
    save_object(speaker_encoder, SPEAKER_ENCODER_PATH)
    print(f"Speaker encoder saved to {SPEAKER_ENCODER_PATH}")

    # Subject Encoding
    subject_encoder = LabelEncoder()
    subject_encoder.fit(train_df['subject'])
    save_object(subject_encoder, SUBJECT_ENCODER_PATH)
    print(f"Subject encoder saved to {SUBJECT_ENCODER_PATH}")

    # Number of unique speakers and subjects (+1 for 'unknown')
    num_speakers = len(speaker_encoder.classes_) + 1
    num_subjects = len(subject_encoder.classes_) + 1

    # Encode speakers and subjects with unknown handling
    train_speaker = encode_with_unknown(train_df['speaker'], speaker_encoder)
    val_speaker = encode_with_unknown(val_df['speaker'], speaker_encoder)
    test_speaker = encode_with_unknown(test_df['speaker'], speaker_encoder)

    train_subject = encode_with_unknown(train_df['subject'], subject_encoder)
    val_subject = encode_with_unknown(val_df['subject'], subject_encoder)
    test_subject = encode_with_unknown(test_df['subject'], subject_encoder)

    # =========================
    # Prepare Labels
    # =========================
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # =========================
    # Mixed Precision Configuration
    # =========================
    try:
        from tensorflow.keras.mixed_precision import set_global_policy

        set_global_policy('mixed_float16')
        print("Mixed precision set to 'mixed_float16'.")
    except Exception as e:
        print(f"Mixed precision could not be set: {e}")
        print("Proceeding without mixed precision.")

    # =========================
    # Build the CNN Model
    # =========================
    def create_cnn_model(vocab_size, embedding_dim, max_length, num_speakers, num_subjects):
        # Text Input
        text_input = Input(shape=(max_length,), name='text_input')
        embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(text_input)
        dropout = Dropout(0.5)(embedding)
        conv = tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu')(dropout)
        pool = tf.keras.layers.GlobalMaxPooling1D()(conv)
        dense = Dense(128, activation='relu')(pool)
        dropout = Dropout(0.5)(dense)
        
        # Speaker Input
        speaker_input = Input(shape=(1,), name='speaker_input')
        speaker_embedding = Embedding(input_dim=num_speakers, output_dim=16, input_length=1)(speaker_input)
        speaker_embedding = Flatten()(speaker_embedding)
        
        # Subject Input
        subject_input = Input(shape=(1,), name='subject_input')
        subject_embedding = Embedding(input_dim=num_subjects, output_dim=16, input_length=1)(subject_input)
        subject_embedding = Flatten()(subject_embedding)
        
        # Concatenate All Features
        combined = Concatenate()([dropout, speaker_embedding, subject_embedding])
        combined = Dense(64, activation='relu')(combined)
        combined = Dropout(0.5)(combined)
        output = Dense(1, activation='sigmoid', dtype='float32')(combined)  # Ensure output is float32
        
        model = Model(inputs=[text_input, speaker_input, subject_input], outputs=output)
        return model

    vocab_size = 10000
    embedding_dim = 128

    model = create_cnn_model(vocab_size, embedding_dim, max_length, num_speakers, num_subjects)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # =========================
    # Prepare TensorFlow Datasets
    # =========================
    batch_size = 32  # Adjust as needed based on your GPU memory

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'text_input': X_train_text,
            'speaker_input': train_speaker,
            'subject_input': train_subject
        },
        y_train
    )).shuffle(len(y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'text_input': X_val_text,
            'speaker_input': val_speaker,
            'subject_input': val_subject
        },
        y_val
    )).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'text_input': X_test_text,
            'speaker_input': test_speaker,
            'subject_input': test_subject
        },
        y_test
    )).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # =========================
    # Define Callbacks
    # =========================
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

    # =========================
    # Train the Model
    # =========================
    epochs = 10  # Adjust as needed

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    # =========================
    # Save the Final Model
    # =========================
    model.save(MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")

    # =========================
    # Evaluate on Test Data
    # =========================
    loss, accuracy = model.evaluate(test_dataset)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()
