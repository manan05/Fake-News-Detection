import os
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Concatenate,
    Embedding,
    Flatten,
    Conv1D,
    GlobalMaxPooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def create_directories(dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def save_object(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved object to {filepath}")

def load_data(train_path, val_path, test_path):
    try:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        print("Successfully loaded CSV files.")
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        sys.exit(1)
    
    required_columns = {'statement', 'speaker', 'subject', 'label'}
    for df, name in zip([train_df, val_df, test_df], ['train', 'validation', 'test']):
        if not required_columns.issubset(df.columns):
            print(f"Missing columns in {name} dataset. Required columns: {required_columns}")
            sys.exit(1)
    
    print("All datasets contain required columns.")
    return train_df, val_df, test_df

def encode_labels(train_df, val_df, test_df):
    label_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    
    # Map labels
    for df, name in zip([train_df, val_df, test_df], ['train', 'validation', 'test']):
        df['label'] = df['label'].map(label_map)
        if df['label'].isnull().any():
            print(f"Found unmapped labels in {name} dataset.")
            sys.exit(1)
    print("Labels successfully mapped to binary.")
    return label_map

def encode_categorical_features(train_df, val_df, test_df):
    speaker_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    subject_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    speaker_encoder.fit(train_df[['speaker']])
    subject_encoder.fit(train_df[['subject']])
    
    def encode_and_reshape(df, column, encoder, feature_name):
        encoded = encoder.transform(df[[column]]).astype(int)
        special_index = len(encoder.categories_[0])
        encoded[encoded == -1] = special_index
        reshaped = encoded.reshape(-1, 1)
        print(f"Encoded and reshaped {feature_name} feature.")
        return reshaped
    
    train_speaker = encode_and_reshape(train_df, 'speaker', speaker_encoder, 'train_speaker')
    val_speaker = encode_and_reshape(val_df, 'speaker', speaker_encoder, 'val_speaker')
    test_speaker = encode_and_reshape(test_df, 'speaker', speaker_encoder, 'test_speaker')
    
    train_subject = encode_and_reshape(train_df, 'subject', subject_encoder, 'train_subject')
    val_subject = encode_and_reshape(val_df, 'subject', subject_encoder, 'val_subject')
    test_subject = encode_and_reshape(test_df, 'subject', subject_encoder, 'test_subject')
    num_speakers = len(speaker_encoder.categories_[0]) + 1
    num_subjects = len(subject_encoder.categories_[0]) + 1
    
    print(f"Number of speakers (including 'unknown'): {num_speakers}")
    print(f"Number of subjects (including 'unknown'): {num_subjects}")
    
    return (train_speaker, val_speaker, test_speaker), (train_subject, val_subject, test_subject), (speaker_encoder, subject_encoder)

def tokenize_and_pad(train_texts, val_texts, test_texts, num_words=10000, max_length=100):
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    print("Tokenizer fitted on training texts.")
    
    X_train = tokenizer.texts_to_sequences(train_texts)
    X_val = tokenizer.texts_to_sequences(val_texts)
    X_test = tokenizer.texts_to_sequences(test_texts)
    
    X_train_padded = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')
    X_val_padded = pad_sequences(X_val, maxlen=max_length, padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')
    
    print("Text data tokenized and padded.")
    return (X_train_padded, X_val_padded, X_test_padded), tokenizer

def prepare_datasets(X_text, X_speaker, X_subject, y, batch_size=32, shuffle=False):
    inputs = {
        'text_input': X_text,
        'speaker_input': X_speaker,
        'subject_input': X_subject
    }
    if y is not None:
        dataset = tf.data.Dataset.from_tensor_slices((inputs, y))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def build_model(vocab_size, embedding_dim, max_length, num_speakers, num_subjects):
    text_input = Input(shape=(max_length,), name='text_input')
    embedding_text = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(text_input)
    dropout_text = Dropout(0.5)(embedding_text)
    conv = Conv1D(filters=128, kernel_size=5, activation='relu')(dropout_text)
    pool = GlobalMaxPooling1D()(conv)
    dense_text = Dense(128, activation='relu')(pool)
    dropout_text_dense = Dropout(0.5)(dense_text)
    
    speaker_input = Input(shape=(1,), name='speaker_input')
    embedding_speaker = Embedding(input_dim=num_speakers, output_dim=16)(speaker_input)
    flatten_speaker = Flatten()(embedding_speaker)
    
    subject_input = Input(shape=(1,), name='subject_input')
    embedding_subject = Embedding(input_dim=num_subjects, output_dim=16)(subject_input)
    flatten_subject = Flatten()(embedding_subject)
    
    concatenated = Concatenate()([dropout_text_dense, flatten_speaker, flatten_subject])
    dense = Dense(64, activation='relu')(concatenated)
    dropout = Dropout(0.5)(dense)
    output = Dense(1, activation='sigmoid')(dropout)
    
    model = Model(inputs=[text_input, speaker_input, subject_input], outputs=output)
    return model

def get_labels_and_predictions(dataset, model):
    true_labels = []
    predictions = []
    
    for batch in dataset:
        inputs, labels = batch
        preds = model.predict(inputs)
        preds = (preds > 0.5).astype(int)
        true_labels.extend(labels.numpy())
        predictions.extend(preds.flatten())
    
    return np.array(true_labels), np.array(predictions)

def main():
    DATA_DIR = '../data'  # Ensure your data is in this directory
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

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"{len(gpus)} Physical GPU(s) found and memory growth set.")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
            sys.exit(1)
    else:
        print("No GPU found. Exiting.")
        sys.exit(1)
    
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        print("Successfully downloaded NLTK dependencies.")
    except Exception as e:
        print(f"Error downloading NLTK dependencies: {e}")
        sys.exit(1)
    
    train_df, val_df, test_df = load_data(TRAIN_PATH, VAL_PATH, TEST_PATH)
    
    label_map = encode_labels(train_df, val_df, test_df)
    
    (train_speaker, val_speaker, test_speaker), (train_subject, val_subject, test_subject), (speaker_encoder, subject_encoder) = encode_categorical_features(train_df, val_df, test_df)
    
    save_object(speaker_encoder, SPEAKER_ENCODER_PATH)
    save_object(subject_encoder, SUBJECT_ENCODER_PATH)
    save_object(label_map, LABEL_MAP_PATH)
    
    # Tokenize and pad text data
    (X_train_padded, X_val_padded, X_test_padded), tokenizer = tokenize_and_pad(
        train_df['statement'],
        val_df['statement'],
        test_df['statement'],
        num_words=10000,
        max_length=100
    )
    
    # Save tokenizer
    save_object(tokenizer, TOKENIZER_SAVE_PATH)
    
    # Prepare labels
    y_train = train_df['label'].values.astype(np.float32)
    y_val = val_df['label'].values.astype(np.float32)
    y_test = test_df['label'].values.astype(np.float32)
    
    vocab_size = min(len(tokenizer.word_index) + 1, 10000)
    embedding_dim = 128
    num_speakers = len(speaker_encoder.categories_[0]) + 1  # +1 for 'unknown'
    num_subjects = len(subject_encoder.categories_[0]) + 1  # +1 for 'unknown'
    
    model = build_model(vocab_size, embedding_dim, max_length=100, num_speakers=num_speakers, num_subjects=num_subjects)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    print("Model compiled successfully.")
    model.summary()
    
    batch_size = 32
    
    train_dataset = prepare_datasets(X_train_padded, train_speaker, train_subject, y_train, batch_size=batch_size, shuffle=True)
    val_dataset = prepare_datasets(X_val_padded, val_speaker, val_subject, y_val, batch_size=batch_size, shuffle=False)
    test_dataset = prepare_datasets(X_test_padded, test_speaker, test_subject, y_test, batch_size=batch_size, shuffle=False)
    
    print("TensorFlow datasets prepared.")
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    epochs = 15  # Reduced epochs for quicker training; adjust as needed
    
    print("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    model.save(MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")
    
    def evaluate_and_print(dataset, dataset_name):
        print(f"\nEvaluating on {dataset_name} Data...")
        loss, accuracy = model.evaluate(dataset)
        print(f"{dataset_name} Loss: {loss:.4f}")
        print(f"{dataset_name} Accuracy: {accuracy * 100:.2f}%")
        
        true_labels, predictions = get_labels_and_predictions(dataset, model)
        
        conf_matrix = confusion_matrix(true_labels, predictions)
        class_report = classification_report(true_labels, predictions, target_names=['class_0', 'class_1'])
        
        print(f"{dataset_name} Confusion Matrix:")
        print(conf_matrix)
        print(f"{dataset_name} Classification Report:")
        print(class_report)
    

    evaluate_and_print(val_dataset, "Validation")
    evaluate_and_print(test_dataset, "Test")

if __name__ == '__main__':
    main()
