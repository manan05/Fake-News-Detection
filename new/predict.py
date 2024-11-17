# predict.py

import os
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizerFast
from sklearn.preprocessing import LabelEncoder

# Constants
DATA_DIR = './data'
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 16
MODEL_SAVE_PATH = './saved_model'

# Load test data
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Load the saved model
model = tf.keras.models.load_model(MODEL_SAVE_PATH)

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Encode texts
def encode_texts(texts):
    return tokenizer(
        texts.tolist(),
        padding='max_length',
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors='tf'
    )

test_texts = encode_texts(test_df['statement'])

# Prepare metadata features
metadata_cols = ['speaker', 'speaker_job_title', 'state_info', 'party_affiliation']

# Load label encoders for metadata
# Assuming encoders are saved as pickle files during training
import joblib

metadata_encoders = {}
for col in metadata_cols:
    encoder_path = os.path.join(MODEL_SAVE_PATH, f'{col}_encoder.pkl')
    metadata_encoders[col] = joblib.load(encoder_path)

# Encode metadata
for col in metadata_cols:
    test_df[col] = test_df[col].astype(str)
    test_df[col] = metadata_encoders[col].transform(test_df[col])

test_metadata = {col: tf.convert_to_tensor(test_df[col].values) for col in metadata_cols}

# Prepare dataset
test_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': test_texts['input_ids'],
        'attention_mask': test_texts['attention_mask'],
        **test_metadata
    }
))

test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Make predictions
predictions = model.predict(test_dataset)
predicted_labels = tf.argmax(predictions, axis=1).numpy()

# Load label encoder for labels
label_encoder = LabelEncoder()
label_encoder.classes_ = joblib.load(os.path.join(MODEL_SAVE_PATH, 'label_classes.npy'))

# Decode predicted labels
decoded_labels = label_encoder.inverse_transform(predicted_labels)

# Save predictions
output_df = test_df.copy()
output_df['predicted_label'] = decoded_labels
output_df.to_csv(os.path.join(DATA_DIR, 'test_predictions.csv'), index=False)

print("Predictions saved to 'test_predictions.csv' in the data directory.")
