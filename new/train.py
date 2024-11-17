
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertTokenizerFast
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import warnings

# ----------------------------
# Suppress Specific Warnings (Optional)
# ----------------------------
warnings.filterwarnings("ignore", message="Skipping full serialization of Keras layer .* because it is not built.")

# ----------------------------
# Enable Mixed Precision Training
# ----------------------------
from tensorflow.keras import mixed_precision

# Enable mixed precision for faster computation and reduced memory usage
mixed_precision.set_global_policy('mixed_float16')

# ----------------------------
# GPU Configuration
# ----------------------------
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Enable memory growth to allocate GPU memory as needed
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("TensorFlow is using the GPU")
    except Exception as e:
        print(f"Error setting up GPU: {e}")
else:
    print("No GPU found, using CPU")

# ----------------------------
# Constants and Paths
# ----------------------------
DATA_DIR = './data'  # Directory containing train.csv and validate.csv
MAX_SEQ_LENGTH = 16  # Further reduced to manage memory
BATCH_SIZE = 2       # Further reduced to prevent OOM
EPOCHS = 10          # Adjust as needed
LEARNING_RATE = 3e-5 # Adjusted learning rate for fine-tuning
MODEL_SAVE_PATH = './saved_model'

# Create the saved model directory if it doesn't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# ----------------------------
# Load Data
# ----------------------------
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
val_df = pd.read_csv(os.path.join(DATA_DIR, 'validate.csv'))

# Print DataFrame columns to verify available features
print("Available columns in training data:", train_df.columns.tolist())
print("Available columns in validation data:", val_df.columns.tolist())

# ----------------------------
# Preprocess Labels
# ----------------------------
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df['label'])
val_labels = label_encoder.transform(val_df['label'])
num_classes = len(label_encoder.classes_)

# Save label encoder classes for future use
np.save(os.path.join(MODEL_SAVE_PATH, 'label_classes.npy'), label_encoder.classes_)

# ----------------------------
# Initialize Tokenizer
# ----------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# ----------------------------
# Function to Tokenize and Encode Texts
# ----------------------------
def encode_texts(texts):
    return tokenizer(
        texts.tolist(),
        padding='max_length',
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors='tf'
    )

# Encode texts
train_texts = encode_texts(train_df['statement'])
val_texts = encode_texts(val_df['statement'])

# ----------------------------
# Function to Create TensorFlow Datasets
# ----------------------------
def create_dataset(texts, labels=None, is_training=True):
    inputs = {
        'input_ids': texts['input_ids'],
        'attention_mask': texts['attention_mask'],
    }

    if labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(inputs)

    if is_training:
        dataset = dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return dataset

# Create TensorFlow datasets
train_dataset = create_dataset(train_texts, train_labels, is_training=True)
val_dataset = create_dataset(val_texts, val_labels, is_training=False)

# ----------------------------
# Build the Model
# ----------------------------
def create_model():
    # Text Inputs
    input_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32, name='attention_mask')

    # DistilBERT Model
    distilbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    distilbert_model.trainable = False  # Freeze all DistilBERT layers

    distilbert_output = distilbert_model(input_ids, attention_mask=attention_mask)
    sequence_output = distilbert_output.last_hidden_state  # (batch_size, seq_length, hidden_size)

    # Text CNN - Simplified
    conv = tf.keras.layers.Conv1D(
        filters=1,          # Single filter to keep parameters minimal
        kernel_size=1,      # Kernel size of 1
        activation='relu'
    )(sequence_output)
    pool = tf.keras.layers.GlobalMaxPooling1D()(conv)
    text_cnn_output = pool  # Shape: (batch_size, 1)

    # Dropout Layer
    dropout = tf.keras.layers.Dropout(0.3)(text_cnn_output)

    # Output Layer
    output = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(dropout)  # Cast back to float32

    # Create Model
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    return model

# Instantiate and Compile the Model
model = create_model()

# Use mixed precision optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
optimizer = mixed_precision.LossScaleOptimizer(optimizer)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print Model Summary to Verify Architecture
model.summary()

# ----------------------------
# Calculate Class Weights
# ----------------------------
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

# ----------------------------
# Define Callbacks
# ----------------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_PATH, 'best_model'),
        save_best_only=True,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=1,
        restore_best_weights=True
    ),
    # Optional: TensorBoard Callback for Monitoring
    tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
]

# ----------------------------
# Perform a Dummy Forward Pass to Build All Layers
# ----------------------------
print("Performing a dummy forward pass to build all layers...")
for inputs, _ in train_dataset.take(1):
    model(inputs)
print("Dummy forward pass complete.")

# ----------------------------
# Train the Model
# ----------------------------
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ----------------------------
# Save the Final Model
# ----------------------------
model.save(os.path.join(MODEL_SAVE_PATH, 'final_model'))
print("Model training complete and saved.")
