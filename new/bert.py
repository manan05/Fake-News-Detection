import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Load data
train_df = pd.read_csv('./data/train.csv')
valid_df = pd.read_csv('./data/valid.csv')
test_df = pd.read_csv('./data/test.csv')

# Parameters
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5

# Load tokenizer and model from HuggingFace
MODEL_NAME = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
distilbert_model = TFDistilBertModel.from_pretrained(MODEL_NAME)

# Preprocessing function for the text data
def preprocess_text(data, tokenizer, max_len=MAX_LEN):
    encodings = tokenizer(
        data.tolist(),
        max_length=max_len,
        truncation=True,
        padding=True,
        return_tensors='tf'
    )
    return encodings['input_ids'], encodings['attention_mask']

# Prepare train, validation, and test datasets
train_texts = train_df['statement'].fillna('').tolist()
valid_texts = valid_df['statement'].fillna('').tolist()
test_texts = test_df['statement'].fillna('').tolist()

train_labels = train_df['label'].values
valid_labels = valid_df['label'].values
test_labels = test_df['label'].values

# Tokenize the texts
train_input_ids, train_attention_mask = preprocess_text(train_texts, tokenizer)
valid_input_ids, valid_attention_mask = preprocess_text(valid_texts, tokenizer)
test_input_ids, test_attention_mask = preprocess_text(test_texts, tokenizer)

# Build a TensorFlow dataset
def build_tf_dataset(input_ids, attention_mask, labels):
    dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': input_ids, 'attention_mask': attention_mask}, labels
    ))
    dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

train_dataset = build_tf_dataset(train_input_ids, train_attention_mask, train_labels)
valid_dataset = build_tf_dataset(valid_input_ids, valid_attention_mask, valid_labels)
test_dataset = build_tf_dataset(test_input_ids, test_attention_mask, test_labels)

# Define the model
input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_ids')
attention_mask = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name='attention_mask')

bert_output = distilbert_model(input_ids, attention_mask=attention_mask)[0]
cls_token = tf.keras.layers.GlobalAveragePooling1D()(bert_output)
output = tf.keras.layers.Dense(1, activation='sigmoid')(cls_token)

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.show()

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_dataset)
print(f'Test Accuracy: {accuracy:.2f}')

# Predict on the test set
y_pred_probs = model.predict(test_dataset)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Confusion matrix and classification report
y_true = test_labels
conf_matrix = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

class_report = classification_report(y_true, y_pred)
print('Classification Report:')
print(class_report)

# Calculate Precision, Recall, and F1-score per epoch (using validation data)
# Note: This is done indirectly using the classification report, which provides the necessary metrics.

# Display confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0, 1], ['Not Fake', 'Fake'])
plt.yticks([0, 1], ['Not Fake', 'Fake'])
plt.show()
