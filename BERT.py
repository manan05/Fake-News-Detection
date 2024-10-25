import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.callbacks import EarlyStopping

# Load the data
def load_data(file_path):
    return pd.read_csv(file_path)

# Build the BERT model
def build_bert_model(max_length):
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    
    input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
    
    # Pass inputs directly as a dictionary to the model
    bert_output = bert_model({'input_ids': input_ids, 'attention_mask': attention_mask})[1]
    
    dense_layer = tf.keras.layers.Dense(32, activation='relu')(bert_output)
    dropout_layer = tf.keras.layers.Dropout(0.5)(dense_layer)
    
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(dropout_layer)
    
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output_layer)
    
    return model

def main():
    # Load the data
    train_df = load_data('data/train.csv')
    val_df = load_data('data/valid.csv')
    test_df = load_data('data/test.csv')

    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize and encode sequences in each dataset
    def encode_texts(texts, max_length):
        return tokenizer(
            texts.tolist(),
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'  # Make sure this returns TensorFlow tensors
        )

    max_length = 100

    # Use 'statement' column directly for encoding
    X_train_encoded = encode_texts(train_df['statement'], max_length)
    X_val_encoded = encode_texts(val_df['statement'], max_length)
    X_test_encoded = encode_texts(test_df['statement'], max_length)

    # Extract labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # Build and compile the BERT model
    model = build_bert_model(max_length)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Display model architecture
    model.summary()

    # Training with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    model.fit(
        {'input_ids': tf.convert_to_tensor(X_train_encoded['input_ids']), 
         'attention_mask': tf.convert_to_tensor(X_train_encoded['attention_mask'])},
        y_train,
        validation_data=({'input_ids': tf.convert_to_tensor(X_val_encoded['input_ids']), 
                          'attention_mask': tf.convert_to_tensor(X_val_encoded['attention_mask'])}, y_val),
        epochs=10,
        batch_size=16,
        callbacks=[early_stopping]
    )

    # Evaluate the model on test data
    loss, accuracy = model.evaluate({'input_ids': tf.convert_to_tensor(X_test_encoded['input_ids']), 
                                      'attention_mask': tf.convert_to_tensor(X_test_encoded['attention_mask'])}, 
                                     y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()
