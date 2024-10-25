import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Ensure nltk stopwords are downloaded
import nltk
nltk.download('stopwords')
nltk.download('punkt')

def clean_text(text):
    # Lowercase conversion
    text = text.lower()
    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Removing numbers
    text = re.sub(r'\d+', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Removing stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Joining back to a single string
    return ' '.join(tokens)

def process_and_save(input_path, output_path):
    # Read the data
    df = pd.read_csv(input_path, encoding='utf-8')
    
    # Clean the 'text' column (replace 'text' with the actual column name in your dataset)
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Save the cleaned data to a new CSV file
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == '__main__':
    # Specify the dataset file path
    input_file = 'path_to_your_dataset.csv'
    output_file = 'cleaned_dataset.csv'
    
    # Run the cleaning process
    process_and_save(input_file, output_file)
