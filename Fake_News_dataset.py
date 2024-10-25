
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
from tabulate import tabulate


# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# For handling text data
from sklearn.feature_extraction.text import TfidfVectorizer


# For displaying images
from sklearn.tree import export_graphviz
from subprocess import call

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Machine learning libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Word Embeddings
from gensim.models import Word2Vec

# For handling sparse matrices
from scipy.sparse import hstack

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')



train_df = pd.read_csv('LIAR2/liar2/train.csv', encoding='utf-8', delimiter=',')
val_df = pd.read_csv('LIAR2/liar2/valid.csv', encoding='utf-8', delimiter=',')
test_df = pd.read_csv('LIAR2/liar2/test.csv', encoding='utf-8', delimiter=',')
pd.set_option('display.max_columns', None)   # Display all columns
pd.set_option('display.max_rows', None)      # Display all rows
pd.set_option('display.width', 1000)         # Adjust display width
pd.set_option('display.precision', 2)        # Set precision for floating-point numbers

data_df = pd.concat([train_df, val_df, test_df], ignore_index=True)


## First 5 Rows
print("First 5 Rows:")
display(data_df.head())

## Data Types
print("\nData Types:")
display(data_df.dtypes.to_frame(name='Data Type'))

## Summary Statistics
print("\nSummary Statistics:")
display(data_df.describe(include='all').transpose())

# Check for missing values
print("\nMissing Values:")
missing_values = data_df.isnull().sum()
missing_df = missing_values[missing_values > 0].to_frame(name='Missing Values')
display(missing_df)

# If you prefer to use tabulate for console output, uncomment the following lines:
print("First 5 Rows:")
print(tabulate(data_df.head(), headers='keys', tablefmt='psql'))

print("\nData Types:")
print(tabulate(data_df.dtypes.to_frame(name='Data Type'), headers='keys', tablefmt='psql'))

print("\nSummary Statistics:")
print(tabulate(data_df.describe(include='all').transpose(), headers='keys', tablefmt='psql'))

print("\nMissing Values:")
print(tabulate(missing_df, headers='keys', tablefmt='psql'))

# List of numerical and categorical features
numerical_features = [
    'true_counts',
    'mostly_true_counts',
    'half_true_counts',
    'mostly_false_counts',
    'false_counts',
    'pants_on_fire_counts'
]

categorical_features = ['label', 'subject', 'speaker', 'state_info']

# Exploratory Data Analysis

## Distribution of Numerical Features
for feature in numerical_features:
    plt.figure(figsize=(10, 4))
    sns.histplot(data_df[feature].dropna(), bins=30, kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

## Boxplots for Numerical Features
for feature in numerical_features:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=data_df[feature].dropna())
    plt.title(f'Boxplot of {feature}')
    plt.xlabel(feature)
    plt.tight_layout()
    plt.show()

## Distribution of Categorical Features
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    # Get the top 20 categories
    order = data_df[feature].value_counts().iloc[:20].index
    sns.countplot(y=feature, data=data_df, order=order)
    plt.title(f'Distribution of {feature}')
    plt.xlabel('Count')
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()

## Label Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=data_df)
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Correlation Matrix
corr_matrix = data_df[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()

# Relationship between Numerical Features and Label
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='label', y=feature, data=data_df)
    plt.title(f'{feature} by Label')
    plt.xlabel('Label')
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()

# Text Length Analysis (if applicable)
if 'statement' in data_df.columns:
    # Calculate text length
    data_df['text_length'] = data_df['statement'].apply(lambda x: len(str(x)))

    # Distribution of Text Length
    plt.figure(figsize=(10, 4))
    sns.histplot(data_df['text_length'], bins=50, kde=True)
    plt.title('Distribution of Text Length')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # Text Length by Label
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='label', y='text_length', data=data_df)
    plt.title('Text Length by Label')
    plt.xlabel('Label')
    plt.ylabel('Text Length')
    plt.tight_layout()
    plt.show()

# Word Cloud of Statements (optional)
try:
    from wordcloud import WordCloud, STOPWORDS

    if 'statement' in data_df.columns:
        text = " ".join(str(statement) for statement in data_df['statement'].dropna())
        wordcloud = WordCloud(
            stopwords=STOPWORDS,
            background_color='white',
            width=800,
            height=400
        ).generate(text)

        plt.figure(figsize=(15, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Statements')
        plt.tight_layout()
        plt.show()
except ImportError:
    print("WordCloud library is not installed. Skipping word cloud visualization.")
    print("You can install it using 'pip install wordcloud'.")


# Import necessary libraries

# Set pandas display options for better readability
pd.set_option('display.max_columns', None)   # Display all columns
pd.set_option('display.width', 1000)         # Adjust display width
pd.set_option('display.precision', 3)        # Set precision for floating-point numbers


# -------------------------
# 1. Identify and Handle Missing Values
# -------------------------

def report_and_handle_missing_values(df):
    print("\n----- Missing Values Report -----\n")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        print("Columns with missing values:")
        display(missing_values.to_frame(name='Missing Values'))

        # Visualize missing values before handling
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title('Heatmap of Missing Values (Before Handling)')
        plt.show()

        # Handling missing values
        # Option 1: Drop rows with missing values
        # df_cleaned = df.dropna()

        # Option 2: Fill missing values with appropriate methods
        df_cleaned = df.copy()
        for col in missing_values.index:
            if df[col].dtype == 'object':
                df_cleaned[col] = df[col].fillna('Unknown')
            else:
                df_cleaned[col] = df[col].fillna(df[col].median())

        print("\nAfter handling missing values:")
        missing_values_after = df_cleaned.isnull().sum()
        display(missing_values_after[missing_values_after > 0].to_frame(name='Missing Values'))

        # Visualize missing values after handling
        plt.figure(figsize=(12, 6))
        sns.heatmap(df_cleaned.isnull(), cbar=False, cmap='viridis')
        plt.title('Heatmap of Missing Values (After Handling)')
        plt.show()

        return df_cleaned
    else:
        print("No missing values found in the dataset.")
        return df

# -------------------------
# 3. Identify and Handle Duplicates
# -------------------------

def report_and_handle_duplicates(df):
    print("\n----- Duplicates Report -----\n")
    duplicate_rows = df[df.duplicated()]
    if not duplicate_rows.empty:
        print(f"Number of duplicate rows: {duplicate_rows.shape[0]}")
        print("\nDuplicate rows (first 5):")
        display(duplicate_rows.head())

        # Handling duplicates by dropping them
        df_cleaned = df.drop_duplicates()
        print(f"\nAfter removing duplicates, new dataset shape: {df_cleaned.shape}")
        return df_cleaned
    else:
        print("No duplicate rows found in the dataset.")
        return df

# -------------------------
# 5. Identify and Correct Data Type Mismatches
# -------------------------

def report_and_correct_datatype_mismatches(df):
    print("\n----- Data Type Mismatches Report -----\n")
    # Expected data types (adjust according to your dataset)
    expected_dtypes = {
        'id': 'int64',
        'label': 'object',
        'statement': 'object',
        'date': 'datetime64[ns]',
        'subject': 'object',
        'speaker': 'object',
        'speaker_description': 'object',
        'state_info': 'object',
        'context': 'object',
        'justification': 'object'
    }

    df_cleaned = df.copy()

    # Correct data types
    for col, expected_dtype in expected_dtypes.items():
        if col in df.columns:
            actual_dtype = df[col].dtype
            if actual_dtype != expected_dtype:
                print(f"Column '{col}' has data type '{actual_dtype}' but expected '{expected_dtype}'. Attempting to convert...")
                try:
                    if expected_dtype == 'datetime64[ns]':
                        df_cleaned[col] = pd.to_datetime(df[col], errors='coerce')
                    else:
                        df_cleaned[col] = df[col].astype(expected_dtype)
                    print(f"Column '{col}' converted to '{expected_dtype}'.")
                except Exception as e:
                    print(f"Failed to convert column '{col}': {e}")
            else:
                print(f"Column '{col}' data type matches expected '{expected_dtype}'.")
        else:
            print(f"Column '{col}' not found in the dataset.")

    # Display data types after correction
    print("\nData types after correction:")
    display(df_cleaned.dtypes.to_frame(name='Data Type'))
    return df_cleaned

# -------------------------
# Run All Reports and Cleaning Steps
# -------------------------

def run_data_cleaning_process(df):
    # Step 1: Handle Missing Values
    df_cleaned = report_and_handle_missing_values(df)

    # Step 3: Handle Duplicates
    df_cleaned = report_and_handle_duplicates(df_cleaned)

    # Step 5: Correct Data Type Mismatches
    df_cleaned = report_and_correct_datatype_mismatches(df_cleaned)

    print("\n----- Data Cleaning Process Completed -----\n")
    print("Final cleaned dataset shape:", df_cleaned.shape)
    return df_cleaned

cleaned_data_df = run_data_cleaning_process(data_df)


# -------------------------
# 1. Select Specified Columns
# -------------------------

columns_to_use = [
    'label',
    'statement',
    'date',
    'subject',
    'speaker',
    'speaker_description',
    'state_info',
    'context',
    'justification'
]
data_df = cleaned_data_df[columns_to_use]

# -------------------------
# 2. Handle Missing Values and Data Cleaning
# -------------------------

# Fill missing values
data_df['subject'] = data_df['subject'].fillna('World')
data_df['context'] = data_df['context'].fillna('Unknown')
data_df['speaker_description'] = data_df['speaker_description'].fillna('No Description')
data_df['state_info'] = data_df['state_info'].fillna('Unknown')

# Convert 'state_info' to lowercase and replace 'national' with 'USA'
data_df['state_info'] = data_df['state_info'].str.lower()
data_df['state_info'] = data_df['state_info'].str.replace('national', 'USA', case=False)

# Remove duplicate rows
data_df = data_df.drop_duplicates()

# -------------------------
# 3. Text Preprocessing Function
# -------------------------


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Apply text preprocessing to all text features
text_features = ['statement', 'speaker_description', 'context', 'justification']
for feature in text_features:
    data_df[feature] = data_df[feature].astype(str).apply(preprocess_text)

# Combine text features for vectorization
data_df['text_combined'] = data_df[text_features].apply(lambda x: ' '.join(x), axis=1)

# -------------------------
# 4. Encode Categorical Variables
# -------------------------

# Encode target variable 'label'
label_encoder = LabelEncoder()
data_df['label_encoded'] = label_encoder.fit_transform(data_df['label'])

# Process date feature
data_df['date'] = pd.to_datetime(data_df['date'], errors='coerce')
data_df['date'] = data_df['date'].fillna(pd.Timestamp('1900-01-01'))

# Extract date features
data_df['year'] = data_df['date'].dt.year
data_df['month'] = data_df['date'].dt.month
data_df['day'] = data_df['date'].dt.day

# Encode categorical features using Label Encoding
categorical_features = ['subject', 'speaker', 'state_info']

for col in categorical_features:
    data_df[col] = data_df[col].astype(str)
    data_df[col] = data_df[col].fillna('Unknown')
    lbl_enc = LabelEncoder()
    data_df[col + '_encoded'] = lbl_enc.fit_transform(data_df[col])

# -------------------------
# 5. Prepare Feature Matrix and Target Vector
# -------------------------

# Features to include
feature_columns = ['year', 'month', 'day', 'subject_encoded', 'speaker_encoded', 'state_info_encoded']
X_numerical = data_df[feature_columns].values

# Target variable
y = data_df['label_encoded']

# -------------------------
# 6. Define a Function to Train, Evaluate Models, and Extract Feature Importances
# -------------------------

def train_evaluate_model(X_text, vectorizer_name, vectorizer=None):
    print(f"\n----- Using {vectorizer_name} -----")
    # Combine numerical and text features
    X = hstack([X_numerical, X_text])

    # Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Evaluate the Model
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy Score:", accuracy)
    print("\nClassification Report:")
    target_names = [str(label) for label in label_encoder.classes_]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Extract and display feature importances (only if vectorizer is provided)
    if vectorizer is not None:
        # Get feature names
        numerical_feature_names = feature_columns
        text_feature_names = vectorizer.get_feature_names_out()
        all_feature_names = numerical_feature_names + list(text_feature_names)

        # Extract feature importances
        importances = rf_classifier.feature_importances_

        # Create DataFrame for feature importances
        feature_importances = pd.DataFrame({
            'feature': all_feature_names,
            'importance': importances
        })

        # Sort features by importance
        feature_importances = feature_importances.sort_values(by='importance', ascending=False)

        # Display top 20 features
        print("\nTop 20 Most Important Features:")
        print(feature_importances.head(20))

        # Visualize the top 20 features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
        plt.title(f'Top 20 Features - {vectorizer_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

    return accuracy

# -------------------------
# 7. Vectorization Techniques and Performance Comparison
# -------------------------

# Initialize an empty dictionary to store accuracies
vectorizer_performance = {}

# --- Technique 1: TF-IDF Vectorization ---
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
text_tfidf = tfidf_vectorizer.fit_transform(data_df['text_combined'])

accuracy_tfidf = train_evaluate_model(text_tfidf, "TF-IDF Vectorization", vectorizer=tfidf_vectorizer)
vectorizer_performance['TF-IDF'] = accuracy_tfidf

# --- Technique 2: Count Vectorization with n-grams ---
count_vectorizer = CountVectorizer(max_features=5000, ngram_range=(1,2))
text_counts = count_vectorizer.fit_transform(data_df['text_combined'])

accuracy_count = train_evaluate_model(text_counts, "Count Vectorization with n-grams", vectorizer=count_vectorizer)
vectorizer_performance['CountVectorizer'] = accuracy_count

# --- Technique 3: Word Embeddings using Word2Vec ---

# Prepare data for Word2Vec
sentences = data_df['text_combined'].apply(lambda x: x.split())

# Train Word2Vec model
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)

# Function to get average Word2Vec embeddings for each document
def get_word2vec_embeddings(sentences, model, num_features):
    embeddings = np.zeros((len(sentences), num_features))
    index2word_set = set(model.wv.index_to_key)
    for i, sentence in enumerate(sentences):
        nwords = 0
        feature_vec = np.zeros((num_features,), dtype='float32')
        for word in sentence:
            if word in index2word_set:
                nwords += 1
                feature_vec = np.add(feature_vec, model.wv[word])
        if nwords > 0:
            embeddings[i] = feature_vec / nwords
    return embeddings

# Get embeddings
text_embeddings = get_word2vec_embeddings(sentences, word2vec_model, 100)

# Convert numerical features to sparse matrix for compatibility
from scipy import sparse
X_numerical_sparse = sparse.csr_matrix(X_numerical)

# Combine numerical and embedding features
X_embeddings = np.hstack((X_numerical_sparse.toarray(), text_embeddings))

# Split Data into Training and Testing Sets
X_train_emb, X_test_emb, y_train_emb, y_test_emb = train_test_split(X_embeddings, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
print("\n----- Using Word2Vec Embeddings -----")
rf_classifier_emb = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_emb.fit(X_train_emb, y_train_emb)

# Evaluate the Model
y_pred_emb = rf_classifier_emb.predict(X_test_emb)
accuracy_emb = accuracy_score(y_test_emb, y_pred_emb)
print("Accuracy Score:", accuracy_emb)
print("\nClassification Report:")
target_names = [str(label) for label in label_encoder.classes_]
print(classification_report(y_test_emb, y_pred_emb, target_names=target_names))

vectorizer_performance['Word2Vec'] = accuracy_emb

# Note: Feature importances for Word2Vec embeddings are not directly interpretable

# -------------------------
# 8. Compare Vectorization Techniques
# -------------------------

# Print the performance comparison
print("\n----- Performance Comparison -----")
for vectorizer, accuracy in vectorizer_performance.items():
    print(f"{vectorizer}: {accuracy:.4f}")

# Visualize the performance comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=list(vectorizer_performance.keys()), y=list(vectorizer_performance.values()))
plt.title('Comparison of Vectorization Techniques')
plt.ylabel('Accuracy Score')
plt.xlabel('Vectorization Technique')
plt.tight_layout()
plt.show()
