import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Directly load the selected dataset file '/content/twitter_sentiment_test.csv'
dataset = pd.read_csv(r'C:\Users\Dell\Downloads\twitter_sentiment_test.csv')

column_mapping = {
    'OriginalTweet': 'text',
    'Sentiment': 'sentiment'
}
dataset = dataset.rename(columns=column_mapping)

# Split into train and test sets automatically for downstream code compatibility
train, test = train_test_split(dataset, test_size=0.2, random_state=42)

# Handle missing values in text column
train['text'] = train['text'].fillna('')
test['text'] = test['text'].fillna('')

# Task 1: Clean the Text Data
def clean_text(text):
    text = text.lower()                                  # Lowercasing
    text = re.sub(r'https?://\S+|www\.\S+', '', text)     # Remove URLs
    text = re.sub(r'<.*?>', '', text)                    # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)              # Remove punctuation & numbers
    text = re.sub(r'\s+', ' ', text).strip()             # Remove extra spaces
    return text

train['clean_text'] = train['text'].apply(clean_text)
test['clean_text'] = test['text'].apply(clean_text)

# Map target labels to integers (including options like Extremely Positive/Negative)
sentiment_map = {
    'negative': 0, 'Extremely Negative': 0,
    'neutral': 1, 'Neutral': 1,
    'positive': 2, 'Extremely Positive': 2, 'Positive': 2, 'Negative': 0
}
train['label'] = train['sentiment'].map(sentiment_map)

# Handle any unmapped or missing sentiments
train['label'] = train['label'].fillna(1).astype(int)

# Task 2 & 3: Tokenization & Sequence Preparation

VOCAB_SIZE = 10000
MAX_LEN = 50
OOV_TOKEN = "<OOV>"

tok = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tok.fit_on_texts(train['clean_text'])

# Convert text to sequences
X_sequences = tok.texts_to_sequences(train['clean_text'])
X_padded = pad_sequences(X_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

y = train['label'].values

# Split into Training and Validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_padded, y, test_size=0.2, random_state=42, stratify=y
)

# Task 4: Build Neural Network Model

EMBEDDING_DIM = 64

model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: negative, neutral, positive
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# Task 5: Train the Model

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=16,
    verbose=1
)

# Task 6: Evaluate Performance

val_preds = np.argmax(model.predict(X_val), axis=1)

print("\n--- Evaluation Report ---")
print(classification_report(y_val, val_preds, target_names=['negative', 'neutral', 'positive'], labels=[0, 1, 2]))

# Task 7 & 8: Correct vs. Incorrect Predictions

results_df = pd.DataFrame({
    'Text': train['clean_text'].iloc[y_train.shape[0]:y_train.shape[0] + y_val.shape[0]].values,
    'Actual': y_val,
    'Predicted': val_preds
})

inv_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
results_df['Actual_Label'] = results_df['Actual'].map(inv_map)
results_df['Pred_Label'] = results_df['Predicted'].map(inv_map)

print("\n--- Examples of Correct Predictions ---")
print(results_df[results_df['Actual'] == results_df['Predicted']].head(3)[['Text', 'Actual_Label', 'Pred_Label']])
print("\n--- Examples of Incorrect Predictions ---")
print(results_df[results_df['Actual'] != results_df['Predicted']].head(3)[['Text', 'Actual_Label', 'Pred_Label']])