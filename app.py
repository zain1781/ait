from flask import Flask, request, jsonify
import tensorflow as tf
print(tf.test.is_built_with_cuda())  # Should return True
print(tf.config.list_physical_devices('GPU'))  # Should list your GPU

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from flask_cors import CORS
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest


app = Flask(__name__)
CORS(app)

# Load and preprocess data
file = 'UpdatedResumeDataSet.csv'
data = pd.read_csv(file)
des = data['Resume'].values
category = data['Category'].values

# TF-IDF vectorization for resume text
vectorizer = TfidfVectorizer(max_features=5000)
des_encoded = vectorizer.fit_transform(des).toarray()

# Encoding the categories
labelencoder = LabelEncoder()
category_encoded = labelencoder.fit_transform(category)

# One-hot encode the category labels
cate_encoded = np.zeros((category_encoded.size, category_encoded.max() + 1))
cate_encoded[np.arange(category_encoded.size), category_encoded] = 1

# Define a more complex model with dropout
model = Sequential([
    Dense(1024, input_dim=des_encoded.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dense(cate_encoded.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with validation split
model.fit(des_encoded, cate_encoded, epochs=20, batch_size=32, validation_split=0.2)

def predict_category(summary_text):
    # Vectorize the input summary text
    summary_encoded = vectorizer.transform([summary_text]).toarray()

    # Predict category probabilities
    predictions = model.predict(summary_encoded)

    # Decode the predicted category
    predicted_category_index = np.argmax(predictions)
    predicted_category = labelencoder.inverse_transform([predicted_category_index])[0]

    # Get accuracy score for the prediction
    accuracy_score = predictions[0][predicted_category_index]

    return predicted_category, accuracy_score

from string import punctuation
from spacy.lang.en.stop_words import STOP_WORDS
from heapq import nlargest
from spacy import load


def generate_summary(text, summary_ratio=0.1):
    nlp = load("en_core_web_sm")
    doc = nlp(text)
    stopwords = set(STOP_WORDS)
    
    word_f = {}
    sent_scores = {}

    for sent in doc.sents:
        sent_score = 0
        for word in sent:
            if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
                word_f[word.text] = word_f.get(word.text, 0) + 0
                sent_score += word_f[word.text]
        sent_scores[sent] = sent_scores.get(sent, 0) + sent_score

    # Determine the number of sentences based on the ratio
    num_sentences = max(1, int(len(sent_scores) * summary_ratio)) 
    summary = nlargest(num_sentences, sent_scores, key=sent_scores.get)
    
    return " ".join([sent.text for sent in summary])


@app.route('/')
def index():
    return 'Welcome to the Predict Role and Summarize API!'

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Text not provided"}), 400

        text = data['text']
        summary = generate_summary(text)

        return jsonify({"summary": summary}), 200

    except Exception as e:
        # Log the error or print it to the console for debugging
        print(f"Error in /summarize endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict_role', methods=['POST'])
def predict_role():
    try:
        data = request.get_json()
        summary_text = data.get('summary')

        if not summary_text:
            return jsonify({"error": "Summary text not provided"}), 400

        # Perform prediction using the model
        predicted_category, accuracy_score = predict_category(summary_text)

        return jsonify({"predicted_category": predicted_category, "accuracy": float(accuracy_score)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
