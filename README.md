IMDB Sentiment Analysis - Ensemble Learning Approach
This project builds a Sentiment Analysis model using the IMDB Dataset of 50K Movie Reviews from Kaggle. The goal is to classify movie reviews as either Positive or Negative by leveraging Natural Language Processing (NLP) techniques and Ensemble Learning algorithms to maximize prediction accuracy.

📋 Overview
The core idea is to move beyond simple models by combining multiple weak learners into a strong learner. This project demonstrates the end-to-end pipeline from raw text processing to deploying an ensemble model.

Key Features & Techniques
 Data Preprocessing:
    Text Cleaning: Removal of HTML tags, special characters, and conversion to lowercase.
    Normalization: Tokenization, Stopwords removal, and Lemmatization.
Feature Engineering:
    CountVectorizer (Bag of Words).
    TfidfVectorizer (Term Frequency-Inverse Document Frequency).
Models:
    Decision Tree.
    Bagging: Random Forest Classifier.
    Boosting: XGBoost, AdaBoost, and Gradient Boosting.