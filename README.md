This project builds a sentiment analysis system for movie reviews, using the IMDB Dataset from Kaggle The goal is to classify reviews as positive or negative based on their text content.
The main model used is XGBoost, a powerful ensemble learning algorithm, combined with TF-IDF techniques to represent text data.

- XGBoost is an algorithm belonging to the ensemble learning group, specifically boosting, in which:
    + Multiple decision trees are trained sequentially.
    + Each new tree focuses on learning from the errors of the previous tree.
    + The trees are combined to create a highly accurate model.
    Why choose XGBoost?
    + Very effective with large datasets.
    + Good overfitting prevention thanks to regularization.
    + Fast training speed.

- Text Processing Techniques:
    + Vectorization: Text data cannot be directly fed into a machine learning model, so it is necessary to convert the text into a numerical (vector) form.
    + TF-IDF (Term Frequency – Inverse Document Frequency): TF-IDF measures the importance of a word in a document compared to the entire dataset.
    Reduces the influence of words that appear too frequently (the, is, and, ...)
        - Emphasizes words with high categorical significance
        - Compatible with traditional ML models like XGBoost
        - Lightweight, fast, and easier to implement than deep embedding (Word2Vec, BERT)

- Data:
    IMDB Dataset of 50K Movie Reviews: https://www.kaggle.com/datasets/lakshmi25npathiimdb-dataset-of-50k-movie-reviews
     
- Folder Structure:
    Sentiment Analysis
    |
    |- app/
    |   |_ main.py
    |
    |- data/
    |   |_ artifacts
    |   |_ processed
    |   |_ raw/
    |       |_ IMDB-Dataset.csv
    |
    |- model/
    |   |_ imdb_xgb_lib_mdel.pkl
    |   |_ tfidf_vectorzer.pkl
    |
    |- notebooks/
    |   |_ notebook.ipynb
    |   |_ Sentiment Analysis.ipynb
    |
    |- src/
    |   |_ __init__.py
    |   |_ data_split.py
    |   |_ txt_process.py
    |   |_ vectorize.py
    |   |_ XG_Boosing.py
    |
    |- app.py
    |- train.py
    |- README.md
