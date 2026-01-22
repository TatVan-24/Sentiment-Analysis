import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.txt_process import processing_txt
from src.data_split import split
from src.vectorize import vectorize
from src.XG_Boosing import xg_boost

label = ['positive', 'negative']
data_path = r'D:\merged_partition_content\Tu_Hoc\AIO2025\MyPoject\Module03\Sentiment Analysis\data\raw\IMDB-Dataset.csv'
df = pd.read_csv(data_path)
df.drop_duplicates()

df["review"] = df["review"].apply(processing_txt)
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

X_train, X_test, y_train, y_test = split(df['sentiment'], df['review'], 0.2, 42)
X_train_encoder, X_test_encoder = vectorize(X_train, y_train, X_test)

model = xg_boost()
model.fit(X_train_encoder,y_train )

joblib.dump(model, r"D:\merged_partition_content\Tu_Hoc\AIO2025\MyPoject\Module03\Sentiment Analysis\model\imdb_xgb_lib_model.pkl")
