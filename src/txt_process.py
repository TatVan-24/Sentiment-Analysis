## file này dùng để lọc các kí tự trong dataset!
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import contractions
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('wordnet')

stop = set(stopwords.words('english'))

def expand_contractions(txt):
    return contractions.fix(txt)

def processing_txt(txt):
    wl = WordNetLemmatizer()
    soup = BeautifulSoup(txt, "html.parser")
    txt = soup.get_text()
    txt = expand_contractions(txt)
    emoji_clean = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    txt = emoji_clean.sub(r"", txt)
    txt = re.sub(r'\.(?=\S)', '. ', txt)
    txt = re.sub(r'http\S+', ' ', txt)
    txt = "".join([char.lower() for char in txt if char not in string.punctuation])
    tokens = [wl.lemmatize(word) for word in txt.split() if word not in stop and word.isalpha()]
    txt = " ".join(tokens)
    return txt
