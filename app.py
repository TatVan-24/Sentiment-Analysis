import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import contractions

st.set_page_config(
    page_title="Movie Sentiment Analysis Demo",
    page_icon="🎬",
    layout="centered"
)

@st.cache_resource
def load_resources():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
    vectorizer      = joblib.load(r"D:\merged_partition_content\Tu_Hoc\AIO2025\MyPoject\Module03\Sentiment Analysis\model\tfidf_vectorizer.pkl")
    XGB_model       = joblib.load(r'D:\merged_partition_content\Tu_Hoc\AIO2025\MyPoject\Module03\Sentiment Analysis\model\imdb_xgb_lib_model.pkl')
    # GB_model        = joblib.load(r'gb_model.pkl') 
    # RF_model        = joblib.load(r'rf_model.pkl')
    # DT_model        = joblib.load(r'dtree_model.pkl') 
    # ADA_model       = joblib.load(r'ADA_model.pkl')   
    
    return vectorizer, XGB_model
try:
    vectorizer, XGB_model = load_resources()
    
    stop = set(stopwords.words('english'))
    wl = WordNetLemmatizer()
except FileNotFoundError as e:
    st.error(f"ERROR: No model data found: {e}")
    st.stop()

def expand_contractions(txt):
    return contractions.fix(txt)

def preprocessing_text(txt):
    wl = WordNetLemmatizer()
    soup = BeautifulSoup(txt, "html.parser")
    txt = soup.get_text()
    txt = expand_contractions(txt)
    emoji_clean = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
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

st.title("🎬 AI Movie Critic")
st.markdown("### Movie Analysis ")

st.sidebar.title("⚙️ Settings")
st.sidebar.write("Algorithm Options :")

model_options = {
    "XGBoost (Default)": XGB_model,
    # "Random Forest": RF_model,
    # "Gradient Boosting": GB_model,
    # "Decision Tree": DT_model,
    # "AdaBoost": ADA_model
}

selected_model_name = st.sidebar.selectbox(
    "Models List:",
    list(model_options.keys()) 
)

current_model = model_options[selected_model_name]

st.sidebar.success(f"Current Model: **{selected_model_name}**")
st.write("---")

user_input = st.text_area("Please enter the comment :", height=150, 
                          placeholder="Ex: This movie is a masterpiece! The plot is amazing...")

if st.button(" Analyze"):
    if user_input.strip() == "":
        st.warning("Plese leave a comment!")
    else:
        with st.spinner(f"🤖 {selected_model_name} ..........."):
            processed_text = preprocessing_text(user_input)
            
            vec_input = vectorizer.transform([processed_text])
            
            prediction = current_model.predict(vec_input)[0]
            
            try:
                probability = current_model.predict_proba(vec_input).max() * 100
            except:
                probability = 0.0 # Nếu model không hỗ trợ tính xác suất

            st.write("---")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(label="Confidence", value=f"{probability:.1f}%")

            with col2:
                if prediction == 1:
                    st.success(f"🌟 **POSITIVE**\n\nModel **{selected_model_name}** evaluates this as a compliment.")
                    st.balloons()
                else:
                    st.error(f"💀 **NEGATIVE**\n\nModel **{selected_model_name}** evaluates this as a criticism.")