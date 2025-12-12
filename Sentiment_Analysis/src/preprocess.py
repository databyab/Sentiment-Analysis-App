import re
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


try:
    stop_words = set(stopwords.words("english"))
except:
    import nltk
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    words = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return " ".join(words)
