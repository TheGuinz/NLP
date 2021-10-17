import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stop_words]
    return " ".join(tokens_without_sw)


def stemming(text):
    ps = PorterStemmer()
    words = word_tokenize(text)
    stem_sentence = []
    for word in words:
        stem_sentence.append(ps.stem(word))
        stem_sentence.append(" ")
    return " ".join(stem_sentence)


def preprocessing(text, stopwords=False):
    text = remove_punctuation(text)
    text = stemming(text)
    if stopwords:
        text = remove_stopwords(text)
    return text


def split_by_space(text):
    # Split by space
    return text.split()
