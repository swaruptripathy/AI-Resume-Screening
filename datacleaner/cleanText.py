import re
import spacy

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Define english stopwords
stop_words = stopwords.words('english')

# load the spacy module and create an object
nlp = spacy.load('en_core_web_sm')


def remove_stopwords(text, stopwords=stop_words, optional_params=False, optional_words=[]):
    if optional_params:
        stopwords.append([a for a in optional_words])
    return [word for word in text if word not in stopwords]


def remove_punctuations(text):
    # Remove punctuations from the text
    text = re.sub(r'[^\w\s]', '', text)
    return word_tokenize(text)


def lemmatize(text):
    # Get the base words from the text
    text_str = nlp(" ".join(text))
    lemmatize_str = []
    for word in text_str:
        lemmatize_str.append(word.lemma_)
    return lemmatize_str


def remove_tags(text, postags=['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV']):
    # Take Tags and eliminate the rest of the words based on Part of Speech(POS).
    filtered = []
    text_str = nlp(" ".join(text))
    for token in text_str:
        if token.pos_ in postags:
            filtered.append(token.text)
    return filtered
