from datacleaner import cleanText
import spacy

nlp = spacy.load('en_core_web_sm')

def text_clean(text):
    # Get extracted text from file and clean it.
    text = cleanText.remove_punctuations(text)
    text = cleanText.remove_stopwords(text)
    text = cleanText.remove_tags(text)
    text = cleanText.lemmatize(text)
    return text


def reduce_repetition(text):
    # use set to reduce word repetition
    return list(set(text))


def get_target_words(text):
    # Use spacy tags to extract relevant words that contain words related to the Job Description.
    target = []
    sentence = " ".join(text)
    doc = nlp(sentence)
    for token in doc:
        if token.tag_ in ['NN', 'NNP']:
            target.append(token.text)
    return target

def preprocess(text):
    sentence = []
    sentence_cleaned = text_clean(text)
    sentence.append(sentence_cleaned)
    sentence_reduced = reduce_repetition(sentence_cleaned)
    sentence.append(sentence_reduced)
    sentence_target = get_target_words(sentence_reduced)
    sentence.append(sentence_target)
    return sentence
