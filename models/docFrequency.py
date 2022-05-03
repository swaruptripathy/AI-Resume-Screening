from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(token):
    model = TfidfVectorizer(max_df=0.05, min_df=0.002)
    model.fit_transform(token)
    sentence = " ".join(model.get_feature_names())
    return sentence