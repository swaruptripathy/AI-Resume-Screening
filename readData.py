from datacleaner import preprocess
from models import docFrequency
import textract as tx

def read_resumes(list_of_resumes, resume_directory):
    placeholder = []
    for res in list_of_resumes:
        temp = []
        temp.append(res)
        text = tx.process(resume_directory+res, encoding='ascii')
        text = str(text, 'utf-8')
        temp.append(text)
        placeholder.append(temp)
    return placeholder

def clean_jd(text):
    data = [text]

    # preprocess data cleaning
    raw = preprocess.preprocess(text)
    data.append(" ".join(raw[0]))
    data.append(" ".join(raw[1]))
    data.append(" ".join(raw[2]))

    sentence = docFrequency.tf_idf(data[3].split(" "))
    data.append(sentence)
    return data

def clean_resumes(data):
    for i in range(len(data)):

        # preprocess data cleaning
        raw = preprocess.preprocess(data[i][1])
        data[i].append(" ".join(raw[0]))
        data[i].append(" ".join(raw[1]))
        data[i].append(" ".join(raw[2]))

        # get tf-idf
        sentence = docFrequency.tf_idf(data[i][3].split(" "))
        data[i].append(sentence)
    return data
