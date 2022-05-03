import textdistance as td

def cosine_similarity(resume, job_desc):
    c = td.cosine.similarity(resume, job_desc)
    return c*100

def jaccard_similarity(resume, job_desc):
    j = td.jaccard.similarity(resume, job_desc)
    return j*100

def sorensen_dice_similarity(resume, job_desc):
    s = td.sorensen_dice.similarity(resume, job_desc)
    return s*100

def normalized_similarity(resume, job_desc):
    o = td.overlap.normalized_similarity(resume, job_desc)
    return o*100

def match_similar(resume, job_desc):
    j = td.jaccard.similarity(resume, job_desc)
    s = td.sorensen_dice.similarity(resume, job_desc)
    c = td.cosine.similarity(resume, job_desc)
    o = td.overlap.normalized_similarity(resume, job_desc)
    total = (j+s+c+o)/4
    return total*100


