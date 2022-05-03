import os
import constants
import streamlit as st
import readData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from models import textDistance
import plotly.express as px

st.title("AIRA")
st.subheader("AI Resume Screening Assistant")
st.markdown("---")

st.header("Paste Job Description")
job_desc = st.text_area("Copy/Paste Job Description here", height = 300)


if job_desc != "" :
    st.subheader("Preprocessed Job Description")
    jd_doc = readData.clean_jd(job_desc)
    jd_arr = np.array([jd_doc])
    jd_df = pd.DataFrame(jd_arr, columns=[
        "Context", "Cleaned", "Selective", "Selective_Reduced", "TF_Based"])
    st.write(jd_df)

    st.subheader("Selective Reduced")
    sel_data = jd_df.Selective_Reduced[0]
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(str(sel_data))
    # Display the generated Word Cloud
    plt.figure(figsize=(7, 7), facecolor=None)
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(plt)

    st.subheader("Reduced Term Frequency")
    tf_data = jd_df.TF_Based[0]
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(str(tf_data))
    # Display the generated Word Cloud
    plt.figure(figsize=(7, 7), facecolor=None)
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(plt)

    st.markdown("---")

    # Upload CVs
    st.subheader("Upload CVs")
    uploaded_files = st.file_uploader("Accepted file formats: txt, doc, docx, pdf", type = ['txt','doc','docx','pdf'], accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()

        #Saving upload
        with open(os.path.join(constants.resume_dir,uploaded_file.name),"wb") as f:
            f.write((uploaded_file).getbuffer())


    resume_names = os.listdir(constants.resume_dir)
    if len(resume_names) > 0 :
        resume_data = readData.read_resumes(resume_names, constants.resume_dir)
        resume_doc = readData.clean_resumes(resume_data)

        st.subheader("Preprocessed Resumes")
        resume_df = pd.DataFrame(resume_doc, columns=[
               "Name", "Context", "Cleaned", "Selective", "Selective_Reduced", "TF_Based"])
        st.write(resume_df)

        st.markdown("---")
        
        # Score Calculation
        st.header("Ranking and Score")
        @st.cache()
        def calculate_scores(resumes, job_description):
            scores = []
            for x in range(resumes.shape[0]):
                score = textDistance.match_similar(
                    resumes['TF_Based'][x], job_description['TF_Based'][0])
                scores.append(score)
            return scores

        def calculate_cosine(resumes, job_description):
            scores = []
            for x in range(resumes.shape[0]):
                score = textDistance.cosine_similarity(
                    resumes['TF_Based'][x], job_description['TF_Based'][0])
                scores.append(score)
            return scores

        def calculate_jaccard(resumes, job_description):
            scores = []
            for x in range(resumes.shape[0]):
                score = textDistance.jaccard_similarity(
                    resumes['TF_Based'][x], job_description['TF_Based'][0])
                scores.append(score)
            return scores

        def calculate_sorensen_dice(resumes, job_description):
            scores = []
            for x in range(resumes.shape[0]):
                score = textDistance.sorensen_dice_similarity(
                    resumes['TF_Based'][x], job_description['TF_Based'][0])
                scores.append(score)
            return scores

        def calculate_normalized(resumes, job_description):
            scores = []
            for x in range(resumes.shape[0]):
                score = textDistance.normalized_similarity(
                    resumes['TF_Based'][x], job_description['TF_Based'][0])
                scores.append(score)
            return scores

        resume_df['Cosine'] = calculate_cosine(resume_df, jd_df)
        resume_df['Jaccard'] = calculate_jaccard(resume_df, jd_df)
        resume_df['Sorensen_Dice'] = calculate_sorensen_dice(resume_df, jd_df)
        resume_df['Normalized'] = calculate_normalized(resume_df, jd_df)
        resume_df['Score'] = calculate_scores(resume_df, jd_df)
        
        ranked_resumes = resume_df.sort_values(
            by=['Score'], ascending=False).reset_index(drop=True)

        ranked_resumes['Rank'] = pd.DataFrame(
            [i for i in range(1, len(ranked_resumes['Score'])+1)])

        rank_df = ranked_resumes.drop(["Context", "Cleaned", "Selective", "Selective_Reduced", "TF_Based"], axis=1)

        st.write(rank_df)

        st.markdown("---")

        # Analytics on Ranking and Score
        st.subheader("Rank Distribution")

        fig1 = px.bar(rank_df, x = rank_df['Name'], y = rank_df['Score'], color='Score')
        st.write(fig1)

        st.subheader("Top Ranked Resume: "+ranked_resumes['Name'][0])
        top_idf = ranked_resumes['TF_Based'][0]
        word_cloud = WordCloud(collocations = False, background_color = 'white').generate(str(top_idf))
        # Display the generated Word Cloud
        plt.figure(figsize=(7, 7), facecolor=None)
        plt.imshow(word_cloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(plt)

        st.markdown("---")

        if st.button('Delete Resumes'):
            for f in os.listdir(constants.resume_dir):
                os.remove(os.path.join(constants.resume_dir, f))

