import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyLDAvis.sklearn

import mongoActions
import nlpActions

from streamlit import components
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


st.title("Article Exploration Tool")

DATE_COLUMN = "publishDate"

@st.cache
def load_sources():
    return mongoActions.get_sources()

sources_list = load_sources()
sources_nparray = np.array(sources_list)
source_option = st.selectbox("Select Article Source:", sources_nparray)

##
# Search by Source and do LDA topic modeling on results
##
if(st.button("Search by Source")):
    mongo_docs_by_source = mongoActions.get_articles_by_source(source_option, for_dataframe=False)
    source_filtered_df = pd.DataFrame(mongo_docs_by_source)
    proc_texts = source_filtered_df["processedBodyWords"].tolist()
    source_filtered_df.drop(["body","processedBodyWords", "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "LAW"],
                            axis=1, inplace=True)
    st.dataframe(source_filtered_df)

    # get total articles, avg word count, min and max publishDates
    total_for_source = source_filtered_df.shape[0]
    avg_wc_source = round(source_filtered_df["wordCount"].mean())
    min_yr_source = source_filtered_df["publishYear"].min()
    max_yr_source = source_filtered_df["publishYear"].max()
    date_range_source = "".join([str(min_yr_source), "-", str(max_yr_source)])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Articles", total_for_source)
    col2.metric("Covers", date_range_source)
    col3.metric("Avg Word Count", avg_wc_source)

    # get article counts for year
    by_pub_year = source_filtered_df.groupby(["publishYear"])['id'].count()
    for_plot = by_pub_year.reset_index()
    for_plot.rename(columns={"publishYear": "Publish Year", "id": "Article Count"})
    st.table(for_plot)

    if len(source_filtered_df) > 3:
        # Get LDA topics from these articles
        lda_objs_tuple = nlpActions.get_lda_objects(proc_texts)

        if len(lda_objs_tuple) == 3:
            prepared_data = pyLDAvis.sklearn.prepare(lda_objs_tuple[0], lda_objs_tuple[1], lda_objs_tuple[2])
            pyLDAvis.save_html(prepared_data, 'lda.html')
            with open('./lda.html', 'r') as f:
                html_string = f.read()
                components.v1.html(html_string, width=1200, height=800, scrolling=False)


##
# Search by Entity and do LDA topic modeling on results
##
entities_types = np.array(["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "LAW"])
entities_option = st.selectbox("Select Entity Type:", entities_types)
desired_entity_text = st.text_input("Desired Entity Text")
if(st.button("Search by Entity")):
    mongo_docs_by_entity = mongoActions.get_articles_by_entity(entities_option, desired_entity_text, for_dataframe=False)

    if len(mongo_docs_by_entity) > 0:
        entity_filtered_df  = pd.DataFrame(mongo_docs_by_entity)
        entity_results_proc_texts = entity_filtered_df["processedBodyWords"].tolist()
        entity_filtered_df.drop(["body", "processedBodyWords", "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "LAW"],
                                axis=1, inplace=True)
        st.dataframe(entity_filtered_df)
        
        # Get LDA topics from these articles
        entity_lda_objs_tuple = nlpActions.get_lda_objects(entity_results_proc_texts)

        if len(entity_lda_objs_tuple) == 3:
            entity_prepared_data = pyLDAvis.sklearn.prepare(entity_lda_objs_tuple[0], entity_lda_objs_tuple[1], entity_lda_objs_tuple[2])
            pyLDAvis.save_html(entity_prepared_data, 'entity_lda.html')
            with open('./entity_lda.html', 'r') as f:
                entity_html_string = f.read()
                components.v1.html(entity_html_string, width=1200, height=800, scrolling=False)


##
# Display article details and contents
##
desired_id = st.text_input("Desired Article ID")
if(st.button("Display Article")):
    mongo_docs_by_id = mongoActions.get_article_by_id(desired_id, for_dataframe=False)
    
    if len(mongo_docs_by_id) > 0:
        selected_doc = mongo_docs_by_id[0]
        st.text(f"Source: {selected_doc.get('source')}")
        st.text(f"Publish Date: {selected_doc.get('publishDate')}")
        st.text(f"Title: {selected_doc.get('title')}")
        st.text(f"Word Count: {selected_doc.get('wordCount')}")
        st.text(f"Entity Mentions: {selected_doc.get('entityMentions')}")
        st.text_area("Summary", selected_doc.get('summary'))
        st.text_area("Body", selected_doc.get('body'))