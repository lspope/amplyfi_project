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

st.set_page_config(layout="wide")
st.title("Article Exploration Tool")

@st.cache
def load_sources():
    return mongoActions.get_sources()

def get_metrics(df):
     # get total articles, avg word count, min and max publishDates
    total = df.shape[0]
    min_yr = df["publishYear"].min()
    max_yr = df["publishYear"].max()
    date_range = "".join([str(min_yr), "-", str(max_yr)])
    avg_wc = round(df["wordCount"].mean())
    return (total, date_range, avg_wc)

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
    source_metrics = get_metrics(source_filtered_df)
    sm_col1, sm_col2, sm_col3 = st.columns(3)
    sm_col1.metric("Total Articles", source_metrics[0])
    sm_col2.metric("Date Range", source_metrics[1])
    sm_col3.metric("Avg Word Count", source_metrics[2])

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
desired_entity_text = st.text_input("Enter Entity Text")
if(st.button("Search by Entity")):
    mongo_docs_by_entity = mongoActions.get_articles_by_entity(entities_option, desired_entity_text, for_dataframe=False)

    if len(mongo_docs_by_entity) > 0:
        entity_filtered_df  = pd.DataFrame(mongo_docs_by_entity)
        entity_results_proc_texts = entity_filtered_df["processedBodyWords"].tolist()
        entity_filtered_df.drop(["body", "processedBodyWords", "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "LAW"],
                                axis=1, inplace=True)
        st.dataframe(entity_filtered_df)

        # get total articles, avg word count, min and max publishDates
        entity_metrics = get_metrics(entity_filtered_df)
        em_col1, em_col2, em_col3 = st.columns(3)
        em_col1.metric("Total Articles", entity_metrics[0])
        em_col2.metric("Date Range", entity_metrics[1])
        em_col3.metric("Avg Word Count", entity_metrics[2])

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
desired_id = st.text_input("Enter Article ID")
if(st.button("Display Article")):
    mongo_docs_by_id = mongoActions.get_article_by_id(desired_id, for_dataframe=False)
    
    if len(mongo_docs_by_id) > 0:
        selected_doc = mongo_docs_by_id[0]
        st.text(f"Source: {selected_doc.get('source')}")
        st.text(f"Publish Date: {selected_doc.get('publishDate')}")
        st.text(f"Title: {selected_doc.get('title')}")
        st.text(f"Word Count: {selected_doc.get('wordCount')}")

        st.text(f"Entity Mentions: {selected_doc.get('entityMentions')}")

        per_list = selected_doc.get("PERSON")
        norp_list = selected_doc.get("NORP")
        fac_list = selected_doc.get("FAC")
        org_list = selected_doc.get("ORG")
        gpe_list = selected_doc.get("GPE")
        loc_list = selected_doc.get("LOC")
        prod_list = selected_doc.get("PRODUCT")
        event_list = selected_doc.get("EVENT")
        law_list  = selected_doc.get("LAW")

        if (len(per_list) > 0):
            st.text(f"PERSON Entities: {selected_doc.get('PERSON')}")
        if (len(norp_list) > 0):
            st.text(f"NORP Entities: {selected_doc.get('NORP')}")
        if (len(fac_list) > 0):
            st.text(f"FAC Entities: {selected_doc.get('FAC')}")
        if (len(org_list) > 0):
            st.text(f"ORG Entities: {selected_doc.get('ORG')}")
        if (len(gpe_list) > 0):
            st.text(f"GPE Entities: {selected_doc.get('GPE')}")
        if (len(loc_list) > 0):
            st.text(f"LOC Entities: {selected_doc.get('LOC')}")
        if (len(prod_list) > 0):
            st.text(f"PRODUCT Entities: { selected_doc.get('PRODUCT')}")
        if (len(event_list) > 0):
            st.text(f"EVENT Entities: {selected_doc.get('EVENT')}")
        if (len(law_list) > 0):
            st.text(f"LAW Entities: {selected_doc.get('LAW')}")

        st.text_area("Summary", selected_doc.get('summary'))
        st.text_area("Body", selected_doc.get('body'))
  
