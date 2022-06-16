import pymongo
import json
import os
import re
import numpy as np
import pandas as pd
import nlpActions
import dns
import streamlit as st

from pymongo import MongoClient, InsertOne
from pymongo.errors import BulkWriteError
from pprint import pprint


# Get creditials from st.secrets
db_username = st.secrets["db_username"]
db_pass = st.secrets["db_password"]
db_clustername = st.secrets["db_clustername"]

creds_str = "mongodb+srv://" + db_username + ":" +  db_pass +"@"+ db_clustername + ".zqqan.mongodb.net/<demo>?retryWrites=true&w=majority"
mongo_client = MongoClient(creds_str)
articles_db = mongo_client.demo
articles_collection = articles_db["articles"]


omitted_for_dataframe = {"_id":0,
                         "body":0,
                         "PERSON":0,
                         "NORP":0,
                         "FAC":0,
                         "ORG":0,
                         "GPE":0,
                         "LOC":0,
                         "PRODUCT":0,
                         "EVENT":0,
                         "LAW": 0,
                         "processedBodyWords":0}


def get_article_by_id(id, for_dataframe=True):
    query = {"id": id}
    omitted_fields = omitted_for_dataframe
    return perform_query(query, omitted_fields, for_dataframe)


def get_sources():
    return articles_collection.distinct("source")


def get_articles_by_source(source, for_dataframe=True):
    query = {"source": source}
    omitted_fields = {}
    if for_dataframe:
        omitted_fields = omitted_for_dataframe
        return perform_query(query, omitted_fields, for_dataframe=True)
    else:
        return perform_query(query, omitted_fields, for_dataframe=False)


def get_articles_by_entity(entity_label, entity_text, for_dataframe=True):
    query = {entity_label: entity_text}
    omitted_fields = {}
    if for_dataframe:
        omitted_fields = omitted_for_dataframe
        return perform_query(query, omitted_fields, for_dataframe=True)
    else:
        return perform_query(query, omitted_fields, for_dataframe=False)

##
# For future use
##
def get_articles_by_publish_year(pub_year, for_dataframe=True):
    query = {"publishYear": pub_year}
    omitted_fields = omitted_for_dataframe
    return perform_query(query, omitted_fields, for_dataframe=True)


def get_articles_by_publish_year_and_month(pub_year, pub_month, for_dataframe=True):
    query = {"publishYear": pub_year, "publishMonth": pub_month}
    omitted_fields = omitted_for_dataframe
    return perform_query(query, omitted_fields, for_dataframe=True)


def get_articles_by_publish_year_and_month_range(start_pub_year, 
                                                 start_pub_month,
                                                 end_pub_year, 
                                                 end_pub_month, 
                                                 for_dataframe=True):
    query = {
            "publishYear": {
                            "$gte": start_pub_year, 
                            "$lte": end_pub_year
                            },
            "publishMonth": {
                            "$gte": start_pub_month, 
                            "$lte": end_pub_month
                            }
            }
    omitted_fields = omitted_for_dataframe
    return perform_query(query, omitted_fields, for_dataframe=True)
##
# end functions for future use
##


def perform_query(query, omitted_fields, for_dataframe=True):
    if not for_dataframe:
        cursor = articles_collection.find(query, {"_id": 0})
    else:
        cursor = articles_collection.find(query, omitted_fields)
    return list(cursor)