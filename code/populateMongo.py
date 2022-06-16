import pymongo
import json
import os
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


def populate_articles_db(path_to_articles):
    write_batch = []
    original_count = articles_collection.count_documents({})
    print(path_to_articles)
    filenames = [filename for filename in os.listdir(path_to_articles) if filename.endswith(".json")]


    for filename in filenames:
        print(filename)
        try:
            filepath = os.path.join(path_to_articles, filename)
            with open(filepath, encoding="utf-8") as orig_json:
                tmp = json.load(orig_json)  # Read the JSON file
                orig_json.close()               # Close the JSON file
            
            # get the derived features
            # meta data
            tmp["wordCount"] = len(tmp["body"].split())
            tmp["publishYear"] = int(tmp["publishDate"][0:4])
            tmp["publishMonth"] = int(tmp["publishDate"][5:7])

            # fet the NLP derived features
            npl_features = nlpActions.get_nlp_features(tmp["body"], 10, 3)
            # get entity info (count and entities listed by type)
            entities_info_dict = npl_features["entities"]
            for key in entities_info_dict:
                if key.isupper():
                    tmp[key] = list(entities_info_dict[key])
                else:
                    tmp[key] = entities_info_dict[key]
            # get summary
            tmp["summary"] = npl_features["summary"]
            # get processed body text for on-the-fly topic modeling (using LDA)
            tmp["processedBodyWords"] = npl_features["processedBodyWords"]

            write_batch.append(InsertOne(tmp))

        except:
            print(f"Something went wrong with reading and/or NLP processing article file {filepath}.")

    print(f"About to bulk write {len(write_batch)} docs")

    if len(write_batch) > 0:
        try:
            result = articles_collection.bulk_write(write_batch)
            print(f"bulk write ack?: {result.acknowledged}")
            print(f"bulk write insert count: {result.inserted_count}")
        except BulkWriteError as bwe:
            pprint(bwe.details)

    new_count = articles_collection.count_documents({})
    print(f"added {new_count-original_count} articles to collection")


##
# Optional in case we ever wanted to store the processed json to file instead of to a DB.
##
def save_json_to_file(json_string, filepath):
    with open(filepath, "w", encoding="utf-8") as to_save:
        to_save.write(json.dumps(json_string))
        to_save.close()


if __name__ == "__main__":
    path_to_articles = os.path.abspath("/home/user/Flatiron/post_work/amplyfi/data/sample/")
    populate_articles_db(path_to_articles)
    print("......... Goodbye!")
