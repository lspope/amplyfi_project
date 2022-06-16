#import spacy and lda - get entities and return them, get topics and return them
# as dictionaries so they can be stored in the article json in MongoDB
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter


target_entity_labels = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "LAW"]
   
#Spacy nlp pipeline with only parts needed for NER
nlp_ner_only = spacy.load("en_core_web_md")
nlp_ner_only.remove_pipe("parser")

#Spacy nlp pipeline without the NER
nlp_no_ner = spacy.load("en_core_web_md")
nlp_no_ner.remove_pipe("ner")
nlp_no_ner.Defaults.stop_words.add("\n")


def get_nlp_features(text, threshold, summary_limit):
    nlp_features = {}

    # get Entities
    entities_info = get_target_entities(text, nlp_ner_only)
    nlp_features.update({"entities": entities_info})

    # Functions to get prepared text for LDA and summary use same nlp pipeline
    processed_doc = nlp_no_ner(text.lower())
    # get Summary
    text_summary = get_summary(processed_doc, threshold, summary_limit)
    nlp_features.update({"summary": text_summary})
    # get Processed Text (for on the fly LDA)
    processed_body_words = get_processed_body_words(processed_doc)
    nlp_features.update({"processedBodyWords": processed_body_words})

    return nlp_features


def get_target_entities(text, nlp) :
    """
    Return detected entities of interest (see list) and total entity count

    Target Entities:
    PERSON,
    NORP (nationalities, religious and political groups),
    FAC (buildings, airports etc.),
    ORG (organizations),
    GPE (countries, cities etc.),
    LOC (mountain ranges, water bodies etc.),
    PRODUCT (products),
    EVENT (event names),
    LAW (legal document titles)
    """
    target_entity_dict = {"PERSON": set(),
                          "NORP": set(), 
                          "FAC": set(),
                          "ORG": set(),
                          "GPE": set(),
                          "LOC": set(),
                          "PRODUCT": set(),
                          "EVENT": set(),
                          "LAW": set()}

    doc = nlp(text)
    ent_count = 0

    for ent in doc.ents:
        if target_entity_labels.__contains__(ent.label_):
            ent_count += 1
            target_entity_dict.get(ent.label_).add(ent.text)

    target_entity_dict["entityMentions"] = ent_count
    return target_entity_dict


def get_summary(doc, threshold, summary_limit):
    """
    Get an n sentence summary of the given text.

    If the text is short (sentence count <= threshold), use a
    simple approach of gettting the first n sentences. 
    Otherwise, detecte the n most important sentences for summary.
    """
    summary_text = ""
    num_sents = len(list(doc.sents))

    if num_sents <= threshold:
        summary_sents = []
        counter = 0
        for sent in doc.sents:
            summary_sents.append(sent.text.capitalize())
            counter += 1
            if(counter >= summary_limit):
                break
        summary_text = " ".join(summary_sents)
    else:
        summary_text = get_nlp_summary(doc, summary_limit)
    return summary_text


def get_nlp_summary(doc, max_sent):
    """
    Get an n sentence summary of the given text.
    Code credit to: 
    https://betterprogramming.pub/extractive-text-summarization-using-spacy-in-python-88ab96d1fd97
    """
    keyword = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    for token in doc:
        if(token.is_stop or token.is_punct or token.like_num or token.like_email or token.like_url):
            continue # ignore stopwords and punctuation
        if(token.pos_ in pos_tag):
            keyword.append(token.text) # Include tokens that are desired Parts of Speech
    
    # Normalize keyword weights
    freq_word = Counter(keyword) # Convert list to dict with respective frequency values
    max_freq = Counter(keyword).most_common(1)[0][1] 
    for w in freq_word:
        freq_word[w] = (freq_word[w]/max_freq) # Normalize frequency

    sent_strength={}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys(): # Add the normalized keyword value to k-v pair of sentence
                if sent in sent_strength.keys():
                    sent_strength[sent]+=freq_word[word.text]
                else:
                    sent_strength[sent]=freq_word[word.text]

    summary = []
    # sort dict  of sentences based on normalized keyword value
    sorted_x = sorted(sent_strength.items(), key=lambda kv: kv[1], reverse=True) 
        
    counter = 0
    for i in range(len(sorted_x)):
        # Append to results and capitalize the sent.
        # Note that original case is not preserved.
        summary.append(str(sorted_x[i][0]).capitalize())
        counter += 1
        if(counter >= max_sent):
            break

    return " ".join(summary)


def get_processed_body_words(doc):
    """
    Process the article body so that LDA can be applied for topic modeling.
    Tokenize text
    Clean (remove stopwords, punctuation, emails, urls and numbers)
    Lemmatize 
    """
    processed_body_words = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.like_num and not token.like_email and not token.like_url:
            processed_body_words.append(token.lemma_)
    return " ".join(processed_body_words)


def get_lda_objects(processed_body_words_list):
    bag_of_words_vectorizer = CountVectorizer(max_df=0.95,
                                              min_df=0.01,
                                              ngram_range=(1,2),
                                              strip_accents="unicode")

    lda = LatentDirichletAllocation(max_iter=5,
                                    learning_method='online', 
                                    n_components=5, 
                                    random_state=42)

    #need to store processed body words as a single string
    #pass list of lists for now - testing
    results_tuple = ()
    the_articles = []
    for article in processed_body_words_list:
        the_articles.append(article)
    try:
        bag_of_words_matrix = bag_of_words_vectorizer.fit_transform(the_articles)
        lda.fit(bag_of_words_matrix)
        results_tuple = (lda, bag_of_words_matrix, bag_of_words_vectorizer)
    except:
        print("Could not conduct LDA on provided text.")
    return results_tuple




