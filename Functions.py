import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


import tensorflow as tf
# import tensorflow_hub as hub
import tensorflow.keras
from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import metrics as kmetrics
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# Bert
import os
import transformers
from transformers import *


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time
import logging

from sklearn import cluster, metrics
from sklearn import manifold, decomposition


# ----------------------------------------------- Text processing functions --------------------------------------------


def tokenizer_fct(sentence):
    """Tokenize a given sentence"""
    
    sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ')
    word_tokens = word_tokenize(sentence_clean)
    return word_tokens


def process_text(doc, rejoin=False, list_rare_words=None, min_len_word=3, force_is_alpha=True, lem_or_stem=None, eng_words=None):
    
    # List unique words
    if not list_rare_words:
        list_rare_words = []
        
    # Lower
    doc = doc.lower().strip()
    
    # Tokenize
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    raw_token_list = tokenizer.tokenize(doc)
    
    # classic stopwords
    stop_words = list(set(stopwords.words('english'))) + ['[', ']', ',', '.', ':', '?', '(', ')']
    cleaned_token_list = [w for w in raw_token_list if w not in stop_words]
    
    # deleting rare tokens
    non_rare_tokens = [w for w in cleaned_token_list if w not in list_rare_words]
    
    # deleting word with too low lenght
    more_than_N = [w for w in non_rare_tokens if len(w) >= min_len_word]
    
    # only alpha characters
    if force_is_alpha:
        alpha_tokens = [w for w in more_than_N if w.isalpha()]
    else:
        alpha_tokens = more_than_N
        
    # stem or lem
    if not lem_or_stem:
        trans_text = alpha_tokens
    elif lem_or_stem == "lem":
        trans = nltk.stem.WordNetLemmatizer()
        trans_text = [trans.lemmatize(i) for i in alpha_tokens]
    else:
        trans = nltk.stem.PorterStemmer()
        trans_text = [trans.stem(i) for i in alpha_tokens]
        
    # delete non english words
    if eng_words:
        eng_text = [i for i in trans_text if i in eng_words]
    else:
        eng_text = trans_text
    
    # manage return type
    if rejoin:
        return " ".join(eng_text)
    else:
        return eng_text
    

# ----------------------------------------------- Text visualization functions --------------------------------------------


def ARI_fct(features, l_cat, y_cat_num):
    """ Computing TSNE, KMeans clustering and computing ARI score between actual labels and KMeans ones"""
    
    time1 = time.time()
    num_labels=len(l_cat)
    tsne = manifold.TSNE(n_components=2, perplexity=30, n_iter=2000, 
                                 init='random', learning_rate=200, random_state=42)
    X_tsne = tsne.fit_transform(features)
    
    # KMeans clustering with TSNE transformed datas 
    cls = cluster.KMeans(n_clusters=num_labels, n_init=100, random_state=42)
    cls.fit(X_tsne)
    
    # Comparing actual labels with clustering using ARI
    ARI = np.round(metrics.adjusted_rand_score(y_cat_num, cls.labels_),4)
    time2 = np.round(time.time() - time1,0)
    print("ARI : ", ARI, "time : ", time2)
    
    return ARI, X_tsne, cls.labels_


def TSNE_visu_fct(X_tsne, y_cat_num, labels, ARI, l_cat):
    """Visualization of TSNE with actual labels and KMeans clustering"""
    
    fig = plt.figure(figsize=(15,6))
    
    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=y_cat_num, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=l_cat, loc="best", title="Categorie")
    plt.title('Représentation des tweets par catégories réelles')
    
    ax = fig.add_subplot(122)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=labels, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=set(labels), loc="best", title="Clusters")
    plt.title('Représentation des tweets par clusters')
    
    plt.show()
    print("ARI : ", ARI)
    

# ----------------------------------------------- BERT functions --------------------------------------------
    

def bert_inp_fct(sentences, bert_tokenizer, max_length):
    """Processing of sentences"""
    
    input_ids=[]
    token_type_ids = []
    attention_mask=[]
    bert_inp_tot = []

    for sent in sentences:
        bert_inp = bert_tokenizer.encode_plus(sent,
                                              add_special_tokens = True,
                                              max_length = max_length,
                                              padding='max_length',
                                              return_attention_mask = True, 
                                              return_token_type_ids=True,
                                              truncation=True,
                                              return_tensors="tf")
    
        input_ids.append(bert_inp['input_ids'][0])
        token_type_ids.append(bert_inp['token_type_ids'][0])
        attention_mask.append(bert_inp['attention_mask'][0])
        bert_inp_tot.append((bert_inp['input_ids'][0], 
                             bert_inp['token_type_ids'][0], 
                             bert_inp['attention_mask'][0]))

    input_ids = np.asarray(input_ids)
    token_type_ids = np.asarray(token_type_ids)
    attention_mask = np.array(attention_mask)
    
    return input_ids, token_type_ids, attention_mask, bert_inp_tot
    

def feature_BERT_fct(model, model_type, sentences, max_length, b_size, mode='HF'):
    """Creation of features"""
    
    
    batch_size = b_size
    batch_size_pred = b_size
    bert_tokenizer = AutoTokenizer.from_pretrained(model_type)
    time1 = time.time()

    for step in range(len(sentences)//batch_size) :
        idx = step*batch_size
        input_ids, token_type_ids, attention_mask, bert_inp_tot = bert_inp_fct(sentences[idx:idx+batch_size], 
                                                                      bert_tokenizer, max_length)
        
        if mode=='HF' :    # Bert HuggingFace
            outputs = model.predict([input_ids, attention_mask, token_type_ids], batch_size=batch_size_pred)
            last_hidden_states = outputs.last_hidden_state

        if mode=='TFhub' : # Bert Tensorflow Hub
            text_preprocessed = {"input_word_ids" : input_ids, 
                                 "input_mask" : attention_mask, 
                                 "input_type_ids" : token_type_ids}
            outputs = model(text_preprocessed)
            last_hidden_states = outputs['sequence_output']
             
        if step ==0 :
            last_hidden_states_tot = last_hidden_states
            last_hidden_states_tot_0 = last_hidden_states
        else :
            last_hidden_states_tot = np.concatenate((last_hidden_states_tot,last_hidden_states))
    
    features_bert = np.array(last_hidden_states_tot).mean(axis=1)
    
    time2 = np.round(time.time() - time1,0)
    print("temps traitement : ", time2)
     
    return features_bert, last_hidden_states_tot


# ----------------------------------------------- USE functions --------------------------------------------


def feature_USE_fct(sentences, b_size, embed):
    """USE text model"""
    
    batch_size = b_size
    time1 = time.time()

    for step in range(len(sentences)//batch_size) :
        idx = step*batch_size
        feat = embed(sentences[idx:idx+batch_size])

        if step ==0 :
            features = feat
        else :
            features = np.concatenate((features,feat))

    time2 = np.round(time.time() - time1,0)
    return features