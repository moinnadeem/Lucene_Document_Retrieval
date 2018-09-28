# coding: utf-8

# # FEVER Document Retrieval

# **Purpose**: the purpose of this notebook is to develop a baseline approach for scoring document retrieval on the FEVER dataset with Apache Lucene.
# 
# **Input**: This document requires the Lucene index, and JSON files to run.

# ## Setting up Lucene Query

# In[78]:


import utils
import pickle
from tqdm import tqdm_notebook
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import numpy as np
import subprocess
import string

# In[2]:


claims, labels, article_list, claim_set, claim_to_article = utils.extract_fever_jsonl_data("../train.jsonl")


# In[32]:


def query_lucene(c):
    # standard query: 
    # java -cp CLASSPATH org.apache.lucene.demo.SearchFiles -query "Loki is the dad of Hel."
    
    # replace the following classpath with your local Lucene instance
    classpath = "/home/moinnadeem/Documents/UROP/lucene-7.4.0/demo/lucene-demo-7.4.0.jar"
    classpath += ":/home/moinnadeem/Documents/UROP/lucene-7.4.0/core/lucene-core-7.4.0.jar"
    classpath += ":/home/moinnadeem/Documents/UROP/lucene-7.4.0/queryparser/lucene-queryparser-7.4.0.jar"
    
    c = c.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    # replace the following with the location of your index
    indexDir = "/home/moinnadeem/Documents/UROP/wiki-pages/index"
    
    
    return subprocess.check_output(["java", "-cp", classpath, "org.apache.lucene.demo.SearchFiles", "-index", indexDir, "-query", c]).decode("utf-8").split("\n")

def process_lucene_output(output):
    assert len(output)==13
    
    filenames = [o.split("/")[-1].split(".txt")[0] for o in output[2:-1]]
    return list(map(utils.preprocess_article_name, filenames))

def calculate_precision(retrieved, relevant, k=None):
    if k==None:
        k = len(retrieved)
    return len(set(retrieved[:k]).intersection(set(relevant))) / len(set(retrieved))

def calculate_recall(retrieved, relevant, k=None):
    if k==None:
        k = len(retrieved)
    return len(set(retrieved[:k]).intersection(set(relevant))) / len(set(relevant))

k = [1,2, 5,10]

def score_claim(claim):
    cleaned_claim = claim.replace("/", " ")
    choices = query_lucene(cleaned_claim)
    retrieved = process_lucene_output(choices)
    relevant = claim_to_article[claim]
    mAP = {}
    for i in k:
        precision = calculate_precision(retrieved=retrieved, relevant=relevant, k=i)
        recall = calculate_recall(retrieved=retrieved, relevant=relevant, k=i)
        mAP[i] = {}
        mAP[i]['precision'] = precision
        mAP[i]['recall'] = recall
    return mAP


result = Parallel(n_jobs=cpu_count(), verbose=1)(delayed(score_claim)(k) for k in list(claim_to_article.keys())[:500])

with open("result.pkl", "wb") as f:
    pickle.dump(result, f)

def calculatemAP(mAP, k):
    mAP_final = {}
    
    for i in k:
        mAP_final[i] = {}
        mAP_final[i]['precision'] = []
        mAP_final[i]['recall'] = []
        
    for ap in mAP:
        for k, v in ap.items():
            mAP_final[k]['precision'].append(v['precision'])
            mAP_final[k]['recall'].append(v['recall'])

    return mAP_final

def displaymAP(mAP):
    for k, v in mAP.items():
        for k_i, v_i in v.items():
            print("{} @ {}: {}".format(k_i, k, np.mean(v_i)))

mAP = calculatemAP(result, k)

displaymAP(mAP)
