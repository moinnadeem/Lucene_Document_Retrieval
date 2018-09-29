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

claims, labels, article_list, claim_set, claim_to_article = utils.extract_fever_jsonl_data("train.jsonl")


k = [1,2, 5,10]

def score_claim(claim):
    cleaned_claim = claim.replace("/", " ")
    choices = utils.query_lucene(cleaned_claim)
    retrieved = utils.process_lucene_output(choices)
    relevant = claim_to_article[claim]
    mAP = {}
    for i in k:
        precision = utils.calculate_precision(retrieved=retrieved, relevant=relevant, k=i)
        recall = utils.calculate_recall(retrieved=retrieved, relevant=relevant, k=i)
        mAP[i] = {}
        mAP[i]['precision'] = precision
        mAP[i]['recall'] = recall
    return mAP

print("Beginning to score documents...")
result = Parallel(n_jobs=16, verbose=1, prefer="threads")(delayed(score_claim)(c) for c in claim_to_article.keys())

print("Saving results to disk...")
with open("result.pkl", "wb") as f:
    pickle.dump(result, f)

mAP = utils.calculatemAP(result, k)

print("Mean Average Precision:")
utils.displaymAP(mAP)
