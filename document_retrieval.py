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
import argparse

claims, labels, article_list, claim_set, claim_to_article = utils.extract_fever_jsonl_data("dev_wo_nei_short.jsonl")

k = [1,2,5,10,20,50,100,200,300,400]

parser = argparse.ArgumentParser(description='Learning the optimal convolution for network.')
parser.add_argument("--small", action="store_true", help="Verify that the script works on a small dataset.", default=False)
parser.add_argument("--jar", help="Location of the JAR to execute.", default="untitled1.jar", type=str)
args = parser.parse_args()
print(args)

def score_claim(claim):
    cleaned_claim = claim.replace("/", " ")
    choices = utils.query_customized_lucene(cleaned_claim, k=max(k), jar_name=args.jar)
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

print("Beginning to score documents using {}...".format(args.jar))
query_set = list(claim_to_article.keys())

if args.small:
    query_set = query_set[:100]

# result = Parallel(n_jobs=15)(delayed(score_claim)(c) for c in query_set)
result = utils.parallel_process(query_set, score_claim, n_jobs=15)

# print("Saving results to disk...")
# with open("result.pkl", "wb") as f:
    # pickle.dump(result, f)

mAP = utils.calculatemAP(result, k)
print(utils.query_customized_lucene("testing", k=5, jar_name=args.jar)[0])
print("Mean Average Precision:")
utils.displaymAP(mAP)
recalls = []
recalls.extend(mAP[1]['recall'])
recalls.extend(mAP[5]['recall'])
recalls.extend(mAP[10]['recall'])
recalls.extend(mAP[20]['recall'])
print("Avg recall: {}".format(np.mean(recalls)))
