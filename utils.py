import json
import unicodedata
import string
import numpy as np
import subprocess
import string

from multiprocessing import cpu_count

FEVER_LABELS = {'SUPPORTS': 0, 'REFUTES': 1}

def retrieve_documents(claim, k):
    cleaned_claim = claim.replace("/", " ")
    choices = query_lucene(cleaned_claim, str(k))
    retrieved = process_lucene_output(choices)
    return retrieved

def extract_fever_jsonl_data(path):
    '''
    HELPER FUNCTION

    Extracts lists of headlines, labels, articles, and a set of
    all distinct claims from a given FEVER jsonl file.

    Inputs:
      path: path to FEVER jsonl file
    Outputs:
      claims: list of claims for each data point
      labels: list of labels for each claim (see FEVER_LABELS in
        var.py)
      article_list: list of names of articles corresponding to
        each claim
      claim_set: set of distinct claim
    '''
    num_train = 0
    total_ev = 0

    claims = []
    labels = []
    article_list = []
    claim_set = set()
    claim_to_article = {}
    with open(path, 'r') as f:
        for item in f:
            data = json.loads(item)
            claim_set.add(data["claim"])
            if data["verifiable"] == "VERIFIABLE":
                evidence_articles = set()
                for evidence in data["evidence"][0]:
                    article_name = unicodedata.normalize('NFC', evidence[2])
                    article_name = preprocess_article_name(article_name)
                    
                    # Ignore evidence if the same article has
                    # already been used before as we are using
                    # the entire article and not the specified
                    # sentence.
                    if article_name in evidence_articles:
                        continue
                    else:
                        article_list.append(article_name)
                        evidence_articles.add(article_name)
                        claims.append(data["claim"])
                        labels.append(FEVER_LABELS[data["label"]])
                        if data['claim'] not in claim_to_article:
                            claim_to_article[data['claim']] = [article_name]
                        else:
                            claim_to_article[data['claim']].append(article_name)

                    total_ev += 1
                num_train += 1

    print("Num Distinct Claims", num_train)
    print("Num Data Points", total_ev)

    return claims, labels, article_list, claim_set, claim_to_article

def preprocess_article_name(s):
    s = s.replace("_", " ")
    s = s.replace("-LRB-", "(")
    s = s.replace("-RRB-", ")")
    s = s.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    return s.lower()

def char_ngrams(s, n):
    s = "#" + s + "#"
    return [s[i:i+n] for i in range(len(s) - 2)]

def query_lucene(c, k):
    # standard query: 
    # java -cp CLASSPATH org.apache.lucene.demo.SearchFiles -query "Loki is the dad of Hel."
    
    # replace the following classpath with your local Lucene instance
    classpath = "/home/moinnadeem/Documents/UROP/lucene-7.4.0/demo/lucene-demo-7.4.0.jar"
    classpath += ":/home/moinnadeem/Documents/UROP/lucene-7.4.0/core/lucene-core-7.4.0.jar"
    classpath += ":/home/moinnadeem/Documents/UROP/lucene-7.4.0/queryparser/lucene-queryparser-7.4.0.jar"
    
    c = c.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    # replace the following with the location of your index
    indexDir = "/home/moinnadeem/Documents/UROP/wiki-pages/index"
    
    
    return subprocess.check_output(["java", "-cp", classpath, "org.apache.lucene.demo.SearchFiles", "-index", indexDir, "-paging", k, "-query", c]).decode("utf-8").split("\n")

def process_lucene_output(output):
    assert len(output)>=13
    
    filenames = [o.split("/")[-1].split(".txt")[0] for o in output[2:-1]]
    return list(map(preprocess_article_name, filenames))

def calculate_precision(retrieved, relevant, k=None):
    if k==None:
        k = len(retrieved)
    return len(set(retrieved[:k]).intersection(set(relevant))) / len(set(retrieved))

def calculate_recall(retrieved, relevant, k=None):
    if k==None:
        k = len(retrieved)
    return len(set(retrieved[:k]).intersection(set(relevant))) / len(set(relevant))

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
