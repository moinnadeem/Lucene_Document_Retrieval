import json
import unicodedata
import joblib
import string
import pickle
import nltk
import numpy as np
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.autonotebook import tqdm 


import unicodedata
import string
import numpy as np
import subprocess
import string

from multiprocessing import cpu_count

FEVER_LABELS = {'SUPPORTS': 0, 'REFUTES': 1}

def retrieve_documents(claim, k):
    cleaned_claim = claim.replace("/", " ")
    choices = query_customized_lucene(cleaned_claim, str(k))
    print(choices)
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

def query_standard_lucene(c, k):
    # standard query: 
    # java -cp CLASSPATH org.apache.lucene.demo.SearchFiles -query "Loki is the dad of Hel."
    
    # replace the following classpath with your local Lucene instance
    classpath = "/usr/users/mnadeem/UROP/lucene-7.4.0/demo/lucene-demo-7.4.0.jar"
    classpath += ":/usr/users/mnadeem/UROP/lucene-7.4.0/core/lucene-core-7.4.0.jar"
    classpath += ":/usr/users/mnadeem/UROP/lucene-7.4.0/queryparser/lucene-queryparser-7.4.0.jar"
    
    c = c.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    # replace the following with the location of your index
    indexDir = "/usr/users/mnadeem/UROP/wiki-pages/index"
    
    
    # return subprocess.check_output(["java", "-cp", classpath, "org.apache.lucene.demo.SearchFiles", "-index", indexDir, "-paging", k, "-query", c]).decode("utf-8").split("\n")
    return subprocess.check_output(["java", "org.apache.lucene.demo.SearchFiles", "-index", indexDir, "-paging", k, "-query", c]).decode("utf-8").split("\n")

def query_customized_lucene(c, k, jar_name="print_scores"):
    # standard query: 
    # java -jar /usr/users/mnadeem/UROP/lucene_java/out/artifacts/untitled1_jar/untitled1.jar -index /usr/users/mnadeem/UROP/lucene_java/out/artifacts/index 
    
    # replace the following classpath with your local Lucene instance
    
    c = c.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))

    # replace the following with the location of your index
    prefix = "/home/moinnadeem/demo/backend/flask/FEVER_Document_Retrieval"

    indexDir = "{}/wiki-pages/index".format(prefix)
    jar_location = "{}/{}.jar".format(prefix, jar_name) 
    
    return subprocess.check_output(["java", "-jar", jar_location, "-index", indexDir, "-paging", str(k), "-query", c]).decode("utf-8").split("\n")

def process_lucene_output(output):
    if output==['']:
        return []
    assert len(output)>=13
    
    filenames = [o.split("/")[-1].split(".txt")[0] for o in output[3:-1]]
    return list(map(preprocess_article_name, filenames))

def calculate_precision(retrieved, relevant, k=None):
    if k==None:
        k = len(retrieved)
    return len(set(retrieved[:k]).intersection(set(relevant))) / len(set(retrieved))

def calculate_recall(retrieved, relevant, k=None):
    """
        retrieved: a list of sorted documents that were retrieved
        relevant: a list of sorted documents that are relevant
        k: how many documents to consider, all by default.
    """
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
    for k in sorted(mAP.keys()):
        v = mAP[k]
        for k_i, v_i in v.items():
            print("{} @ {}: {}".format(k_i, k, np.mean(v_i)))

def parallel_process(array, function, n_jobs=12, use_kwargs=False, front_num=3):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out

