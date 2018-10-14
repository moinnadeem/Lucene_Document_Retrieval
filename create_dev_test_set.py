import utils
import pickle
from tqdm import tqdm_notebook
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import json
from tqdm import tqdm_notebook

#print("Beginning transformation...")
#claims, labels, article_list, claim_set, claim_to_article = utils.extract_fever_jsonl_data("shared_task_test.jsonl")

def retrieve_documents(claim, k):
    cleaned_claim = claim.replace("/", " ")
    choices = utils.query_lucene(cleaned_claim, str(k))
    filenames = []
    for o in choices[2:-1]:
        a = o.split("/")[-1].split(".txt")[0]
        a = a.replace("-LRB-", "(")
        a = a.replace("-RRB-", ")")
        a = a.replace("-COLON-", ":")
        filenames.append("http://wikipedia.org/wiki/{}".format(a))
    return {"claim": claim, "evidence": filenames}

#result = Parallel(n_jobs=8, verbose=1)(delayed(retrieve_documents)(c, 400) for c in claims)

#print("Dumping shared dev task...")
#with open("shared_task_dev.pkl", "wb") as f:
#    pickle.dump(result, f)

print("Beginning test task...")
claims = []
with open('shared_task_test.jsonl') as f:
    inp = f.readlines()
    for c in inp:
        j = json.loads(c)
        claims.append(j['claim'])

result = Parallel(n_jobs=8, verbose=1)(delayed(retrieve_documents)(c, 400) for c in claims)

print("Dumping test task...")
with open("shared_task_test.pkl", "wb") as f:
    pickle.dump(result, f)
