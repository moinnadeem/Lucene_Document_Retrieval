from . import utils
import pickle
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def retrieve_documents(claim, k=20):
    cleaned_claim = claim.replace("/", " ")
    choices = utils.query_customized_lucene(cleaned_claim, str(k), jar_name="print_scores")
    filenames = []
    scores = []
    for o in choices[3:-1]:
        o, score = o.split(", score=")
        a = o.split("/")[-1].split(".txt")[0]
        a = a.replace("-LRB-", "(")
        a = a.replace("-RRB-", ")")
        a = a.replace("-COLON-", ":")
        scores.append(score)
        filenames.append("http://wikipedia.org/wiki/{}".format(a))
    return {"claim": claim, "evidence": filenames, "scores": scores}

#result = Parallel(n_jobs=6, verbose=1)(delayed(retrieve_documents)(c, 400) for c in claims)

#print("Beginning test task...")
#claims = []
#with open('shared_task_test.jsonl') as f:
#    inp = f.readlines()
#    for c in inp:
#        j = json.loads(c)
#        claims.append(j['claim'])
#
#result = Parallel(n_jobs=8, verbose=1)(delayed(retrieve_documents)(c, 400) for c in claims)
#
#print("Dumping test task...")
#with open("shared_task_test.pkl", "wb") as f:
#    pickle.dump(result, f)

if __name__=="__main__":
    filename = "shared_task_dev.jsonl" 
    print("Beginning transformation for file {}...".format(filename))
    claims, labels, article_list, claim_set, claim_to_article = utils.extract_fever_jsonl_data(filename)

    claims = list(set(claims))

    result = utils.parallel_process(claims, retrieve_documents, n_jobs=30)

    print("Running training task...")
    with open("shared_task_dev.pkl", "wb") as f:
        pickle.dump(result, f)

