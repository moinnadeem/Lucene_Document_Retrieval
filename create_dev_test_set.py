import utils
import pickle
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

print("Beginning transformation...")
claims, labels, article_list, claim_set, claim_to_article = utils.extract_fever_jsonl_data("train.jsonl")

def retrieve_documents(claim, k=400):
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


def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=3):
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

#result = Parallel(n_jobs=6, verbose=1)(delayed(retrieve_documents)(c, 400) for c in claims)
result = parallel_process(claims, retrieve_documents, n_jobs=6)

print("Dumping shared dev task...")
with open("train.pkl", "wb") as f:
    pickle.dump(result, f)

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
