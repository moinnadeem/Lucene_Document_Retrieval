import json
import unicodedata
import string

FEVER_LABELS = {'SUPPORTS': 0, 'REFUTES': 1}

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