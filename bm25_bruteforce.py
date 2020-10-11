import math
import sys
import time
import metapy
import pytoml
import numpy as np

def load_ranker(cfg_file, i):
    """
    Use this function to return the Ranker object to evaluate, 
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index.
    """
    return metapy.index.OkapiBM25(k1=i, b=0.75, k3=3.25)

cfg = 'config.toml'
print('Building or loading index...')
idx = metapy.index.make_inverted_index(cfg)
# ranker = load_ranker(cfg)
ev = metapy.index.IREval(cfg)

with open(cfg, 'r') as fin:
    cfg_d = pytoml.load(fin)

query_cfg = cfg_d['query-runner']
top_k = 10
query_path = query_cfg.get('query-path', 'queries.txt')
query_start = query_cfg.get('query-id-start', 0)
query = metapy.index.Document()
ndcg = 0.0
num_queries = 0

print('Running queries')
with open(query_path) as query_file:
    for i in np.arange(1, 3.1, 0.01):
        for query_num, line in enumerate(query_file):
            query.content(line.strip())
            ranker = load_ranker(cfg, i)
            results = ranker.score(idx, query, top_k)
            ndcg += ev.ndcg(results, query_start + query_num, top_k)
            num_queries+=1
        ndcg= ndcg / num_queries        
        print("NDCG@{}: {} i@{}".format(top_k, ndcg, i))
