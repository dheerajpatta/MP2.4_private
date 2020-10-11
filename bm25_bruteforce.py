import math
import sys
import time
import metapy
import pytoml
import numpy as np

def load_ranker(cfg_file):
    """
    Use this function to return the Ranker object to evaluate, 
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index.
    """
    return metapy.index.OkapiBM25(k1=1.88, b=0.749, k3=3.55)

cfg = 'config.toml'
print('Building or loading index...')
idx = metapy.index.make_inverted_index(cfg)
ranker = load_ranker(cfg)
ev = metapy.index.IREval(cfg)

with open(cfg, 'r') as fin:
    cfg_d = pytoml.load(fin)

query_cfg = cfg_d['query-runner']
if query_cfg is None:
    print("query-runner table needed in {}".format(cfg))
    sys.exit(1)

    start_time = time.time()
    top_k = 10
    query_path = query_cfg.get('query-path', 'queries.txt')
    query_start = query_cfg.get('query-id-start', 0)

    query = metapy.index.Document()
    ndcg = 0.0
    num_queries = 0

