from rank_bm25 import BM25Okapi
import math
import sys
import time
import metapy
import pytoml
corpus = 'cranfield.txt'

def load_ranker(cfg_file):
    """
    Use this function to return the Ranker object to evaluate, 
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index.
    """
    corpus = cfg_file
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    print(bm25.get_scores(tokenized_query))

if __name__ == '__main__':
    #if len(sys.argv) != 2:
        #print("Usage: {} config.toml".format(sys.argv[0]))
        #sys.exit(1)

    cfg = 'cranfield-queries.txt' #sys.argv[1]
    query = 'queries.txt'
    tokenized_query = query.split(" ")
    load_ranker(cfg)