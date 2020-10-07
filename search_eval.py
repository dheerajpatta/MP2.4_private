import math
import sys
import time
import metapy
import pytoml

class InL2Ranker(metapy.index.RankingFunction):
    """
    Create a new ranking function in Python that can be used in MeTA.
    """
    def __init__(self, some_param=0.5):
        self.c = some_param
        # You *must* call the base class constructor here!
        super(InL2Ranker, self).__init__()

    def score_one(self, sd):
        """
        Scoring function for current Query (Q) and doucment (D)
        sd : score data object
        N : the total number of documents in the corpus C
        avgdl : the average document length
        c > 0 : is a parameter
        """
        tfn = sd.doc_term_count * math.log(1.0 + (sd.avg_dl/sd.doc_size),2)
        
        tfn_by_tfn_plus_c = sd.query_term_weight * (tfn/(tfn+self.c))
        log_n_plus_one_by_corpus = math.log((sd.num_docs+1)/(sd.corpus_term_count + 0.5),2)
        
        return (tfn_by_tfn_plus_c * log_n_plus_one_by_corpus)

class PL2Ranker(metapy.index.RankingFunction):
    """
    Create a new ranking function in Python that can be used in MeTA
    """
    def __init__(self, c_param=0.75):
        self.c = c_param
        super(PL2Ranker, self).__init__()

    def score_one(self, sd):
        lda = sd.num_docs / sd.corpus_term_count
        tfn = sd.doc_term_count * math.log2(1.0 + self.c * sd.avg_dl /
                sd.doc_size)
        if lda < 1 or tfn <= 0:
            return 0.0
        numerator = tfn * math.log2(tfn * lda) \
                        + math.log2(math.e) * (1.0 / lda - tfn) \
                        + 0.5 * math.log2(2.0 * math.pi * tfn)
        return sd.query_term_weight * numerator / (tfn + 1.0)

def load_ranker(cfg_file):
    """
    Use this function to return the Ranker object to evaluate, 
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index.
    """
    # return metapy.index.OkapiBM25(k1=1.2,b=0.75,k3=7.0)
    # return InL2Ranker(some_param=0.5)
    return PL2Ranker(c_param=0.75)

if __name__ == '__main__':
    # if len(sys.argv) != 2:
        # print("Usage: {} config.toml".format(sys.argv[0]))
        # sys.exit(1)

    # cfg = sys.argv[1]
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
    ndcg = 0.0 #110
    num_queries = 0

    print('Running queries')
    with open(query_path) as query_file:
        for query_num, line in enumerate(query_file):
            query.content(line.strip())
            results = ranker.score(idx, query, top_k)
            ndcg += ev.ndcg(results, query_start + query_num, top_k)
            num_queries+=1
    ndcg= ndcg / num_queries
            
    print("NDCG@{}: {}".format(top_k, ndcg))
    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))