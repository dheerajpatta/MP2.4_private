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
        tfn = sd.doc_term_count * math.log(1.0 + self.c * sd.avg_dl /
                sd.doc_size,2)
        if lda < 1 or tfn <= 0:
            return 0.0
        numerator = tfn * math.log(tfn * lda,2) \
                        + math.log(math.e,2) * (1.0 / lda - tfn) \
                        + 0.5 * math.log(2.0 * math.pi * tfn,2)
        return sd.query_term_weight * numerator / (tfn + 1.0)

def avg_doc_len(coll):
    tot_dl = 0
    for id, doc in coll.get_docs().items():
        tot_dl = tot_dl + doc.get_doc_len()
    return tot_dl / coll.get_num_docs()


def bm25(coll, q, df):
    bm25s = {}
    avg_dl = avg_doc_len(coll)
    no_docs = coll.get_num_docs()
    for id, doc in coll.get_docs().items():
        query_terms = q.split()
        qfs = {}
        for t in query_terms:
            term = stem(t.lower())
            try:
                qfs[term] += 1
            except KeyError:
                qfs[term] = 1
        k = 1.2 * ((1 - 0.75) + 0.75 * (doc.get_doc_len() / float(avg_dl)))
        bm25_ = 0.0;
        for qt in qfs.keys():
            n = 0
            if qt in df.keys():
                n = df[qt]
                f = doc.get_term_count(qt);
                delta = 1
                qf = qfs[qt]
                bm = math.log(1.0 / ((n) / (no_docs+1)), 2) * ((((1.2 + 1) * f) / (k + f) )+ delta)
                bm25_ += bm
        bm25s[doc.get_docid()] = bm25_
    return bm25s

def load_ranker(cfg_file):
    """
    Use this function to return the Ranker object to evaluate, 
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index.
    """
    return metapy.index.OkapiBM25(k1=1.575,b=0.75,k3=0.375)
    # return InL2Ranker(some_param=0.5)
    # return PL2Ranker(c_param=0.75)
    # return bm25()

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