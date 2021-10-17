class LM:

    def __init__(self):
        self.lm = {}
        # Param u for DIR

    def unigram_ml_lm(self, vocabulary, v_len):
        for word in vocabulary:
            cnt = vocabulary[word]
            self.lm[word] = float(cnt / v_len)
        return self.lm

    def unigram_dirichlet_smooth_lm(self, vocabulary, doc_len, coll_vocabulary, collection_lm, u=200):
        for word in coll_vocabulary:
            if word in vocabulary.keys():
                cnt = vocabulary[word]
            else:
                cnt = 0
            p_w_c = collection_lm[word]
            self.lm[word] = float((cnt + u * p_w_c) / (doc_len + u))
        return self.lm

    def load_lm_from_file(self, file_path):
        lm = {}
        with open(file_path) as f:
            for l in f.readlines():
                word = l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[0]
                prob = float(l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[1])
                lm[word] = prob
        self.lm = lm
        return lm

