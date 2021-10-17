class Query:

    def __init__(self, queryid):
        self.queryid = queryid
        self.len = 0  # All word counts in doc
        self.words = {}  # Word and it's cnt
        self.likelihood_retrieval_score = []
        self.KL_retrieval_score = []
        self.base_lm = {}

    def SetWords(self, words):
        for word in words:
            # Add tf of word
            self.words.setdefault(word, 0)
            self.words[word] += 1
        # Add doc len
        self.len = len(words)

    def GetQueryTf(self, word, fuzzy=False):
        if not fuzzy:
            return self.words.get(word, 0)
        else:
            tf = 0
            for (key, cnt) in self.words.items():
                if word in key:
                    tf += cnt
            return tf

    def GetQueryLen(self):
        return self.len

    def GetQueryID(self):
        return self.queryid

