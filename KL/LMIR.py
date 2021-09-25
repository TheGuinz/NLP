import string
from operator import itemgetter
import math

punc = ",./<>?;'\":\`~!@#$%^&*_-+=()"
# table = string.maketrans(punc, " "*len(punc))
table = bytes.maketrans(str.encode(punc), b" "*len(punc))

def analysis_text(text, stemming=False):
    # Lower and remove punc
    text = text.lower().translate(table)
    # Split by space
    return text.split()

def analysis_url(url):
    # Lower and remove /....
    url = urllib.parse.unquote(url)
    # Remove ( and )
    url = url.replace("(", "").replace(")", "")
    text = url.lower().translate(table)
    return text.split()

class Doc:
    def __init__(self, docid):
        self.docid = docid
        self.len = 0 # All word counts in doc
        self.words = {} # Word and it's cnt

    def SetWords(self, words):
        for word in words:
            # Add tf of word
            self.words.setdefault(word, 0)
            self.words[word] += 1
        # Add doc len
        self.len = len(words)

    def GetDocTf(self, word, fuzzy=False):
        if not fuzzy:
            return self.words.get(word, 0)
        else:
            tf = 0
            for (key, cnt) in list(self.words.items()):
                if word in key:
                    tf += cnt
            return tf

    def GetDocLen(self):
        return self.len

    def GetDocID(self):
        return self.docid

class Collection:
    def	__init__(self):
       	self.len = 0 # All word counts in collection
        self.words = {}

    def AddWords(self, words):
       	for word in words:
       	    # Add tf of word
       	    self.words.setdefault(word, 0)
       	    self.words[word] +=	1
	# Add doc len
       	self.len += len(words)

class LMIR:
    def __init__(self):
        self.coll = Collection()
        self.docs = []
        # Param lamada for LMIR.JM
        self.lmd_short = 0.65
        self.lmd_long = 0.25
        # Param u for DIR
        self.u = 2000

    def AddDocText(self, docid, text):
        # Analysis text words
        words = analysis_text(text)
        # Collection words
        self.coll.AddWords(words)
        # Doc words
        doc = Doc(docid)
        doc.SetWords(words)
        self.docs.append(doc)

    def AddDocUrl(self, docid, url):
        # Analysis url words
        words = analysis_url(url)
        # Collection words
        self.coll.AddWords(words)
       	# Doc words
        doc = Doc(docid)
        doc.SetWords(words)
        self.docs.append(doc)

    # Rank doc in colls according to query, score by Kullback-Leibler Divergence(KLD)
    def RankKL(self, query, fuzzy=False):
        # Analysis query words
        qws = analysis_text(query)
        # Score each doc
        result = []
        for doc in self.docs:
            score = 0.0
            # score += -p(t|q)*log(P(t|d))
            for qw in qws:
                ptq = float(1) / float(len(qws))
                ptd = float(doc.GetDocTf(qw, fuzzy))/float(doc.GetDocLen())
                if ptd == 0.0:
                    continue
                lptd = math.log(ptd, 2)
                score += -ptq*lptd
            # Add to result
            result.append((doc.GetDocID(), score))
        # Sort & return
        return sorted(result, key=itemgetter(1), reverse=True)

    def test(self):
        # Print collection
        print(self.coll.len)
        print(self.coll.words)
        # Print doc
        for doc in self.docs:
            print(doc.docid)
            print(doc.len)
            print(doc.words)
