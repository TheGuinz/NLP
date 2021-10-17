import data_preparation as dp
from data_preprocessing import preprocessing, split_by_space
from language_model import *


class Collection:
    def __init__(self):
        self.len = 0  # All word counts in collection
        self.words = {}
        self.docs = {}

    def AddWords(self, words):
        for word in words:
            # Add tf of word
            self.words.setdefault(word, 0)
            self.words[word] += 1
        # Add doc len
        self.len += len(words)

    def GetCollTf(self, word, fuzzy=False):
        if not fuzzy:
            return self.words.get(word, 0)
        else:
            tf = 0
            for (key, cnt) in self.words.items():
                if word in key:
                    tf += cnt
            return tf

    def GetCollLen(self):
        return self.len

    def AddDoc(self, doc):
        self.docs[doc.doc_id] = doc


def create_collection_lm(stopwords=False):

    docs = dp.get_docs(-1)  # load al docs is collection

    collection = Collection()  # create collection object

    for doc in docs:  # add all docs word's to collection
        words = preprocessing(docs[doc], stopwords)  # pre processing the words
        words = split_by_space(words)  # split string by space
        collection.AddWords(words)  # add qords to collection

    # Create a placeholder for model
    clm = LM()
    lm = clm.unigram_ml_lm(collection.words, collection.len)  # create unigram lm with ML

    # save lm to disc as csv file

    if stopwords:
        filename = 'BGLM.csv'

    else:
        filename = 'BGLM_without_stopwords.csv'

    with open(filename, 'w', encoding="utf-8") as f:
        for key in lm.keys():
            f.write("%s %s\n" % (key, lm[key]))

    return filename

if __name__ == "__main__":

    create_collection_lm(stopwords=False)




