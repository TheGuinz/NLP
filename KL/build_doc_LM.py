import data_preparation as dp
from build_BG_LM import Collection
from language_model import *
from data_preprocessing import preprocessing, split_by_space


class Doc:
    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.len = 0  # All word counts in doc
        self.words = {}  # Word and it's cnt

    def SetWords(self, words):
        for word in words:
            # Add tf of word
            self.words.setdefault(word, 0)
            self.words[word] += 1
        # Add doc len
        self.len = len(words)

    def GetDocLen(self):
        return self.len

    def GetDocID(self):
        return self.doc_id


def create_docs_lm(bg_lm_pat, stopwords=False):

    # Create a placeholder for model
    K = -1
    docs = dp.get_docs(K)  # K <=0 load all docs ,
    collection = Collection()  # Collection object to store all docs
    docs_lm_path = 'DocsLM/'

    bglm = LM()  # load background LM for dirichlet smoothing
    bg_lm = bglm.load_lm_from_file(bg_lm_pat)

    total_word_count = 0

    for doc in docs:

        words = preprocessing(docs[doc], stopwords) # pre processing  doc
        words = split_by_space(words)  # split text to words  by space
        total_word_count += len(words)  # add doc len to the total word count
        d = Doc(int(doc))  # init doc
        d.SetWords(words)  # set doc word's
        collection.AddWords(words)  # add doc word's to collection
        collection.AddDoc(d)  # add docs to collection

    avg_doc_size = int(collection.len / len(collection.docs)) # set u in dirichlet to avg doc size

    for doc in docs:
        dlm = LM()  # create LM for each doc
        lm = dlm.unigram_dirichlet_smooth_lm(collection.docs[int(doc)].words,
                                             collection.docs[int(doc)].len,
                                             bg_lm.keys(),
                                             bg_lm, avg_doc_size)  # calc lm using dirichlet smoothing

        filename = '{0}Doc_{1}.csv'.format(docs_lm_path, doc.zfill(6)) # save lm as csv file
        with open(filename, 'w', encoding="utf-8") as f:
            for key in lm.keys():
                f.write("%s %s\n" % (key, lm[key]))

    return docs_lm_path


if __name__ == "__main__":

    path = 'BGLM.CSV'
    create_docs_lm(path, stopwords=False)



