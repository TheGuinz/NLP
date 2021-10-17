from operator import itemgetter
import math
import pandas as pd
import data_preparation as dp
from data_preprocessing import preprocessing, split_by_space
from language_model import LM
import glob
from pathlib import Path
from performance_measures import *
from Query import *


def get_query_likelihood_retrieval_score(query_words, docs_lm):

    # Analysis query words
    result = []

    for doc_lm in docs_lm:

        query_likelihood_score = 0
        lm = docs_lm[doc_lm]
        # score = log p(q|d) => score = sum [ c(t|q)*log(P(t|d)) ]

        for qw in query_words:
            c_w_q = query_words[qw]
            if qw in lm.keys():
                p_w_c = lm[qw]
            else:
                continue

            log_pd = float(math.log(float(p_w_c), 2))*c_w_q
            query_likelihood_score += log_pd

        # Add to result
        result.append((doc_lm, query_likelihood_score))
    # Sort & return
    return sorted(result, key=itemgetter(1), reverse=True)


def get_KL_retrieval_score(query_lm, docs_lm):

    # Analysis query words
    result = []

    for doc in docs_lm:

        kl_score = 0.0

        doc_lm = docs_lm[doc]
        # score += -p(t|q)*log(P(t|d))
        for qw in query_lm:
            ptq = query_lm[qw]
            if qw in doc_lm.keys():
                ptd = doc_lm[qw]
            else:
                ptd = 1
            log_ptd = math.log(ptd, 2)
            kl_score += -ptq * log_ptd
        # Add to result
        result.append((doc, kl_score))
    # Sort & return
    return sorted(result, key=itemgetter(1), reverse=False)


def get_feedback_lm_by_KL(top_n_docs, docs_lm, bg_lm, lamda=0.2):

    # Analysis query words
    cutoff_prob = 0.001
    feedback_lm = LM()
    word_list = []

    # add all words in all docs to one list
    for doc in top_n_docs:
        for word in docs_lm[doc]:
            if word not in word_list:
                word_list.append(word)
    sum = 0

    # for each word in the word list calc th prob
    for word in word_list:
        p_w_d = 0
        p_w_c = 0

        # get the word prob in the collection
        if word in bg_lm:
            p_w_c = math.log(bg_lm[word])

        # get the word prob in each doc ( if word is in the doc)
        for doc in top_n_docs:
            if word in docs_lm[doc]:
                p_w_d += math.log(docs_lm[doc][word])

        # calc the p(w|f) for each word to build feedback lam
        p_w_f = math.exp((float(p_w_d)/(float(1-lamda)*len(top_n_docs))) - float(p_w_c)*(float(lamda)/float(1-lamda)))

        # add the word to the lm only if the prob > cutff prob
        if p_w_f > cutoff_prob:
            sum += p_w_f
            feedback_lm.lm[word] = p_w_f

    # re normalized he feedback lm
    for word in feedback_lm.lm:
        feedback_lm.lm[word] = feedback_lm.lm[word]/sum
    return feedback_lm.lm


def get_feedback_lm_by_KL_new_approche(top_n_docs, docs_lm, bg_lm, lamda=0.2):

    # Analysis query words
    cutoff_prob = 0.001
    feedback_lm = LM()
    word_list = []
    n = len(top_n_docs)
    topn_sum = np.sum(list(range(1, n+1)))

    # add all words in all docs to one list
    for doc in top_n_docs:
        for word in docs_lm[doc]:
            if word not in word_list:
                word_list.append(word)
    sum = 0

    # for each word in the word list calc th prob
    for word in word_list:
        p_w_d = 0
        p_w_c = 0

        # get the word prob in the collection
        if word in bg_lm:
            p_w_c = math.log(bg_lm[word])

        # get the word prob in each doc ( if word is in the doc)
        i = 0
        for doc in top_n_docs:
            if word in docs_lm[doc]:
                p_w_d += math.log(docs_lm[doc][word])*(n-i)
                i += 1
        # calc the p(w|f) for each word to build feedback lam
        p_w_f = math.exp((float(p_w_d)/(float(1-lamda)*topn_sum)) - float(p_w_c)*(float(lamda)/float(1-lamda)))

        # add the word to the lm only if the prob > cutff prob
        if p_w_f > cutoff_prob:
            sum += p_w_f
            feedback_lm.lm[word] = p_w_f

    # re normalized he feedback lm
    for word in feedback_lm.lm:
        feedback_lm.lm[word] = feedback_lm.lm[word]/sum
    return feedback_lm.lm

def update_query_lm(query_lm, alpha, feedback_lm):

    # Analysis query words
    new_query_lm = LM()
    for word in feedback_lm:

        # update the query lm for words in feedback lm
        if word in query_lm:
            p_w = (1-alpha)*query_lm[word] + alpha*feedback_lm[word]
        else:

            p_w = alpha*feedback_lm[word]
        if p_w > 0:
            new_query_lm.lm[word] = p_w

    # update the query lm for words that not in feedback lm
    for word in query_lm:
        if word not in new_query_lm.lm:
            p_w = (1-alpha)*query_lm[word]
            new_query_lm.lm[word] = p_w

    return new_query_lm.lm


if __name__ == "__main__":


    bglm_file_name = 'BGLM.csv'

    if bglm_file_name == 'BGLM_without_stopwords.csv' :
        stop = True
    else:
        stop = False

    bglm = LM()
    bg_lm = bglm.load_lm_from_file(bglm_file_name)

    # load Lm of all docs

    docs_lm_path = 'DocsLm/'
    docs_list = [f for f in glob.glob("{0}*.csv".format(docs_lm_path))]
    docs_lm_dict = {}

    for doc in docs_list:
        print(f"add doc {doc} to lm dict")
        dlm = LM()
        doc_id = int((Path(doc).stem.split('_')[1].lstrip('0')))
        docs_lm_dict[doc_id] = dlm.load_lm_from_file(doc)

    N = 50  # number of retrieval docs to take from performance measures
    topN = 3  # number of docs to use in feedback
    alpha = 0.5
    lamda = 0.5

    queries = dp.get_queries()  # load all queries

    # select queries
    # [1, 7, 8] # for take all queries use list(range(1, len(queries)+1)) , for take fist m list(range(1,m)
    queries_num = list(range(1, len(queries)+1))
    ground_truth = dp.get_ground_truth()  # load ground truth

    # columns for data frames storing the results
    columns_total = ["Method", "AvgPr", "Recall", "lamda", "alpha"]
    rows_total = []

    queries_dict = {}

    # build query dict
    for query_num in queries_num:

        query = Query(query_num)
        query_term = queries[str(query_num)]
        query_term = preprocessing(query_term, stopwords=stop)
        words = split_by_space(query_term)
        query.SetWords(words)
        query.likelihood_retrieval_score = get_query_likelihood_retrieval_score(query.words, docs_lm_dict)

        qlm = LM()
        query.base_lm = qlm.unigram_ml_lm(query.words, query.GetQueryLen())
        query.KL_retrieval_score = get_KL_retrieval_score(query.base_lm, docs_lm_dict)
        queries_dict[query_num] = query

    # start performance measures

    rows_total = []
    queries_with_result = len(queries)+1

    precision_sum_ql = precision_sum_kl = precision_sum_fd = precision_sum_fd_new = 0
    total_retrieved_fb = total_relevant_fb = total_retrieved_fb_new = total_relevant_fb_new = 0
    total_retrieved_ql = total_relevant_ql = total_retrieved_kl = total_relevant_kl = 0

    # for const alpha and lambda values
    for query_num in queries_num:

        if str(query_num) in ground_truth.keys():

            # query likelihood retrieval
            likelihood_retrieval_score = queries_dict[query_num].likelihood_retrieval_score
            total_retrieved, total_relevant, avg_precision, N_docs = performance_measures(N,
                                                                                          likelihood_retrieval_score,
                                                                                          query_num,
                                                                                          ground_truth)

            precision_sum_ql += avg_precision
            total_retrieved_ql += total_retrieved
            total_relevant_ql += total_relevant

            # -----------------------------------
            # -----------------------------------

            # KL feedback + KL retrieval after QL retrieval

            feedback_lm = get_feedback_lm_by_KL(N_docs[:topN], docs_lm_dict, bg_lm, lamda)
            new_q_lm = update_query_lm(queries_dict[query_num].base_lm, alpha, feedback_lm)
            score = get_KL_retrieval_score(new_q_lm, docs_lm_dict)
            total_retrieved, total_relevant, avg_precision, N_docs = performance_measures(N,
                                                                                          score,
                                                                                          query_num,
                                                                                          ground_truth)

            precision_sum_fd += avg_precision
            total_retrieved_fb += total_retrieved
            total_relevant_fb += total_relevant

            # -----------------------------------
            # -----------------------------------

            # New KL feedback + KL retrieval QL retrieval

            feedback_lm = get_feedback_lm_by_KL_new_approche(N_docs[:topN], docs_lm_dict, bg_lm, lamda)
            new_q_lm = update_query_lm(queries_dict[query_num].base_lm, alpha, feedback_lm)

            score = get_KL_retrieval_score(new_q_lm, docs_lm_dict)
            total_retrieved, total_relevant, avg_precision, N_docs = performance_measures(N,
                                                                                          score,
                                                                                          query_num,
                                                                                          ground_truth)

            precision_sum_fd_new += avg_precision
            total_retrieved_fb_new += total_retrieved
            total_relevant_fb_new += total_relevant

            # -----------------------------------
            # -----------------------------------

            # KL retrieval

            kl_retrieval_score = queries_dict[query_num].KL_retrieval_score
            total_retrieved, total_relevant, avg_precision, N_docs = performance_measures(N,
                                                                                          kl_retrieval_score,
                                                                                          query_num,
                                                                                          ground_truth)

            precision_sum_kl += avg_precision
            total_retrieved_kl += total_retrieved
            total_relevant_kl += total_relevant

        else:
            queries_with_result -= 1

    row = ['simple LM', precision_sum_ql/queries_with_result,
           '{0}/{1}'.format(total_retrieved_ql, total_relevant_ql), lamda, alpha]
    rows_total.append(row)

    row = ['KL', precision_sum_kl / queries_with_result,
           '{0}/{1}'.format(total_retrieved_kl, total_relevant_kl), lamda, alpha]
    rows_total.append(row)

    row = ['Min KL', precision_sum_fd / queries_with_result,
           '{0}/{1}'.format(total_retrieved_fb, total_relevant_fb), lamda, alpha]
    rows_total.append(row)

    row = ['Min KL NEW', precision_sum_fd_new / queries_with_result,
           '{0}/{1}'.format(total_retrieved_fb_new, total_relevant_fb_new), lamda, alpha]

    rows_total.append(row)
    df_total = pd.DataFrame(rows_total, columns=columns_total)
    file_name = 'Top{0}.csv'.format(N)
    df_total.to_csv(file_name, encoding='utf-8', index=False)

    # for different alpha values and const lambda values
    alpha_points = np.linspace(0, 1, 11)
    rows_total = []

    for alpha in alpha_points:

        queries_with_result = len(queries) + 1
        precision_sum_ql = precision_sum_kl = precision_sum_fd = precision_sum_fd_new = 0
        total_retrieved_fb = total_relevant_fb = total_retrieved_fb_new = total_relevant_fb_new = 0
        total_retrieved_ql = total_relevant_ql = total_retrieved_kl = total_relevant_kl = 0

        for query_num in queries_num:

            if str(query_num) in ground_truth.keys():

                # query likelihood retrieval
                likelihood_retrieval_score = queries_dict[query_num].likelihood_retrieval_score
                total_retrieved, total_relevant, avg_precision, N_docs = performance_measures(N,
                                                                                              likelihood_retrieval_score,
                                                                                              query_num,
                                                                                              ground_truth)

                precision_sum_ql += avg_precision
                total_retrieved_ql += total_retrieved
                total_relevant_ql += total_relevant

                # -----------------------------------
                # -----------------------------------

                # KL feedback + KL retrieval after QL retrieval

                feedback_lm = get_feedback_lm_by_KL(N_docs[:topN], docs_lm_dict, bg_lm, lamda)
                new_q_lm = update_query_lm(queries_dict[query_num].base_lm, alpha, feedback_lm)
                score = get_KL_retrieval_score(new_q_lm, docs_lm_dict)
                total_retrieved, total_relevant, avg_precision, N_docs = performance_measures(N,
                                                                                              score,
                                                                                              query_num,
                                                                                              ground_truth)

                precision_sum_fd += avg_precision
                total_retrieved_fb += total_retrieved
                total_relevant_fb += total_relevant

                # -----------------------------------
                # -----------------------------------

                # New KL feedback + KL retrieval QL retrieval

                feedback_lm = get_feedback_lm_by_KL_new_approche(N_docs[:topN], docs_lm_dict, bg_lm, lamda)
                new_q_lm = update_query_lm(queries_dict[query_num].base_lm, alpha, feedback_lm)

                score = get_KL_retrieval_score(new_q_lm, docs_lm_dict)
                total_retrieved, total_relevant, avg_precision, N_docs = performance_measures(N,
                                                                                              score,
                                                                                              query_num,
                                                                                              ground_truth)

                precision_sum_fd_new += avg_precision
                total_retrieved_fb_new += total_retrieved
                total_relevant_fb_new += total_relevant

                # -----------------------------------
                # -----------------------------------

                # KL retrieval

                kl_retrieval_score = queries_dict[query_num].KL_retrieval_score
                total_retrieved, total_relevant, avg_precision, N_docs = performance_measures(N,
                                                                                              kl_retrieval_score,
                                                                                              query_num,
                                                                                              ground_truth)

                precision_sum_kl += avg_precision
                total_retrieved_kl += total_retrieved
                total_relevant_kl += total_relevant

            else:
                queries_with_result -= 1

        row = ['simple LM', precision_sum_ql / queries_with_result,
               '{0}/{1}'.format(total_retrieved_ql, total_relevant_ql), lamda, alpha]
        rows_total.append(row)

        row = ['KL', precision_sum_kl / queries_with_result,
               '{0}/{1}'.format(total_retrieved_kl, total_relevant_kl), lamda, alpha]
        rows_total.append(row)

        row = ['Min KL', precision_sum_fd / queries_with_result,
               '{0}/{1}'.format(total_retrieved_fb, total_relevant_fb), lamda, alpha]
        rows_total.append(row)

        row = ['Min KL NEW', precision_sum_fd_new / queries_with_result,
               '{0}/{1}'.format(total_retrieved_fb_new, total_relevant_fb_new), lamda, alpha]

        rows_total.append(row)

    df_alpha = pd.DataFrame(rows_total, columns=columns_total)
    df_alpha.sort_values(by=['alpha'], inplace=True)

    file_name = 'Top{0}_alpha_values.csv'.format(N)
    df_alpha.to_csv(file_name, encoding='utf-8', index=False)

    # for different lambda values and const alpha values
    alpha = 0.5
    lamda_points = np.linspace(0, 1, 11)
    rows_total = []

    for lamda in lamda_points:

        if lamda >= 1:
            lamda = 0.98

        queries_with_result = len(queries) + 1
        precision_sum_ql = precision_sum_kl = precision_sum_fd = precision_sum_fd_new = 0
        total_retrieved_fb = total_relevant_fb = total_retrieved_fb_new = total_relevant_fb_new = 0
        total_retrieved_ql = total_relevant_ql = total_retrieved_kl = total_relevant_kl = 0

        for query_num in queries_num:

            if str(query_num) in ground_truth.keys():

                # query likelihood retrieval
                likelihood_retrieval_score = queries_dict[query_num].likelihood_retrieval_score
                total_retrieved, total_relevant, avg_precision, N_docs = performance_measures(N,
                                                                                              likelihood_retrieval_score,
                                                                                              query_num,
                                                                                              ground_truth)

                precision_sum_ql += avg_precision
                total_retrieved_ql += total_retrieved
                total_relevant_ql += total_relevant

                # -----------------------------------
                # -----------------------------------

                # KL feedback + KL retrieval after QL retrieval

                feedback_lm = get_feedback_lm_by_KL(N_docs[:topN], docs_lm_dict, bg_lm, lamda)
                new_q_lm = update_query_lm(queries_dict[query_num].base_lm, alpha, feedback_lm)
                score = get_KL_retrieval_score(new_q_lm, docs_lm_dict)
                total_retrieved, total_relevant, avg_precision, N_docs = performance_measures(N,
                                                                                              score,
                                                                                              query_num,
                                                                                              ground_truth)

                precision_sum_fd += avg_precision
                total_retrieved_fb += total_retrieved
                total_relevant_fb += total_relevant

                # -----------------------------------
                # -----------------------------------

                # New KL feedback + KL retrieval QL retrieval

                feedback_lm = get_feedback_lm_by_KL_new_approche(N_docs[:topN], docs_lm_dict, bg_lm, lamda)
                new_q_lm = update_query_lm(queries_dict[query_num].base_lm, alpha, feedback_lm)

                score = get_KL_retrieval_score(new_q_lm, docs_lm_dict)
                total_retrieved, total_relevant, avg_precision, N_docs = performance_measures(N,
                                                                                              score,
                                                                                              query_num,
                                                                                              ground_truth)

                precision_sum_fd_new += avg_precision
                total_retrieved_fb_new += total_retrieved
                total_relevant_fb_new += total_relevant

                # -----------------------------------
                # -----------------------------------

                # KL retrieval

                kl_retrieval_score = queries_dict[query_num].KL_retrieval_score
                total_retrieved, total_relevant, avg_precision, N_docs = performance_measures(N,
                                                                                              kl_retrieval_score,
                                                                                              query_num,
                                                                                              ground_truth)

                precision_sum_kl += avg_precision
                total_retrieved_kl += total_retrieved
                total_relevant_kl += total_relevant

            else:
                queries_with_result -= 1

        row = ['simple LM', precision_sum_ql / queries_with_result,
               '{0}/{1}'.format(total_retrieved_ql, total_relevant_ql), lamda, alpha]
        rows_total.append(row)

        row = ['KL', precision_sum_kl / queries_with_result,
               '{0}/{1}'.format(total_retrieved_kl, total_relevant_kl), lamda, alpha]
        rows_total.append(row)

        row = ['Min KL', precision_sum_fd / queries_with_result,
               '{0}/{1}'.format(total_retrieved_fb, total_relevant_fb), lamda, alpha]
        rows_total.append(row)

        row = ['Min KL NEW', precision_sum_fd_new / queries_with_result,
               '{0}/{1}'.format(total_retrieved_fb_new, total_relevant_fb_new), lamda, alpha]

        rows_total.append(row)

    df_lamda = pd.DataFrame(rows_total, columns=columns_total)
    df_lamda.sort_values(by=['lamda'], inplace=True)

    file_name = 'Top{0}_lamda_values.csv'.format(N)
    df_lamda.to_csv(file_name, encoding='utf-8', index=False)


