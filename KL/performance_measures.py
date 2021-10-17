import numpy as np


def get_performance_measures(result, query_id, ground_truth):

    precision_at = []
    if str(query_id) in ground_truth.keys():
        ground_truth = ground_truth[str(query_id)]
        total_retrieved, total_relevant = get_recall_at(result, ground_truth)

        for i in range(0, len(result)):
            tp = 0
            for j in range(0, i+1):
                if str(result[j]) in ground_truth:
                    tp += 1
            if str(result[j]) in ground_truth:
                precision_at.append(float(tp/(i+1)))
        if len(precision_at) == 0:
            non_interpolated_average_precision= 0
        else:
            non_interpolated_average_precision = float(np.sum(precision_at))/len(precision_at)
    else:
        total_retrieved = total_relevant = non_interpolated_average_precision = 0

    return total_retrieved, total_relevant, non_interpolated_average_precision


def get_recall_at(result, ground_truth):

    total_retrieved = 0
    for res in result:
        if str(res) in ground_truth:
            total_retrieved += 1

    if len(ground_truth) > len(result):
        total_relevant = len(result)

    else:
        total_relevant = len(ground_truth)

    return total_retrieved, total_relevant

def performance_measures(N, score_dict,query_num, ground_truth):

    N_docs = []
    for d in range(N):
        N_docs.append(score_dict[d][0])
    total_retrieved, total_relevant, avg_precision = get_performance_measures(N_docs, query_num, ground_truth)
    return total_retrieved, total_relevant, avg_precision , N_docs

