import re

def parse_queries(path="./Data/CISI.QRY"):
    """
        Description: Function to parse 'CISI.QRY' file into a dictionary with the following format:
        {
            "<QUERY_ID_1>":"<QUERY_TEXT>",
            "<QUERY_ID_2>":"<QUERY_TEXT>",
            "<QUERY_ID_3>":"<QUERY_TEXT>",
            ...
        }

        return: dictionary
    """

    # initialize variables for queries
    queries = {}
    # Read queries file and split into queries list
    for q in re.split('\n.I ', open(path).read()):
        # remove all new lines and '.I' (ID) mark
        q = re.sub(' +', ' ', q.replace('.I', '').replace('\n', ' '))
        # extract query ID, search all characters until first occurrence of ' ' (space)
        q_id = re.search(r'([^\s]+)', q).group(0)
        # queries that has more than ID (.I) and query text (.W)
        if ' .B' in q:
            # all characters between '.W' and '.B'
            q_txt = re.search(r' .W.*?\.B ', q).group(0).replace('.W', '').replace('.B', '').rstrip().lstrip()
        # queries that has only ID (.I) and query text (.W)
        else:
            # all characters from '.W' to end
            q_txt = re.search(r' .W.*$', q).group(0).replace('.W', '').rstrip().lstrip()
        # store value as dictionary item
        queries[q_id] = q_txt

    return(queries)

def parse_documents(path="./Data/CISI.ALL"):

    """
        Description: Function to parse 'CISI.ALL' file into a dictionary with the following format:
        {
            'DOCUMENT_ID_0': {"title": '<DOCUMENT TITLE>', "body": '<DOCUMENT TEXT>'},
            'DOCUMENT_ID_1': {"title": '<DOCUMENT TITLE>', "body": '<DOCUMENT TEXT>'},
            ...
        }
        return: dictionary
    """

    # initialize variables for documents
    documents = {}
    # Read documents file and split into document dictionary
    for d in re.split('\n.I ', open(path).read()):
        # remove all new lines
        # d = re.sub(' +', ' ', d.replace('\n', ' '))
        # remove all new lines and '.I' (ID) mark
        d = re.sub(' +', ' ', d.replace('.I', '').replace('\n', ' '))
        # extract query ID, search all characters until first occurrence of ' ' (space)
        d_id = re.search(r'([^\s]+)', d).group(0)
        # extract document title, search all characters between '.T' and '.A'
        d_tit = re.search(r' .T.*?\.A ', d).group(0).replace('.T', '').replace('.A', '').rstrip().lstrip()
        # extract document text, search all characters between '.W' and '.X'
        d_txt = re.search(r' .W.*?\.X ', d).group(0).replace('.W', '').replace('.X', '').rstrip().lstrip()
        # store value as dictionary item
        documents[d_id] = {"title": d_tit, "body": d_txt}

    return documents

def parse_data_ground_truth(path="./Data/CISI.REL"):
    """
        Description: Function to parse 'CISI.REL' file into a dictionary with the following format:
        {
            'QUERY_ID_0': [<DOCUMENT_ID>, <DOCUMENT_ID>, <DOCUMENT_ID>, ...],
            'QUERY_ID_1': [<DOCUMENT_ID>, <DOCUMENT_ID>, <DOCUMENT_ID>, ...],
            ...
        }
        return: dictionary
    """

    # initialize a dictionary variables for a query-to-document relationship
    rel = {}

    # Read relationship file and for each query ID store all related document
    for row in open(path).read().splitlines():
        q_id, d_id, _, _ = row.split()

        if q_id not in rel.keys():
            rel[q_id] = []

        rel[q_id].append(d_id)

    return rel


    # return rel