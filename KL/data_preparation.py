docs_file_path = "Data/CISI.ALL"
queries_file_path = "Data/CISI.QRY"
ground_truth_file_path = "Data/CISI.REL"


def get_docs(n=-1):
    #if N =< 0 take all else take the first N
    with open(docs_file_path) as f:
        lines = ""
        for l in f.readlines():
            lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
        lines = lines.lstrip("\n").split("\n")
    doc_set = {}
    doc_id = ""
    doc_text = ""
    for l in lines:
        if l.startswith(".I"):
            doc_id = l.split(" ")[1].strip()
            if n >= 0 and int(doc_id) > n:
                break
        elif l.startswith(".X"):
            doc_set[doc_id] = doc_text.lstrip(" ")
            doc_id = ""
            doc_text = ""
        else:
            doc_text += l.strip()[3:] + " "  # The first 3 characters of a line can be ignored.
    return doc_set


def get_queries():
    with open(queries_file_path) as f:
        lines = ""
        for l in f.readlines():
            lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
        lines = lines.lstrip("\n").split("\n")

    qry_set = {}
    qry_id = ""
    for l in lines:
        if l.startswith(".I"):
            qry_id = l.split(" ")[1].strip()
        elif l.startswith(".W"):
            qry_set[qry_id] = l.strip()[3:]
            qry_id = ""
    return qry_set


def get_ground_truth():
    rel_set = {}
    with open(ground_truth_file_path) as f:
        for l in f.readlines():
            qry_id = l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[0]
            doc_id = l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[-1]
            if qry_id in rel_set:
                rel_set[qry_id].append(doc_id)
            else:
                rel_set[qry_id] = []
                rel_set[qry_id].append(doc_id)
    return rel_set

