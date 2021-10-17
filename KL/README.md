
run build_BG_LM.py - contain the collection object and function to  build the collection LM

run build_doc_LM.py - contain the doc object and function to build LM for each document in the collection(you must have a collection LM before running this file)

data_preparation.py - is used to load the dataset data

data_preprocessing.py - is used to do pre-processing on the data

language_model.py - contain the LM object:

 1.unigram_ml_lm -  build unigram LM with M.L
 
 2.unigram_dirichlet_smooth_lm - build unigram LM with dirichlet smooth 

 3.load_lm_from_file - load LM form csv file

Query.py - contains the query object.

performance_measures.py - used to do performance measures like recall and precision

information_retrieval_main.py - run full cycle and save the result to CSV files also contain the retrieval function used in this project

 Language_Modeling_Approach_to_Information_Retrieval.ipynb- notebook sumerzie our project
  
