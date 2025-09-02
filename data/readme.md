Place all training data here

The code is designed to train on million of clinical notes and reports in resource-constrained scenario. 
The data loaders assume that loading all data at once is not possible.
/data/ folder is expected to have several text files; each file represents one patient or one abstract of PubMed (any other file organization is ok)
The data loaders loop through the list of file paths and read each file and form text snippets of N number of sentences

Snippet length may need to be different given distinct linguistic styles of notes and abstracts. It is advised to divide the data/ into two subfolders; notes/ and abstracts/