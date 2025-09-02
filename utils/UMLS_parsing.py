import pandas as pd
import pickle as pkl
from tqdm.auto import tqdm
import sys
import os

from quickumls import QuickUMLS

'''2023ABquickumls_fp is the directory where the QuickUMLS data files are installed.
overlapping_criteria (optional, default: "score") is the criteria used to deal with overlapping concepts; choose "score" if the matching score of the concepts should be consider first, "length" if the longest should be considered first instead.
threshold (optional, default: 0.7) is the minimum similarity value between strings.
similarity_name (optional, default: "jaccard") is the name of similarity to use. Choose between "dice", "jaccard", "cosine", or "overlap".
window (optional, default: 5) is the maximum number of tokens to consider for matching.
accepted_semtypes (optional, default: see constants.py) is the set of UMLS semantic types concepts should belong to. Semantic types are identified by the letter "T" followed by three numbers (e.g., "T131", which identifies the type "Hazardous or Poisonous Substance").
'''
'''
# T033: finding, 
# T184: symptom, 
# T060: diagnostoc proc, 
# T130: diagnostic aid, 
# T059, T034: lab, 
# T047: disease,
# T023: organ, 
# T191: newplastic
# T061: therapeutic proc
# T200, T201, T203: drug or clinical 
# orga|T032|Organism Attribute
# orgf|T040|Organism Function
# orgm|T001|Organism
# ortf|T042|Organ or Tissue Function
# patf|T046|Pathologic Function
# phob|T072|Physical Object
# phsf|T039|Physiologic Function
# phsu|T121|Pharmacologic Substance
# aapp|T116|Amino Acid, Peptide, or Protein
# acab|T020|Acquired Abnormality
# amas|T087|Amino Acid Sequence
# anab|T190|Anatomical Abnormality#
# anst|T017|Anatomical Structure
# antb|T195|Antibiotic
# bacs|T123|Biologically Active Substance
# bact|T007|Bacterium
# bdsu|T031|Body Substance
# bdsy|T022|Body System
# biof|T038|Biologic Function
# blor|T029|Body Location or Region
# bmod|T091|Biomedical Occupation or Discipline
# bpoc|T023|Body Part, Organ, or Organ Component
# bsoj|T030|Body Space or Junction
# hlca|T058|Health Care Activity
# hops|T131|Hazardous or Poisonous Substance
# horm|T125|Hormone
'''

def umls_parsing(header_in, header_out, files_list):
    '''
    keeping only the following semantic types -- explanation above
    '''
    stypes = ['T033', 'T184', 'T060', 'T130', 'T059', 'T034', 'T047',  'T023', 'T191',  'T061',  'T200', 'T201', 'T203', 'T032', 
    'T040',  'T001',  'T042',  'T046',  'T072',  'T039',  'T121',  'T116',  'T020',  'T087',  'T190',  'T017',  'T195', 
    'T123', 'T007',  'T031',  'T022',  'T038',  'T029',  'T091',  'T023',  'T030', 'T058', 'T131',  'T125']
    matcher = QuickUMLS('UMLS', threshold=0.75, accepted_semtypes=stypes)
    print('matcher created')
    sys.stdout.flush()

    sys.stdout.flush()

    for idx in tqdm(range(len(files_list))):

        fname = files_list[idx]
        f = open(fname, 'r')
        lines = f.readlines()
        f.close()
        fname = fname.replace(header_in, header_out) #create file path to store pkl output
        fname = fname.replace('.txt', '.pkl') # change type too
        lst = []
        try:
            for line in lines:
                dct = {'text': line}
                out = matcher.match(line, best_match=True, ignore_syntax=False)
                dct['entities'] = out
                lst.append(dct)
            pkl.dump(lst, open(fname, 'wb'))
        except:
            print("problem with ", fname)
if __name__ == "__main__":
    '''
    arguments
    1: data_directoty
    2: output folder to store pkl files
    3: train_val_test_files - dct of train, test, validation files of text in the data folder
    
    '''
    print('command line arguments', str(sys.argv))


    header_in = sys.argv[1]
    header_out = sys.argv[2]
    files_dct = sys.argv[3]

    dct = pkl.load(open(files_dct, "rb"))
    files_list = list(dct["train"])+ list(dct["val"])+ list(dct["test"]) #all files need to be parsed
    umls_parsing(header_in, header_out, files_list)

'''
python3 UMLS_parsing.py data_dir out_dir train_val_test_files >> log/umls_parsing.txt &
'''