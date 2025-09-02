from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm.auto import tqdm
import sys

from datasets import load_dataset

def main(save_dir):
    pubmed = load_dataset('pubmed', streaming=True)

    pcount=0

    for idx, entry in enumerate(pubmed['train']):
        txt = entry['MedlineCitation']['Article']['Abstract']['AbstractText']
        if len(txt)>0 and 'prostate' in txt.lower() and ('cancer' in txt.lower() or 'carcinoma' in txt.lower()):
            pcount+=1
            sentences = sent_tokenize(txt)
            txt = '\n'.join(sentences)
            f = open(save_dir+str(pcount)+'.txt', 'w')
            f.write(txt)
            f.close()
        if idx % 5000==0:
            print('train', idx, pcount)
            sys.stdout.flush()

    for idx, entry in enumerate(pubmed['test']):
        txt = entry['MedlineCitation']['Article']['Abstract']['AbstractText']
        if len(txt)>0 and 'prostate' in txt.lower() and ('cancer' in txt.lower() or 'carcinoma' in txt.lower()):
            pcount+=1
            sentences = sent_tokenize(txt)
            txt = '\n'.join(sentences)
            f = open(save_dir+str(pcount)+'.txt', 'w')
            f.write(txt)
            f.close()
        if idx % 5000==0:
            print('test', idx, pcount)
            sys.stdout.flush()

    for idx, entry in enumerate(pubmed['val']):
        txt = entry['MedlineCitation']['Article']['Abstract']['AbstractText']
        if len(txt)>0 and 'prostate' in txt.lower() and ('cancer' in txt.lower() or 'carcinoma' in txt.lower()):
            pcount+=1
            sentences = sent_tokenize(txt)
            txt = '\n'.join(sentences)
            f = open(save_dir+str(pcount)+'.txt', 'w')
            f.write(txt)
            f.close()
        if idx % 5000==0:
            print('val', idx, pcount)
            sys.stdout.flush()

if __name__ == "__main__":
    '''
    arguments
    1: save directory for data storage
    '''
    print('command line arguments', str(sys.argv))


    save_dir = sys.argv[1]
    main(save_dir)


'''
python3 pubmed_prostate_cancer_data_extraction.py save_dir  >> log/pudmed_data_extraction.txt &
'''