import pickle as pkl
import sys
import numpy as np
import time
from striprtf.striprtf import rtf_to_text
import re

from tokenizers import (
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)



def apply_rtf_to_text(txt):
    try:
        return rtf_to_text(txt).encode('utf-8', 'replace').decode()
    except:
        return "unknown"
def clean_abstract(txt):
    txt = re.sub(r"<...>|</...>|&|#|\]|\[", "", txt)
    return txt
def remove_long_numerics(txt):
    try:
        txt = re.sub("\d{5,15}", "" , txt)
    except:
        print("long number removal failed")
        print(txt)
        sys.stdout.flush()
    return txt


def get_training_corpus(files_list, jump=10000):
    for idx in range(0, len(files_list), jump):
        texts = np.array([])
        for ii in range(idx, idx+jump):
            fname = files_list[ii]
            with open(fname) as f:
                text = "\n".join(f.readlines())
            texts.append(text)
        yield texts


if __name__ == "__main__":
    '''
    arguments
    1: save_dir
    2: train_val_test_files - dct of train, test, validation files of text in the data folder
    
    '''
    print('command line arguments', str(sys.argv))


    save_dir = sys.argv[1]
    files_dct = sys.argv[2]

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(vocab_size=50000, special_tokens=["<|endoftext|>"])

    print("tokenizer set")
    sys.stdout.flush()
    
    dct = pkl.load(open(files_dct, "rb"))
    files_list = dct["train"]

    tokenizer.train_from_iterator(get_training_corpus(files_list), trainer=trainer)
    print("training done")
    st = time.time()
    sys.stdout.flush()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.save(save_dir+'tokenizer.json')
    print("tokenizer saved")
    sys.stdout.flush()
    
'''
python3 tokenizer_trainig.py save_dir train_val_test_files >> log/tokenizer_training.txt &
'''