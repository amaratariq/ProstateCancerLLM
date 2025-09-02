import pickle as pkl
import pandas as pd
import numpy as np
import torch
import sys
import os
import tqdm

from transformers import pipeline
from transformers import PreTrainedTokenizerFast
from transformers import  GPT2LMHeadModel,  PreTrainedTokenizerFast
from tokenizers import models, Tokenizer

from utils.data_loader_phase2 import get_terms, filter_text_with_clinical_terms, read_sentences, belongs_to_term
def load_model_and_tokenizer(token_path, model_path):
    '''
    model_path: model name on huggingface or saved model
    token_path: folder path where tokenizer is available
    '''
    tokenizer = Tokenizer(models.BPE())
    tokenizer = Tokenizer.from_file(token_path+"config.json")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    tokenizer.model_max_length = 1024
    print('tokenizer loaded and wrapped')
    sys.stdout.flush()


    model = GPT2LMHeadModel.from_pretrained(model_path) 
    model.resize_token_embeddings(len(tokenizer))
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")
    print('model loaded')
    sys.stdout.flush()

    #match special token ids between model and tokenizer
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.mask_token_id = tokenizer.mask_token_id
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model



def get_labelbatch(inputbatch, termsbatch, tokenizer):

    labels = torch.clone(inputbatch["input_ids"]) #same as input text ids 
    terms_tokens = []
    for idx in range(inputbatch["input_ids"].shape[0]):

        terms = termsbatch[idx]
        
        a = torch.randint(0, len(terms), (int(0.5*len(terms)), )) #
        a = [True if i in a else False for i in range(len(terms))]
        if any(a):
            terms = np.array(terms)[a]
        else:
            terms = [terms[np.random.choice([i for i in range(len(terms))])]]
        ttoks = []
        for idx2 in range(inputbatch["input_ids"].shape[1]):
            
            pos = inputbatch["offset_mapping"][idx, idx2]
            pos = pos.numpy()
            if pos[1]!=0: #not padded token 
                
                if any([belongs_to_term(pos, t_pos) for t_pos in terms]):
                    ttoks.append(inputbatch["input_ids"][idx, idx2].clone().item())
                    inputbatch["input_ids"][idx, idx2] = tokenizer.mask_token_id #mask from input
                    inputbatch["attention_mask"][idx, idx2] = 0
                else:
                    labels[idx, idx2]=-100 #ignore from loss computation

        terms_tokens.append(ttoks)

    return labels, terms_tokens


def get_labelbatch_wo_offset(inputbatch, termsbatch, tokenizer):

    labels = torch.clone(inputbatch["input_ids"]) #same as input text ids 
    for idx in range(inputbatch["input_ids"].shape[0]):

        terms = termsbatch[idx]
        #pick one term only
        term_for_masking = terms[np.random.choice([i for i in range(len(terms))])]
        tokens_for_masking = torch.tensor(tokenizer(term_for_masking[0])["input_ids"][1:])#drop the initial token of biogpt
        for idx2 in range(inputbatch["input_ids"].shape[1]-len(tokens_for_masking)):
            if all(inputbatch["input_ids"][idx][idx2:idx2+len(tokens_for_masking)] == tokens_for_masking):
                inputbatch["input_ids"][idx][idx2:idx2+len(tokens_for_masking)]=tokenizer.mask_token_id
                inputbatch["attention_mask"][idx][idx2:idx2+len(tokens_for_masking)]=0
                labels[idx2:idx2+len(tokens_for_masking)]=-100 #ignore from loss computation
                break
    return labels, tokens_for_masking

def prep_data(sentences_w_terms, tokenizer, return_offsets_mapping=True):
    txt = [sentences_w_terms[idx]["text"][:-1] for idx in range(len(sentences_w_terms))]
    terms = [sentences_w_terms[idx]["terms"] for idx in range(len(sentences_w_terms))]
    if return_offsets_mapping:
        inputbatch = tokenizer(txt, padding='max_length', max_length=128, truncation=True, return_tensors="pt", return_offsets_mapping=return_offsets_mapping)
        labelbatch, terms_tokens = get_labelbatch(inputbatch, terms, tokenizer)
    else:
        inputbatch = tokenizer(txt, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
        labelbatch, terms_tokens = get_labelbatch_wo_offset(inputbatch, terms, tokenizer)
    return inputbatch, labelbatch, terms_tokens

def recallatK(gt, preds, K=10):
    mt = 0
    for i in range(len(gt)):
        if gt[i] in preds[i][:K]:
            mt+=1
    return mt/len(gt)

if __name__ == "__main__":
    '''
    arguments
    1: model_path
    2: token_path
    3: train_val_test_files - dct of train, test, validation files of text in the data folder
    
    '''
    print('command line arguments', str(sys.argv))

    model_path = sys.argv[1]
    token_path = sys.srgv[2]
    files_dct = sys.argv[3]

    tokenizer, model = load_model_and_tokenizer(token_path, model_path)

    dct = pkl.load(open(files_dct, "rb"))
    files_test = dct["test"]
    max_length = 256

    sentences_val, MRN_NO_VAL = read_sentences([], files_test, 0,  loop_through_files=False)

    recalls10 = []
    recalls5 = []
    recalls1 = []
    no_mask = 0

    mask_filler = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    for idx in tqdm(range(0, len(sentences_val))): 
        inputbatch, labels, terms_tokens = prep_data([sentences_val[idx]], tokenizer, return_offsets_mapping=True)
            
        txt = tokenizer.decode(inputbatch["input_ids"][0])
        txt = txt.split('<|endoftext|>')[0].replace('Ġ', '')
        ##others
        terms_tokens = terms_tokens[0]

        no_of_masks = len([m for m in inputbatch["input_ids"][0] if m==tokenizer.mask_token_id])#len(terms_tokens)


        if no_of_masks>0:
            answers = mask_filler(txt, top_k=10)
            answer_tokens = []

            if no_of_masks>1:
                for i in range(no_of_masks):
                    toks = [answers[0][t]['token'] for t in range(10)] #token, token_str
                    answer_tokens.append(toks)
            else:
                toks = [answers[t]['token'] for t in range(10)] #token, token_str
                answer_tokens.append(toks)

            rec10 = recallatK(terms_tokens, answer_tokens, K=10)#(terms_tokens[0], answer_tokens, K=10)
            rec5 = recallatK(terms_tokens, answer_tokens, K=5)
            rec1 = recallatK(terms_tokens, answer_tokens, K=1)
            recalls10.append(rec10)
            recalls5.append(rec5)
            recalls1.append(rec1)
        else:
            no_mask+=1
        if idx%100==0:
            print(np.mean(recalls10), np.mean(recalls5), np.mean(recalls1), no_mask)        
    print(np.mean(recalls10), np.mean(recalls5), np.mean(recalls1), no_mask)

'''
python3 masked_clinical_term_prediction.py model_path token_path  >> log/clinical_term_retrieval.txt &
'''