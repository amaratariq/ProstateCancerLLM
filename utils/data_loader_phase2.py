import numpy as np
import pandas as pd
import pickle as pkl
import os
import sys
import transformers
import torch
from torch.utils.data import Dataset

SENTENCES_LIM = 1e4
df_terms = pd.read_csv("../data/parsed/clinical_term_distribution_preferred_only.csv")
clinical_terms = [df_terms.Term.values[i] for i in range(len(df_terms)) if df_terms.DocCount.values[i]<=5000]
print("Number of clinical terms", len(clinical_terms))

def get_terms(ents, sim_thresh=1.0):
    terms = []
    start_indices = []
    not_to_include = ['patient', 'pt', 'date', 'general', 'revision', 'unable', 'possible', 'body']
    not_to_include_semtypes = []
    for ent in ents:
        for e in ent:
            if e["preferred"]==1 and e["similarity"]>=sim_thresh:
                t = (e["ngram"], e["start"], e["end"])
                '''
                several reasons to not include this particular clinical term
                t[0] not in terms - no already included
                t[0].upper()!=t[0] - not numbers and special chars only
                t[0].lower() not in not_to_include - not in terms to be filtered
                t[0] in clinical_terms - in selected terms
                e["semtypes"] not in not_to_include_semtypes - semtype is allowed - if certain types are to be excluded
                t[1]<375 - some are mistakenly marked too long
                only preferred and >sim_thresh terms being included
                t[1]>1 - mask at the frist index resutls in Nan loss

                '''
                
                if t not in terms and t[0].upper()!=t[0] and t[0].lower() not in not_to_include \
                    and t[0] in clinical_terms and e["semtypes"] not in not_to_include_semtypes and \
                        t[1]<375 and t[1]>1:
                    terms.append(t)
                    start_indices.append(t[1])
    idx = np.argsort(start_indices)
    terms = np.array(terms)[idx] #in order of occurrence
    return list(terms)
    

def filter_text_with_clinical_terms(dct):
    out = []
    for d in dct:
        txt = d["text"]
        terms = get_terms(d["entities"])
        if len(terms)>0:
            out.append({'text': txt, 'terms': terms})
    return out


def read_sentences(sentences, files_list, file_no,  loop_through_files=False):
    while len(sentences)<SENTENCES_LIM and file_no<len(files_list):
        fname = files_list[file_no]
        if os.path.exists(fname):
            dct = pkl.load(open(fname, 'rb'))
            sentences = sentences + filter_text_with_clinical_terms(dct)
        file_no+=1
        if loop_through_files:
            file_no = file_no%len(files_list) #rotate through mrn_number
    return sentences, file_no


def belongs_to_term( pos, term_pos):
    st = max(0, int(term_pos[1])-1) # there seems to be indexing differnce in umls parser and tokenizer
    ed = int(term_pos[2])
    if pos[0]>=st and pos[1]<=ed: #start and end indices
        return True
    else:
        return False

def get_labels(inputs, terms, mask_token_id, mask=False):

    labels = torch.clone(inputs["input_ids"]) #same as input text ids 

    if mask:
        a = torch.randint(0, len(terms), (int(0.5*len(terms)), )) #
        a = [True if i in a else False for i in range(len(terms))]
        #pick atleast 1 term
        if any(a):
            terms = np.array(terms)[a]
        else:
            terms = [terms[np.random.choice([i for i in range(len(terms))])]]
    else:
        terms = [] # no terms to be masked
    for idx2 in range(inputs["input_ids"].shape[1]):
        pos = inputs["offset_mapping"][0,idx2]
        pos = pos.numpy()
        if pos[1]!=0: #not padded token 
            if any([belongs_to_term(pos, t_pos) for t_pos in terms]):
                inputs["input_ids"][0, idx2] = mask_token_id
                inputs["attention_mask"][0, idx2] = 0
            else:
                labels[0, idx2]=-100 
    return inputs, labels

class CustomDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer, files_list, FILE_NO, max_length, loop_through_files, max_steps_allowed):
        super(CustomDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.files_list = files_list
        self.loop_through_files = loop_through_files
        self.max_steps_allowed=max_steps_allowed
        sentences_out = read_sentences([], self.files_list, FILE_NO, loop_through_files=self.loop_through_files)
        self.sentences = sentences_out[0]
        self.FILE_NO = sentences_out[1]
        self.MARGIN = 0
        

    def __len__(self):
        if self.max_steps_allowed is not None:
            return int(self.max_steps_allowed)
        else:
            return len(self.sentences)

    def __getitem__(self, i):
        i = i%len(self.sentences)

        mask_sen_id = torch.randint(0, 11, (1,) ) #pick a number between 0 and 10, mask clinical term from that sentence only 
        for id in range(11):
            iii = (i+id)%len(self.sentences)
            inputs1= self.tokenizer(self.sentences[iii]["text"], return_tensors="pt", return_offsets_mapping=True)
            inputs1, labels1 = get_labels(inputs1, self.sentences[iii]["terms"], self.tokenizer.mask_token_id, mask=id==mask_sen_id) #mask=True for one sentence only
            if id==0:
                input_ids = torch.clone(inputs1["input_ids"])
                attention_mask = torch.clone(inputs1["attention_mask"])
                labels = torch.clone(labels1)
            else:
                input_ids = torch.cat((input_ids, inputs1["input_ids"]), 1)
                attention_mask = torch.cat((attention_mask, inputs1["attention_mask"]), 1)
                labels = torch.cat((labels, labels1), 1)

        diff = self.max_length - input_ids.shape[1]
        if diff>0:
            zeros = torch.zeros(1, diff, dtype=torch.int64)
            hds = -100*torch.ones(1, diff, dtype=torch.int64)

            input_ids = torch.cat((input_ids, zeros), 1)#.type(torch.int64)
            attention_mask = torch.cat((attention_mask, zeros), 1)
            labels = torch.cat((labels, hds), 1)

        else:

            input_ids = input_ids[:,:self.max_length]
            attention_mask = attention_mask[:,:self.max_length]
            labels = labels[:,:self.max_length]

        inputs = {"input_ids":input_ids, "attention_mask": attention_mask}

        if self.max_steps_allowed is not None:
            self.MARGIN+=1
        if self.MARGIN >= SENTENCES_LIM and self.max_steps_allowed is not None:
            print("reading more", end='\t')
            sys.stdout.flush()
            sentences_out = read_sentences([], self.files_list, self.FILE_NO,  loop_through_files=self.loop_through_files) 
            self.sentences = sentences_out[0]
            self.FILE_NO = sentences_out[1]
            self.MARGIN=0
            sys.stdout.flush()

        return dict(input_ids=inputs["input_ids"], labels=labels, attention_mask=inputs["attention_mask"])


class DataCollatorForCustomDataset(object):
    """Collate examples for custom dataset"""
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, examples_in):#examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        examples = [e["input_ids"] for e in examples_in]
        #we know everything is set for tokenize routput, padding and all
        inputs = examples[0].new_full([len(examples), examples[0].shape[1]], tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            inputs[i, :] = example


        examples = [e["labels"] for e in examples_in]
        labels = examples[0].new_full([len(examples), examples[0].shape[1]], tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            labels[i, :] = example

        examples = [e["attention_mask"] for e in examples_in]
        attention_mask = examples[0].new_full([len(examples), examples[0].shape[1]], tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            attention_mask[i, :] = example


        return {"input_ids": inputs, "labels": labels, "attention_mask": attention_mask}