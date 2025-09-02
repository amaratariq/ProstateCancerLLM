import numpy as np
import os
import sys
import transformers
import torch
from torch.utils.data import Dataset

SENTENCES_LIM = 1e5
MAX_NO_SNETS_ABSTRACT = 7
MAX_NO_SNETS_NOTES = 11

ST_SENT_IDX_ABS = int(np.floor(MAX_NO_SNETS_ABSTRACT/2))
ED_SENT_IDX_ABS = int(np.ceil(MAX_NO_SNETS_ABSTRACT/2))

ST_SENT_IDX_N = int(np.floor(MAX_NO_SNETS_NOTES/2))
ED_SENT_IDX_N = int(np.ceil(MAX_NO_SNETS_NOTES/2))


def read_sentences(sentences, files_list, file_no, loop_through_files=False):
    '''
    sentences: set of text snippets
    files_list: list of all data files
    file_no: index from files_list where to start loading from
    loop_through_files: if True, the loader will keep looping through data, otherwise will stop after one loop
    '''
    while len(sentences)<SENTENCES_LIM and file_no<len(files_list):
        fname = files_list[file_no]
        abstract = "abstract/" in fname #is the file abstrat or note
        if os.path.exists(fname):
            f = open(fname)
            sentences_new = list(f.readlines())
            if abstract and len(sentences_new)>MAX_NO_SNETS_ABSTRACT:
                sentences_new = [' '.join(sentences_new[i-ST_SENT_IDX_ABS:i+ED_SENT_IDX_ABS]) for i in range(ST_SENT_IDX_ABS, len(sentences_new)-ED_SENT_IDX_ABS)] 
            elif abstract==False and len(sentences_new)>MAX_NO_SNETS_NOTES:
                sentences_new = [' '.join(sentences_new[i-ST_SENT_IDX_N:i+ED_SENT_IDX_N]) for i in range(ST_SENT_IDX_N, len(sentences_new)-ED_SENT_IDX_N)] 
            else:
                sentences_new = [" ".join(sentences_new)]
            sentences = sentences+sentences_new
            f.close()
        file_no+=1
        if loop_through_files:
            file_no = file_no%len(files_list) 
    return sentences, file_no


class CustomDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer, files_list, FILE_NO, max_length, loop_through_files, max_steps_allowed):
        super(CustomDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.files = files_list
        self.loop_through_mrns = loop_through_files
        self.max_steps_allowed=max_steps_allowed
        sentences_out = read_sentences([], self.files, FILE_NO,  loop_through_files=self.loop_through_files) # read the first batch of snippets
        self.sentences = sentences_out[0]
        self.FILE_NO = sentences_out[1]
        self.MARGIN = 0
        sys.stdout.flush()
        

    def __len__(self):
        if self.max_steps_allowed is not None: #set number of steps for dataloader
            return int(self.max_steps_allowed)
        else:                                   # otherwise go through set of snippets only
            return len(self.sentences)

    def __getitem__(self, i):
        i = i%len(self.sentences)
        inputbatch = self.tokenizer(self.sentences[i], padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
        labels = torch.clone(inputbatch["input_ids"])
        labels[labels==self.tokenizer.pad_token_id]=-100 #excluding padding tokens from loss computation

        if self.max_steps_allowed is not None:
            self.MARGIN+=1
        if self.MARGIN >= SENTENCES_LIM and self.max_steps_allowed is not None:
            # set of snippets used, read more
            print("reading more", end='\t')
            sys.stdout.flush()
            sentences_out = read_sentences([], self.files, self.FILE_NO,  loop_through_files=self.loop_through_files) 
            self.sentences = sentences_out[0]
            self.FILE_NO = sentences_out[1]
            self.MARGIN=0

        return dict(input_ids=inputbatch["input_ids"], labels=labels, attention_mask=inputbatch["attention_mask"])


class DataCollatorForCustomDataset(object):
    """Collate examples for custom dataset"""
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, examples_in):
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
