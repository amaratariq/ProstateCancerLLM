import pandas as pd
import numpy as np
import torch
import sys
import os
import pickle as pkl

from transformers import PreTrainedTokenizerFast
from transformers import  GPT2LMHeadModel,  PreTrainedTokenizerFast
from transformers import  Trainer, TrainingArguments


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tokenizers import models, Tokenizer

from utils.data_loader_phase2 import CustomDataset, DataCollatorForCustomDataset


sys.stdout.flush()
device_ids = [0, 1, 2, 3]


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



if __name__ == "__main__":
    '''
    arguments
    1: model_path
    2: token_path
    3: save_dir
    4: train_val_test_files - dct of train, test, validation files of text in the data folder
    5: resume_from_checkpoint - assume model path contains the checkpoint
    '''
    print('command line arguments', str(sys.argv))

    model_path = sys.argv[1]
    token_path = sys.srgv[2]
    save_dir = sys.argv[3]
    files_dct = sys.argv[4]
    resume_from_checkpoint = sys.argc[5]=="true"

    tokenizer, model = load_model_and_tokenizer(token_path, model_path)
    if os.path.exists(save_dir)==False:
        print(save_dir, os.path.exists(save_dir))
        os.mkdir(save_dir)

    dct = pkl.load(open(files_dct, "rb"))
    files_train = dct["train"]
    files_test = dct["test"]
    files_val = dct["val"]
    FILE_NO_TRAIN = 0
    FILE_NO_TEST = 0
    FILE_NO_VAL = 0
    max_length = 256

    save_steps = 5000
    eval_steps = 5000
    sentences_lim = 1e4
    max_training_steps = 2e8


    
    val_dataset = CustomDataset(tokenizer, files_val, FILE_NO_VAL,  loop_through_files=False, max_steps_allowed=None)
    train_dataset = CustomDataset(tokenizer, files_train, FILE_NO_TRAIN, loop_through_files=True, max_steps_allowed=max_training_steps)
    print(val_dataset.__len__(), train_dataset.__len__())
    print('datasets created')
    sys.stdout.flush()

    datacollator = DataCollatorForCustomDataset()

    

    training_args = TrainingArguments(
                output_dir = save_dir,  # The output directory
                overwrite_output_dir = True,  # overwrite the content of the output directory
                num_train_epochs = 1,  # number of training epochs 5
                per_device_train_batch_size = 1,  # batch size for training (ORIGINAL WAS 32)
                per_device_eval_batch_size = 1,  # batch size for evaluation (ORIGINAL WAS 64)
                gradient_accumulation_steps=32*4,
                logging_steps = eval_steps,
                evaluation_strategy="steps",
                eval_steps = eval_steps,  # Number of update steps between two evaluations.
                save_strategy="steps",
                save_steps = save_steps,  # after # steps model is saved
                save_total_limit=20,

                learning_rate=1e-5,
                
                warmup_steps = 100,  # number of warmup steps for learning rate scheduler 500
                prediction_loss_only = False,
                load_best_model_at_end = True,
                # save_safetensors=False
            )

    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = datacollator,
        train_dataset = train_dataset,
        eval_dataset = val_dataset
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()

''' 
deepspeed phase2_training.py  --deepspeed --include="localhost:0,2" tests/deepspeed/ds_config_zero3.json >> log/phase2_trainig.txt &

'''