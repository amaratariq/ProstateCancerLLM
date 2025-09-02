import pandas as pd
import numpy as np
import pickle as pkl
import sys
import os
import transformers
from transformers import  GPT2LMHeadModel
import torch
from transformers import PreTrainedTokenizerFast
from tokenizers import models,  Tokenizer
from transformers import  Trainer, TrainingArguments
from utils.data_loader_phase1 import CustomDataset, DataCollatorForCustomDataset

device_ids = [0, 1, 2, 3]

MAX_LENGTH = 256

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


    save_steps = 5000
    eval_steps = 5000
    max_training_steps = 5e8


    val_dataset = CustomDataset(tokenizer, files_val, FILE_NO_VAL, MAX_LENGTH, loop_through_files=False, max_steps_allowed=None)
    train_dataset = CustomDataset(tokenizer, files_train, FILE_NO_TRAIN, loop_through_files=True, max_steps_allowed=max_training_steps)
    print(val_dataset.__len__(), train_dataset.__len__())
    print('datasets created')
    sys.stdout.flush()

    datacollator = DataCollatorForCustomDataset()

    

    training_args = TrainingArguments(
                output_dir = save_dir,  # The output directory
                overwrite_output_dir = True,  # overwrite the content of the output directory
                num_train_epochs = 1,  # number of training epochs 5
                per_device_train_batch_size = 32,  # batch size for training (ORIGINAL WAS 32)
                per_device_eval_batch_size = 32,  # batch size for evaluation (ORIGINAL WAS 64)
                gradient_accumulation_steps=4,
                logging_steps = eval_steps,
                evaluation_strategy="steps",
                eval_steps = eval_steps,  # Number of update steps between two evaluations.
                save_strategy="steps",
                save_steps = save_steps,  # after # steps model is saved
                save_total_limit=20,
                warmup_steps = 100,  # number of warmup steps for learning rate scheduler 500
                prediction_loss_only = False,
                # save_safetensors=False,
                load_best_model_at_end = True,

                learning_rate = 1e-3
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
deepspeed phase1_trainig.py model_path token_path save_dir train_val_test_files resume_from_checkpoint --deepspeed tests/deepspeed/ds_config_zero3.json >> log/phase1_training.txt &

'''