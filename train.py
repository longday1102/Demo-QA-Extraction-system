import transformers
from transformers import XLMRobertaTokenizerFast
from transformers import XLMRobertaForQuestionAnswering
from transformers import XLMRobertaConfig, XLMRobertaModel
import numpy as np
import evaluate
import collections
from tqdm.auto import tqdm
from processing import Create_datasetDict, read_data
from features import Features

# xlm-roberta-base
model_checkpoint = 'bhavikardeshna/xlm-roberta-base-vietnamese'
prepare_model = XLMRobertaModel.from_pretrained(model_checkpoint)
tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_checkpoint)
config = XLMRobertaConfig()
config.vocab_size = tokenizer.vocab_size

test_size  = 0.06
max_length = 256
stride = 64
batch_size = 32
learning_rate = 2e-5
epochs = 3
max_norm = 1.0
model_name_fine_tuned = 'fine-tune-xlm-roberta-base-vietnamese-for-viquad'
VERBOSE = True

dataset_dir = 'Dataset/UIT-ViQuAD.json'

viquad = read_data(dataset_dir)
vn_qadts = Create_datasetDict()(test_size = test_size)

if VERBOSE:
    print('### PARAMETERS ###')
    print()
    print(f'DATASET PATH: {dataset_dir}')
    print(f'DATASETDICT: {vn_qadts}')
    print()
    print(f'MODEL USED: {model_checkpoint}')
    print('MODEL ARCHITECTURE: ')
    print(config)
    print(f'FINE TUNED NAME: {model_name_fine_tuned}')
    print()
    print(f'TEST SIZE: {test_size}')
    print(f'MAX LENGTH: {max_length}')
    print(f'STRIDE: {stride}')
    print(f'BATCH SIZE: {batch_size}')
    print(f'LEARNING RATE: {learning_rate}')
    print(f'EPOCHS: {epochs}')
    print(f'MAX NORM: ', {max_norm})
    print()
    print(60 * '=')
    print()

train_fn = Features.train_processing
valid_fn = Features.valid_processing
train_dataset = vn_qadts['train'].map(train_fn,
                                      batched = True,
                                      remove_columns = vn_qadts['train'].column_names) 
valid_dataset = vn_qadts['validation'].map(valid_fn,
                                           batched = True,
                                           remove_columns = vn_qadts['validation'].column_names)

QA_model = XLMRobertaForQuestionAnswering(prepare_model.config).from_pretrained(model_checkpoint)

# Fine-Tune with TrainingArguments and Trainer
args = transformers.TrainingArguments(model_name_fine_tuned,
                                      evaluation_strategy = 'no',
                                      do_train = True,
                                      per_device_train_batch_size = batch_size,
                                      per_device_eval_batch_size = batch_size,
                                      learning_rate = learning_rate,
                                      weight_decay = 0.01,
                                      adam_beta1 = 0.9,
                                      adam_beta2 = 0.999,
                                      adam_epsilon = 1e-8,
                                      max_grad_norm = max_norm,
                                      fp16 = True,
                                      num_train_epochs = epochs,
                                      logging_strategy = 'epoch')

trainer = transformers.Trainer(model = QA_model,
                               args = args,
                               train_dataset = train_dataset,
                               eval_dataset = valid_dataset,
                               tokenizer = tokenizer)

# train
trainer.train()
