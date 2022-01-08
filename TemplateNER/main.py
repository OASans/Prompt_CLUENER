import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
sys.path.append(os.getcwd() + '/myPrompt/TemplateNER')
import math
import json
import torch
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from simpletransformers.config.model_args import Seq2SeqArgs
from transformers import BartTokenizer

from data_process import BartPromptDataModule
from models import BartModel
from ner_metrics import *

pl.seed_everything(4)


class Config:
    def __init__(self):
        self.debug = False
        self.preprocess = False
        self.accelerator = 'gpu'
        self.en_train = False
        self.en_inference = True
        self.test_model_path = 'models/bartprompt-epoch=03-val_loss=0.35.ckpt'
        # hardware
        self.num_processes = 48 if self.accelerator == 'gpu' else 12
        self.gpus = 4 if self.accelerator == 'gpu' else 1
        self.strategy = 'dp' if (self.accelerator == 'gpu' and self.gpus > 1) else None
        self.precision = 16 if self.accelerator == 'gpu' else 32

        # utils
        self.log_dir = 'logs/'
        self.model_dir = 'models/'

        # model
        self.batch_size = 16
        self.total_steps = 0
        self.plm_name = "facebook/bart-large"
        self.model_args = {
            "reprocess_input_data": True,
            "max_seq_length": 50,
            "train_batch_size": self.batch_size * self.gpus,
            "num_train_epochs": 20,
            "save_eval_checkpoints": False,
            "save_model_every_epoch": False,
            "evaluate_during_training": True,
            "evaluate_generated_text": True,
            "evaluate_during_training_verbose": True,
            "use_multiprocessing": False,
            "max_length": 25,
            "manual_seed": 4,
            "save_steps": 11898,
            "gradient_accumulation_steps": 1,
        }

        # fit
        # self.auto_scale_batch_size = "binsearch"

        # inference
        self.ngram = 9
        self.entity_dict = {0: 'LOC', 1: 'PER', 2: 'ORG', 3: 'MISC', 4: 'O'}
        self.template_list = [" is a location entity .", 
                                " is a person entity .", 
                                " is an organization entity .",
                                " is an other entity .", 
                                " is not a named entity ."]


def _load_model_args(args):
    loaded_args = Seq2SeqArgs()
    loaded_args.update_from_dict(args)
    return loaded_args

def get_inference_inputs(words, source, tokenizer, template_list):
    words_length = len(words)
    source = [source]*(5*words_length)

    input_ids = tokenizer(source, return_tensors='pt')['input_ids']

    temp_list = []
    for i in range(words_length):
        for j in range(len(template_list)):
            temp_list.append(words[i]+template_list[j])

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    output_ids[:, 0] = 2
    output_length_list = [0]*5*words_length

    for i in range(len(temp_list)//5):  # 同一个gram的不同template
        base_length = ((tokenizer(temp_list[i * 5], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[1] - 4
        output_length_list[i*5:i*5+ 5] = [base_length]*5
        output_length_list[i*5+4] += 1
    return input_ids, output_ids, output_length_list

def get_ngrams(start, input_TXT_list, window=9):
    words = []
    for j in range(1, window):
        word = (' ').join(input_TXT_list[start:start+j])
        words.append(word)
    return words

def cal_scores(output, words_length):
    score = [1] * 5 * words_length
    for i in range(output_ids.shape[1] - 3):
        # print(input_ids.shape)
        logits = output[:, i, :]
        logits = logits.softmax(dim=1)
        # values, predictions = logits.topk(1,dim = 1)
        logits = logits.detach().numpy() if logits.device == 'cpu' else logits.cpu().detach().numpy()
        # print(output_ids[:, i+1].item())
        for j in range(0, 5*words_length):
            if i < output_length_list[j]:
                score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]
    return score

def refine_entity_list(entity_list):
    i = 0
    if len(entity_list) > 1:
        while i < len(entity_list):
            j = i+1
            while j < len(entity_list):
                if (entity_list[i][1] < entity_list[j][0]) or (entity_list[i][0] > entity_list[j][1]):
                    j += 1
                else:
                    if entity_list[i][3] < entity_list[j][3]:
                        entity_list[i], entity_list[j] = entity_list[j], entity_list[i]
                        entity_list.pop(j)
                    else:
                        entity_list.pop(j)
            i += 1
    return entity_list

def get_label_list(entity_list, length):
    label_list = ['O'] * length
    for entity in entity_list:
        label_list[entity[0]:entity[1]+1] = ["I-"+entity[2]]*(entity[1]-entity[0]+1)
        label_list[entity[0]] = "B-"+entity[2]
    return label_list


if __name__ == '__main__':
    config = Config()
    config.model_args = _load_model_args(config.model_args)

    logger = TensorBoardLogger(
        save_dir=config.log_dir,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=config.model_dir,
        filename="bartprompt-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = Trainer(
        fast_dev_run=config.debug,
        max_epochs=config.model_args.num_train_epochs,
        accelerator=config.accelerator,
        gpus=config.gpus,
        strategy=config.strategy,
        logger=logger,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", patience=config.model_args.early_stopping_patience)],
        precision=config.precision,
        gradient_clip_val=config.model_args.max_grad_norm
    )

    data_module = BartPromptDataModule(config)
    if config.preprocess:
        data_module.my_prepare_data()
    
    if config.en_train:
        data_module.setup(stage='fit')
        # update args
        tb_size = config.model_args.train_batch_size * max(1, config.gpus)
        ab_size = float(config.model_args.num_train_epochs) // trainer.accumulate_grad_batches
        config.total_steps = (data_module.train_len // tb_size) * ab_size
        config.model_args.warmup_steps = math.ceil(config.total_steps * config.model_args.warmup_ratio)

        model = BartModel(config)
        trainer.fit(model, data_module)

        output_json = {
            "best_model_path": checkpoint_callback.best_model_path,
        }
        print(output_json)

    if config.en_inference:
        print('loading trained model from {}'.format(config.test_model_path))
        tokenizer = BartTokenizer.from_pretrained(config.plm_name)
        model = BartModel.load_from_checkpoint(checkpoint_path=config.test_model_path, config=config)
        if torch.cuda.is_available():
            model.cuda()
        device = model.device

        with open(data_module.processed_test_path, "r") as f:
            test_data = json.load(fp=f)
        
        model.eval()
        with torch.no_grad():
            trues_list = []
            preds_list = []
            str = ' '
            num_01 = len(test_data)
            num_point = 0
            for example in tqdm(test_data):
                source = str.join(example['words'])
                input_TXT_list = example['words']
                entity_list = []
                for i in range(len(input_TXT_list)):
                    words = get_ngrams(i, input_TXT_list, config.ngram)
                    input_ids, output_ids, output_length_list = get_inference_inputs(words, source, tokenizer, config.template_list)
                    input_ids, output_ids = input_ids.to(device), output_ids.to(device)
                    output = model(input_ids=input_ids, decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2])[0]
                    score = cal_scores(output, len(words))
                    end = i + (score.index(max(score))//5)
                    entity = [i, end, config.entity_dict[(score.index(max(score))%5)], max(score)] #[start_index,end_index,label,score]
                    if entity[1] >= len(input_TXT_list):
                        entity[1] = len(input_TXT_list)-1
                    if entity[2] != 'O':
                        entity_list.append(entity)
            
                entity_list = refine_entity_list(entity_list)
                label_list = get_label_list(entity_list, len(input_TXT_list))
                preds_list.append(label_list)
                trues_list.append(example['labels'])
                print('%d/%d'%(num_point+1, num_01))
                print('Pred:', preds_list[num_point])
                print('Gold:', trues_list[num_point])
                num_point += 1
            true_entities = get_entities_bio(trues_list)
            pred_entities = get_entities_bio(preds_list)
            results = {
                "precision": precision_score(true_entities, pred_entities),
                "recall": recall_score(true_entities, pred_entities),
                "f1": f1_score(true_entities, pred_entities)
            }
            print(results)
            print(classification_report(true_entities, pred_entities))
            for num_point in range(len(preds_list)):
                preds_list[num_point] = ' '.join(preds_list[num_point]) + '\n'
                trues_list[num_point] = ' '.join(trues_list[num_point]) + '\n'
            with open('./pred.txt', 'w') as f0:
                f0.writelines(preds_list)
            with open('./gold.txt', 'w') as f0:
                f0.writelines(trues_list)


