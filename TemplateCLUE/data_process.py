import os
import sys
sys.path.append(os.getcwd() + '/myPrompt/TemplateNER')
import json
import logging
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer
from pytorch_lightning import LightningDataModule

logger = logging.getLogger(__name__)


class BartPromptDataSet(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index]


class BartPromptDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.processed_train_path = 'processed_data/train.csv'
        self.processed_dev_path = 'processed_data/dev.csv'
        self.processed_test_path = 'processed_data/test.json'
        self.processed_inference_path = 'processed_data/inference.json'

        self.tokenized_train_path = 'processed_data/train.pt'
        self.tokenized_dev_path = 'processed_data/dev.pt'

        self.max_seq_length = config.model_args.max_seq_length
        self.tokenizer_name = config.plm_name
        self.tokenizer = None

        self.train_len = 0
        self.batch_size = config.model_args.train_batch_size
        self.num_workers = config.num_processes
    
    def my_prepare_data(self) -> None:
        def _preprocess_data(data):
            input_text, target_text = data
            input_ids = self.tokenizer.batch_encode_plus(
                [input_text], max_length=self.max_seq_length, padding='max_length', truncation=True, return_tensors="pt",
            )

            target_ids = self.tokenizer.batch_encode_plus(
                [target_text], max_length=self.max_seq_length, padding='max_length', truncation=True, return_tensors="pt"
            )

            return {
                "source_ids": input_ids["input_ids"].squeeze(),
                "source_mask": input_ids["attention_mask"].squeeze(),
                "target_ids": target_ids["input_ids"].squeeze(),
            }

        def _save_data(data, path):
            torch.save(data, path)
        
        def _main_process(in_path, out_path):
            self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
            data = pd.read_csv(in_path, sep=',', index_col=0).values
            data = [(d[0], d[1]) for d in data]
            data = [_preprocess_data(d) for d in tqdm(data)]
            _save_data(data, out_path)

        _main_process(self.processed_train_path, self.tokenized_train_path)
        _main_process(self.processed_dev_path, self.tokenized_dev_path)


    def setup(self, stage) -> None:
        if stage in (None, "fit"):
            if not os.path.isfile(self.tokenized_train_path) or not os.path.isfile(self.tokenized_dev_path):
                raise ValueError(
                    "train_data and eval_data not prepared!"
                    )
            train_data = torch.load(self.tokenized_train_path)
            self.train_data = BartPromptDataSet(train_data)
            eval_data = torch.load(self.tokenized_dev_path)
            self.eval_data = BartPromptDataSet(eval_data)
            self.train_len = self.train_data.__len__()
        # if stage in (None, "test"):
        #     with open(self.processed_test_path, "r") as f:
        #         test_data = json.load(fp=f)
        #     self.test_data = InferenceDataSet(test_data)
        #     self.test_batch_size = len(self.test_data)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.eval_data, batch_size = self.batch_size, num_workers=self.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(self.test_data, batch_size = self.test_batch_size, num_workers=self.num_workers)
