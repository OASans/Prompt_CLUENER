import os
import sys
import json
import random
from matplotlib.pyplot import axis
import pandas as pd
import collections

random.seed(5)


def test_process(in_path, out_path):
    """
    produce a list of dicts
    example:
    [{"words": ["SOCCER", "-", "JAPAN", "GET", "LUCKY", "WIN", ",", "CHINA", "IN", "SURPRISE", "DEFEAT", "."], 
    "labels": ["O", "O", "B-LOC", "O", "O", "O", "O", "B-PER", "O", "O", "O", "O"]}]
    """
    examples = []
    with open(in_path, "r", encoding="utf-8") as f:
        for sample in f:
            words = []
            labels = []
            if sample.startswith("-DOCSTART-") or sample == "" or sample == "\n":
                continue
            else:
                line = json.loads(sample)
                words = list(line['text'])
                labels = ['O'] * len(words)
                # "label": {"address": {"台湾": [[15, 16]]}, "name": {"彭小军": [[0, 2]]}}
                for k, v in line['label'].items():
                    # k = 'address'
                    # v = dict of entity and its positions(may have many entities and one entities may coincide with several positions)
                    for entity, positions in v.items():
                        for p in positions:
                            # p = [0, 2]
                            labels[p[0]] = 'B-' + k
                            for i in range(p[0] + 1, p[1] + 1):
                                labels[i] = 'I-' + k
            if words and labels:
                examples.append({'words': words, 'labels': labels})
    with open(out_path, "w") as f:
        json.dump(examples, f)


def inference_process(in_path, out_path):
    examples = []
    with open(in_path, "r", encoding="utf-8") as f:
        for sample in f:
            if sample.startswith("-DOCSTART-") or sample == "" or sample == "\n":
                continue
            line = json.loads(sample)
            id = line['id']
            words = list(line['text'])
            if words:
                examples.append({'id': id, 'words': words})
    with open(out_path, "w") as f:
        json.dump(examples, f)


def train_dev_process(path, processed_path, template_dict):
    processed_data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            source = d['text']
            for k, v in d['label'].items():
                for value, position in v.items():
                    target = value + template_dict[k]
                    processed_data.append({'source': source, 'target': target, 'entity': value, 'type': k})

    processed_data = pd.DataFrame(processed_data)
    processed_data.columns = ['Source sentence', 'Answer sentence', 'Entity', 'Type']
    processed_data.to_csv(processed_path)


def add_non_entity(raw_path, processed_path, template_dict, gram_window=(2, 10), negative_ratio=1.5):
    raw_data = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            raw_data.append(line)
    df = pd.read_csv(processed_path, index_col=0)

    idx2spans = collections.defaultdict(list)
    for i in range(len(raw_data)):
        sample = json.loads(raw_data[i])
        entity_spans = []
        for k, v in sample['label'].items():
            for value, positions in v.items():
                entity_spans.extend(positions)
        entity_spans = list(entity_spans)
        idx2spans[i] = entity_spans
    
    sentence_num = len(raw_data)
    entity_num = df.shape[0]
    non_num = negative_ratio * entity_num

    non_set = []
    while len(non_set) < non_num:
        print(len(non_set), non_num)
        sample_idx = random.randint(0, sentence_num - 1)
        sample = json.loads(raw_data[sample_idx])
        text = sample['text']
        text_len = len(text)
        if text_len < gram_window[0]:
            continue
        entity_spans = idx2spans[sample_idx]

        sample_window = random.randint(gram_window[0], gram_window[1])
        try_times = 0
        while sample_window >= text_len:
            sample_window = random.randint(gram_window[0], gram_window[1])
            try_times += 1
            if try_times >= 5:
                break
        if try_times >= 5:
            continue

        sample_start = random.randint(0, text_len - sample_window)
        non_entity_span = [sample_start, sample_start + sample_window - 1]
        try_times = 0
        while non_entity_span in entity_spans:
            sample_start = random.randint(0, text_len - sample_window - 1)
            non_entity_span = [sample_start, sample_start + sample_window - 1]
            try_times += 1
            if try_times >= 5:
                break
        if try_times >= 5:
            continue

        non_entity = text[non_entity_span[0]:non_entity_span[1]+1]
        idx2spans[sample_idx].extend(non_entity_span)

        target = non_entity + template_dict['non']
        non_set.append({'Source sentence': text, 'Answer sentence': target, 'Entity': non_entity, 'Type': 'non'})
    
    non_df = pd.DataFrame(list(non_set))
    df = pd.concat([df, non_df], axis=0)
    df = df.sort_values(by='Source sentence').reset_index(drop=True)
    df = df.drop(columns=['Entity', 'Type'])
    df.to_csv(processed_path)


if __name__ == '__main__':
    raw_dir = 'cluener_public'
    train_path = raw_dir + '/train.json'
    dev_path = raw_dir + '/dev.json'
    test_path = raw_dir + '/dev.json'
    inference_path = raw_dir + '/test.json'

    processed_dir = 'processed_data'
    processed_train_path = processed_dir + '/train.csv'
    processed_dev_path = processed_dir + '/dev.csv'
    processed_test_path = processed_dir + '/test.json'
    processed_inference_path = processed_dir + '/inference.json'

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    entity_list = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
    template_list = ["是一个地址实体。", "是一个书名实体。", "是一个公司实体。","是一个游戏实体。", 
                    "是一个政府实体。", "是一个电影实体。", "是一个人名实体。", "是一个组织机构实体。", 
                    "是一个职位实体。", "是一个景点实体。"]
    template_dict = {}
    for i in range(len(entity_list)):
        template_dict[entity_list[i]] = template_list[i]
    template_dict['non'] = '不是一个实体。'

    # train_dev_process(train_path, processed_train_path, template_dict)
    # train_dev_process(dev_path, processed_dev_path, template_dict)

    # add_non_entity(train_path, processed_train_path, template_dict, gram_window=(2, 10), negative_ratio=1.5)
    # add_non_entity(dev_path, processed_dev_path, template_dict, gram_window=(2, 10), negative_ratio=1.5)

    # test_process(test_path, processed_test_path)
    inference_process(inference_path, processed_inference_path)