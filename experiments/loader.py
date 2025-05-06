import os
import random
import torch
from datasets import load_dataset, load_from_disk
from typing import Optional, Literal, Union, List
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import DataCollatorForSeq2Seq


def get_calibration_dataloader(
    dataset_name: Literal["wikitext2", "ptb", "c4", 'boolq', 'openbookqa', 'arc_easy', 'arc_challenge', 'hellaswag', 'winogrande', 'piqa', 'mathqa'],
    tokenizer,
    num_samples: Optional[int] = 128, # wikitext: 512
    seq_len: Optional[float] = 2048,
    padding: Optional[Union[str, bool]] = 'max_length',
    batch_size: Optional[int] = 1, # wikitext2: 4
    seed: Optional[int] = 42,
    mix: Optional[bool] = False
):
    random.seed(seed)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, return_tensors='pt', padding=True
    )
    class TrainDataset(Dataset):
        def __init__(self, input_tensors) -> None:
            self.inputs = input_tensors
            self.targets = input_tensors.clone()
        
        def __len__(self):
            return len(self.inputs)
        
        def __getitem__(self, index):
            result = {}
            result["input_ids"] = self.inputs[index, :-1]
            result["labels"] = self.targets[index, 1:]
            return result

    def tokenize(prompt, add_eos_token: bool =True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=seq_len,
            padding=padding,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < seq_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()[1:]
        result["input_ids"] = result["input_ids"][:-1]
        result["attention_mask"] = result["attention_mask"][:-1]
        return result

    def process_pretrain_data(train_data, tokenizer, seq_len, field_name):
        train_ids = tokenizer("\n\n".join(train_data[field_name]), return_tensors='pt').input_ids[0]
        train_ids_batch = []
        nsamples = train_ids.numel() // seq_len

        for i in range(nsamples):
            batch = train_ids[(i * seq_len):((i + 1) * seq_len)]
            train_ids_batch.append(batch)
        train_ids_batch = torch.stack(train_ids_batch)
        return TrainDataset(input_tensors=train_ids_batch)
    
    def process_task_data(train_data):
        data_point = train_data["text"]
        tokenized_prompt = tokenize(data_point)
        return tokenized_prompt

    if 'wikitext2' in dataset_name:
        # train_data = load_dataset(
        #     'wikitext',
        #     'wikitext-2-raw-v1',
        #     split='train'
        # )
        train_data = load_from_disk("datasets/wikitext/train")
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_data = train_data.select(random_indices)
        train_dataset = process_pretrain_data(train_data, tokenizer, seq_len, 'text')
        data_collator = None

    elif 'c4' in dataset_name:
        # train_data = load_dataset(
        #     "allenai/c4",
        #     data_files={"train": "en/c4-validation.00000-of-00008.json.gz"},
        #     split="train",
        #     trust_remote_code=True
        # )
        train_data = load_from_disk("datasets/c4/train")
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_data = train_data.select(random_indices)
        train_dataset = process_pretrain_data(train_data, tokenizer, seq_len, 'text')
        data_collator = None

    else:
        raise NotImplementedError

    print("=======> Done Loading Data!")
    if mix:
        return train_dataset
    else:
        return DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=True)
    

def get_mix_calibration_dataloader(
    dataset_names = ["wikitext2", 'openbookqa', 'arc_easy', 'arc_challenge', 'hellaswag', 'winogrande', 'piqa', 'mathqa'],
    tokenizer = None,
    num_samples: Optional[int] = 128, # wikitext: 512
    dataset_proportion: Optional[List[int]] = None,
    seq_len: Optional[float] = 2048,
    padding: Optional[Union[str, bool]] = 'max_length',
    batch_size: Optional[int] = 1, # wikitext2: 4
    seed: Optional[int] = 42  
    ):
    random.seed(seed)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, return_tensors='pt', padding=True
    )
    if not tokenizer:
        raise ValueError("Tokenizer should be given")

    if not dataset_proportion:
        union_proportion = 1 / len(dataset_names)
        dataset_proportion = [union_proportion for _ in range(len(dataset_names))]

    train_dataset_list = []
    for idx, dataset_name in enumerate(dataset_names):
        current_num_samples = num_samples * dataset_proportion[idx]
        current_padding = padding if "wikitext2" in dataset_name else False

        train_dataset = get_calibration_dataloader(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            num_samples=current_num_samples,
            seq_len=seq_len, 
            padding=current_padding,
            batch_size=batch_size, 
            seed=seed, mix=True
        )
        train_dataset_list.append(train_dataset)

    train_datasets = ConcatDataset(train_dataset_list)
    return DataLoader(train_datasets, batch_size=batch_size, collate_fn=data_collator, shuffle=True)


def get_evaluation_dataloader(dataset_name: Literal["wikitext2", "ptb", "c4"], tokenizer):
    if "wikitext2" in dataset_name:
        # testdata = load_dataset(
        #     "wikitext",
        #     "wikitext-2-raw-v1",
        #     split="test",
        # )
        testdata = load_from_disk("datasets/wikitext/test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    if "ptb" in dataset_name:
        # valdata = load_dataset(
        #     "ptb_text_only",
        #     "penn_treebank",
        #     split="validation",
        #     trust_remote_code=True
        # )
        valdata = load_from_disk("datasets/ptb/validation")
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
        return testenc
    if "c4" in dataset_name:
        # testdata = load_dataset(
        #     "allenai/c4",
        #     data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        #     split="validation",
        #     trust_remote_code=True
        # )
        testdata = load_from_disk("datasets/c4/validation")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    raise NotImplementedError

def get_test_data(name, tokenizer, seq_len=2048, batch_size = 4):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)
    ####
    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)
    ####
    if 'wikitext2' in name:
        # test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        test_data = load_from_disk("datasets/wikitext/test")
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'ptb' in name:
        # test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        test_data = load_from_disk("datasets/ptb/test")
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    elif 'c4' in name:
        # test_data = load_dataset(
        #     "allenai/c4",
        #     data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        #     split="train",
        #     trust_remote_code=True
        # )
        test_data = load_from_disk("datasets/c4/validation")
        test_dataset = process_data(test_data[0:2000], tokenizer, seq_len, 'text')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader