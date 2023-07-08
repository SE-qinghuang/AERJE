#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import csv
import pandas as pd
import numpy as np
import json, time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass
import torch
import logging
import random
import math
from typing import Optional, Union
from collections import OrderedDict
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from transformers.file_utils import PaddingStrategy

from AERJE.extraction.record_schema import RecordSchema
from AERJE.extraction.dataset_processer import spot_prompt, asoc_prompt
from AERJE.extraction.constants import BaseStructureMarker, text_start
from AERJE.extraction.utils import convert_to_record_function
from AERJE.extraction.noiser.spot_asoc_noiser import SpotAsocNoiser


logger = logging.getLogger("__main__")


class DynamicSSIGenerator():
    """
    Sample negative spot and asoc to construct SSI
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, schema: RecordSchema, positive_rate=1, negative=5, ordered_prompt=False) -> None:
        self.spot_dict = self.get_ordered_dict(schema.type_list, tokenizer)
        self.asoc_dict = self.get_ordered_dict(schema.role_list, tokenizer)
        self.spot_list = list(self.spot_dict.keys())
        self.asoc_list = list(self.asoc_dict.keys())
        self.spot_prompt = tokenizer.get_vocab()[spot_prompt]
        self.asoc_prompt = tokenizer.get_vocab()[asoc_prompt]
        self.text_start = tokenizer.get_vocab()[text_start]
        self.positive_rate = positive_rate if positive_rate > 0 and positive_rate < 1 else 1
        self.negative = negative
        self.ordered_prompt = ordered_prompt
        logger.info(f"Meta Sample, Negative: {self.negative}, Ordered Prompt: {self.ordered_prompt}")

    @staticmethod
    def get_ordered_dict(schema_name_list, tokenizer):
        schema_ordered_dict = OrderedDict()
        for name in schema_name_list:
            schema_ordered_dict[name] = tokenizer.encode(name, add_special_tokens=False)
        return schema_ordered_dict

    @staticmethod
    def sample_negative(postive, candidates, k=5):
        if k < 0:
            k = len(candidates)
        negative_set = set()
        for index in torch.randperm(len(candidates))[:k].tolist():
            negative = candidates[index]
            if negative not in postive:
                negative_set.add(negative)
        return list(negative_set)

    def sample_spot(self, positive):
        """ Sample spot
        """
        negative_spot = self.sample_negative(postive=positive, candidates=self.spot_list, k=self.negative)
        ### [] ###
        positive_spot = random.sample(positive, math.floor(len(positive) * self.positive_rate))
        ### ['opinion','aspect'] ###
        prefix_spot_candidates = positive_spot + negative_spot
        ### ['opinion','aspect'] ###
        converted_spot_prefix = self.convert_prefix(
            candidates=prefix_spot_candidates,
            prompt=self.spot_prompt,
            ### <spot> ###
            mapper=self.spot_dict,
            ### (aspect:2663),(opinion:3474) ###
            ordered_prompt=self.ordered_prompt,
        )

        return converted_spot_prefix, positive_spot, negative_spot

    def sample_asoc(self, positive):
        """ Sample Asoc
        """
        negative_asoc = self.sample_negative(postive=positive, candidates=self.asoc_list, k=self.negative)
        prefix_asoc_candidates = positive + negative_asoc
        converted_asoc_prefix = self.convert_prefix(
            candidates=prefix_asoc_candidates,
            prompt=self.asoc_prompt,
            mapper=self.asoc_dict,
            ordered_prompt=self.ordered_prompt,
        )
        return converted_asoc_prefix, negative_asoc

    def full_spot(self, shuffle=False):
        # Random Prompt + Shuffle
        if not self.ordered_prompt and shuffle:
            ordered_prompt = False
        else:
            ordered_prompt = True
        return self.convert_prefix(
            candidates=self.spot_list,
            prompt=self.spot_prompt,
            mapper=self.spot_dict,
            ordered_prompt=ordered_prompt,
        )

    def full_asoc(self, shuffle=False):
        # Random Prompt + Shuffle
        if not self.ordered_prompt and shuffle:
            ordered_prompt = False
        else:
            ordered_prompt = True
        return self.convert_prefix(
            candidates=self.asoc_list,
            prompt=self.asoc_prompt,
            mapper=self.asoc_dict,
            ordered_prompt=ordered_prompt,
        )

    @staticmethod
    def convert_prefix(candidates, prompt, mapper, ordered_prompt=True):
        prefix = list()
        if ordered_prompt:
            candidate_sorted = sorted([(candidate, index) for index, candidate in enumerate(candidates)])
            index_list = [index for _, index in candidate_sorted]
        else:
            index_list = torch.randperm(len(candidates)).tolist()

        for index in index_list:
            prefix += [prompt]
            prefix += mapper[candidates[index]]
        return prefix
    ### [32100(<spot>),3474(<aspect>),32100(<spot>),2663(<opinion>)] ###
    ### [32101(<asoc>),7163(postive),32101(<asoc>),1465(negative),32101(<asoc>),2841(<neutral>)]

bert_path = "./base_models/AERJE-generator-en/"  # 该文件夹下存放三个文件（'vocab.txt', 'pytorch_model.bin', 'config.json'）
tokenizer_bert = BertTokenizer.from_pretrained(bert_path)  # 初始化分词器
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Bert_Model(nn.Module):
    def __init__(self, bert_path, classes=5):
        super(Bert_Model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)  # 导入模型超参数
        self.bert = BertModel.from_pretrained(bert_path)  # 加载预训练模型权重
        self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]  # 池化后的输出 [bs, config.hidden_size]
        logit = self.fc(out_pool)  # [bs, classes]
        return logit

model = Bert_Model(bert_path).to(DEVICE)
model.load_state_dict(torch.load("./classifier_data/saved_bert_model/best_bert_model.pth"))
@dataclass
class DataCollatorForMetaSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        max_target_length (:obj:`int`, `optional`):
            Maximum length of target sequence length.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    negative_sampler: DynamicSSIGenerator
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    max_target_length: Optional[int] = None
    max_prefix_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    spot_asoc_nosier: SpotAsocNoiser = None
    decoding_format: str = 'spotasoc'

    def __call__(self, features):
        """ Make Meta Schema Batch

        Args:
            features (Dict): [description]
                - sample_prompt: indicates sample_prompt example, need pop after call
                - spots (List[str]): List of spots in this sentence, need pop after call
                - asocs (List[str]): List of asocs in this sentence, need pop after call
                - input_ids
                - attention_mask
                - labels

        Returns:
        """


        for feature in features:

            sample_prompt = feature['sample_prompt']

            #################################################################################
            encode_dict = tokenizer_bert.encode_plus(text=feature['text'], max_length=100,
                                                     padding='max_length', truncation=True)
            input_ids = encode_dict['input_ids']
            input_types = encode_dict['token_type_ids']
            input_masks = encode_dict['attention_mask']

            input_ids = torch.LongTensor(np.array(input_ids))
            input_ids = input_ids.unsqueeze(0)
            input_types = torch.LongTensor(np.array(input_types))
            input_types = input_types.unsqueeze(0)
            input_masks = torch.LongTensor(np.array(input_masks))
            input_masks = input_masks.unsqueeze(0)
            model.eval()
            # val_pred = []
            relation1 = ['function similarity', 'behavior difference', 'function replace',
                         'function collaboration',
                         'type conversion', 'logic constraint', 'efficiency comparison']
            with torch.no_grad():
                y_pred = model(input_ids.to(DEVICE), input_masks.to(DEVICE), input_types.to(DEVICE))
                y_pred = torch.topk(y_pred,3,dim=1)
                y_pred = y_pred[1][0].detach().cpu().numpy().tolist()
            rel = ""
            for i in y_pred:
                rel = rel + " <asoc> " + (relation1[i])

            dynamic_prefix_list = self.tokenizer.encode(rel)
            if 1 in dynamic_prefix_list:
                dynamic_prefix_list.remove(1)
                dynamic_prefix = dynamic_prefix_list
            else:
                dynamic_prefix = dynamic_prefix_list

            ######################################################################################
            if not sample_prompt:
                # Evaluation using Ordered SSI
                converted_spot_prefix = self.negative_sampler.full_spot(shuffle=self.model.training)
                converted_asoc_prefix = self.negative_sampler.full_asoc(shuffle=self.model.training)
            else:
                # Sample SSI    create prefix from schema_file
                converted_spot_prefix, positive_spot, negative_spot = self.negative_sampler.sample_spot(positive=feature.get('spots', []))
                converted_asoc_prefix, negative_asoc = self.negative_sampler.sample_asoc(positive=feature.get('asocs', []))
                ### [32101, 2841, 32101, 7163, 32101, 1465] ### [<asoc>postive<asoc>....]
                # Dynamic generating spot-asoc during training
                if 'spot_asoc' in feature:

                    # Deleted positive example Spot in Target that was not sampled by Prefix
                    feature['spot_asoc'] = [spot_asoc for spot_asoc in feature['spot_asoc'] if spot_asoc["label"] in positive_spot]

                    ### [{'span': 'easy', 'label': 'opinion', 'asoc': []},{'span': 'navigate', 'label': 'aspect', 'asoc': [['positive', 'easy']]}] ###

                    # Inject rejection noise
                    if self.spot_asoc_nosier is not None:
                        if isinstance(self.spot_asoc_nosier, SpotAsocNoiser):
                            feature['spot_asoc'] = self.spot_asoc_nosier.add_noise(
                                feature['spot_asoc'],
                                spot_label_list=negative_spot,
                                asoc_label_list=negative_asoc,
                            )
                        else:
                            raise NotImplementedError(f'{self.spot_asoc_nosier} is not implemented.')

                    # Generate new record, because the process of rejection causes new record.
                    record = convert_to_record_function[self.decoding_format](
                        feature['spot_asoc'],
                        structure_maker=BaseStructureMarker()
                    )
                    feature["labels"] = self.tokenizer.encode(record)

            feature.pop('sample_prompt') if 'sample_prompt' in feature else None
            feature.pop('spot_asoc') if 'spot_asoc' in feature else None
            feature.pop('spots') if 'spots' in feature else None
            feature.pop('asocs') if 'asocs' in feature else None
            feature.pop('text') if 'text' in feature else None
            prefix = converted_spot_prefix + dynamic_prefix  # truncate `prefix` to max length
            if self.max_prefix_length is not None and self.max_prefix_length >= 0:
                prefix = prefix[:self.max_prefix_length]

            feature['input_ids'] = prefix + [self.negative_sampler.text_start] + feature['input_ids'] ###prefix + starttext + inputid

            # truncate `input_ids` to max length
            if self.max_length:
                feature['input_ids'] = feature['input_ids'][:self.max_length]
            if self.max_target_length and 'labels' in feature:
                feature['labels'] = feature['labels'][:self.max_target_length]

            feature['attention_mask'] = [1] * len(feature['input_ids'])

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(_label) for _label in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        ### decoder_input: work with text_input to infer the record,
        return features
