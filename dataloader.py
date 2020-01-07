# -*- coding:utf8 -*-

import os
import json
import logging
import numpy as np
from collections import Counter
import jieba
from nltk.translate.bleu_score import sentence_bleu
import io

def word_tokenize(sent):
    if isinstance(sent, list):
        # tokens = "".join(sent)
        # tokens = list(cut_api.cut(sent))
        tokens = sent
        return [token for token in tokens if len(token) >= 1]
    else:
        tokens = jieba.lcut(sent)
        return [token for token in tokens if len(token) >= 1]


class DataLoader(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """

    def __init__(self, max_p_num, max_p_len, max_q_len, max_char_len,
                 train_files=[], dev_files=[], test_files=[]):
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.max_char_len = max_char_len

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.logger.info('---train file-----{}'.format(train_file))
                self.train_set += self._load_dataset(train_file, train=True)
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.logger.info('---dev file-----{}'.format(dev_file))
                self.dev_set += self._load_dataset(dev_file, train=True)
            self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file,train=False)
            self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        max_char_num = 0
        max_char_list = []
        flag=True
        numflag=0
        with io.open(data_path, encoding='utf-8') as fin:
            data_set = []
            for lidx, line in enumerate(fin):
                sample = json.loads(line.strip())
                #print(sample['documents'])
                try:
                    if(train):
                        if sample['segmented_answers']==[] or sample['segmented_question']==[] :
                            continue
                except KeyError:
                    continue
                if train:
                    if len(sample['answer_spans']) == 0:
                        continue
                    if sample['answer_spans'][0][1] >= self.max_p_len:
                        continue

                if 'answer_docs' in sample:
                    sample['answer_passages'] = sample['answer_docs']

                question_tokens = word_tokenize(sample['segmented_question'])
                sample['question_tokens'] = question_tokens
                question_chars = [list(token) for token in question_tokens]
                sample['question_chars'] = question_chars

                for char in question_chars:
                    if len(char) > max_char_num:
                        max_char_num = len(char)
                        max_char_list = char

                sample['passages'] = []
                for d_idx, doc in enumerate(sample['documents']):
                    #print(d_idx)
                    # #print(1111)
                    # # 预处理
                    # try:
                    #     if train:
                    #         doc['segmented_paragraphs'] = self.sort_passage(
                    #             doc['segmented_paragraphs'],
                    #             doc['paragraphs'],
                    #             doc['segmented_title'],
                    #             sample['segmented_answers'],
                    #             sample['segmented_question'], train)
                    #         doc['most_related_para'] = 0
                    #     else:
                    #         doc['segmented_paragraphs'] = self.sort_passage(
                    #             doc['segmented_paragraphs'],
                    #             doc['paragraphs'],
                    #             doc['segmented_title'],
                    #             [],
                    #             sample['segmented_question'],
                    #             train)
                    #    # print(1112)
                    # except:
                    #     pass
                    # #print(2222)
                    # ###
                    if train:
                        para_infos = []
                        for para_tokens in doc['segmented_paragraphs']:
                            para_tokens = word_tokenize(para_tokens)
                            #question_tokens = word_tokenize(sample['segmented_answers'])
                            question_tokens = word_tokenize(sample['segmented_answers'])

                            #print(question_tokens)
                            try:
                                recall_wrt_question = sentence_bleu(para_tokens, question_tokens[0], weights=(0.25, 0.25, 0.25, 0.25))
                            except KeyError:
                                flag=False
                                break
                            flag=True
                            para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))

                        para_infos.sort(key=lambda x: (-x[1], x[2]))
                        fake_passage_tokens = []

                        for para_info in para_infos[:1]:

                            #not [0]
                            fake_passage_tokens += para_info[0]

                        most_related_para = 0
                        passage_tokens = word_tokenize(doc['segmented_paragraphs'][most_related_para])
                        passage_chars = [list(token) for token in passage_tokens]

                        for char in passage_chars:
                            if len(char) > max_char_num:
                                max_char_num = len(char)
                                max_char_list = char


                        sample['passages'].append(
                            {'passage_tokens': fake_passage_tokens,
                             'is_selected': doc['is_selected'],
                             'passage_chars': [list(token) for token in fake_passage_tokens]}
                        )

                        # sample['passages'].append(
                        #     {'passage_tokens': passage_tokens,
                        #      'is_selected': doc['is_selected'],
                        #      'passage_chars': passage_chars}
                        # )
                    else:
                        #print(sample['segmented_question'])
                        para_infos = []
                        for para_tokens in doc['segmented_paragraphs']:
                            para_tokens = word_tokenize(para_tokens)
                            question_tokens = word_tokenize(sample['segmented_question'])
                            try:
                                recall_wrt_question = sentence_bleu(para_tokens, question_tokens[0], weights=(0.25, 0.25, 0.25, 0.25))
                            except KeyError:
                                flag=False
                                break
                            flag=True
                            para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
                        para_infos.sort(key=lambda x: (-x[1], x[2]))
                        fake_passage_tokens = []
                        i=1
                        for para_info in para_infos[:1]:

                            fake_passage_tokens += para_info[0]

                        sample['passages'].append({'passage_tokens': fake_passage_tokens,
                                                   'passage_chars': [list(token) for token in fake_passage_tokens]})
                if flag:
                    data_set.append(sample)
                    #print(sample)
                    numflag=numflag+1
                    #print(numflag)
        return data_set

    def _one_mini_batch(self, data, indices, pad_id, pad_char_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'question_char_ids': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],
                      'passage_char_ids': [],
                      'start_id': [],
                      'end_id': []}
        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)
        for sidx, sample in enumerate(batch_data['raw_data']):
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    batch_data['question_char_ids'].append(sample['question_char_ids'])
                    batch_data['question_length'].append(len(sample['question_token_ids']))
                    passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                    passage_char_ids = sample['passages'][pidx]['passage_char_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
                    batch_data['passage_char_ids'].append(passage_char_ids)
                else:
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['question_char_ids'].append([[]])
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
                    batch_data['passage_char_ids'].append([[]])

        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id, pad_char_id)
        for sample in batch_data['raw_data']:
            if 'answer_passages' in sample and len(sample['answer_passages']):
                gold_passage_offset = padded_p_len * sample['answer_passages'][0]
                batch_data['start_id'].append(gold_passage_offset + sample['answer_spans'][0][0])
                batch_data['end_id'].append(gold_passage_offset + sample['answer_spans'][0][1])
            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
        return batch_data

    def _one_mini_batch2(self, data, indices, pad_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],
                      'start_id': [],
                      'end_id': []}
        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)
        for sidx, sample in enumerate(batch_data['raw_data']):
            #print(sample)
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    batch_data['question_length'].append(len(sample['question_token_ids']))
                    passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
                else:
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding2(batch_data, pad_id)
        for sample in batch_data['raw_data']:
            if 'answer_passages' in sample and len(sample['answer_passages']):
                gold_passage_offset = padded_p_len * sample['answer_passages'][0]
                batch_data['start_id'].append(gold_passage_offset + sample['answer_spans'][0][0])
                batch_data['end_id'].append(gold_passage_offset + sample['answer_spans'][0][1])
            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
        return batch_data

    def _dynamic_padding2(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        return batch_data, pad_p_len, pad_q_len

    def _dynamic_padding(self, batch_data, pad_id, pad_char_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_char_len = self.max_char_len
        pad_p_len = self.max_p_len  # min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = self.max_q_len  # min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        for index, char_list in enumerate(batch_data['passage_char_ids']):
            # print(batch_data['passage_char_ids'])
            for char_index in range(len(char_list)):
                if len(char_list[char_index]) >= pad_char_len:
                    char_list[char_index] = char_list[char_index][:self.max_char_len]
                else:
                    char_list[char_index] += [pad_char_id] * (pad_char_len - len(char_list[char_index]))
            batch_data['passage_char_ids'][index] = char_list
        batch_data['passage_char_ids'] = [(ids + [[pad_char_id] * pad_char_len] * (pad_p_len - len(ids)))[:pad_p_len]
                                          for ids in batch_data['passage_char_ids']]

        # print(np.array(batch_data['passage_char_ids']).shape, "==========")

        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        for index, char_list in enumerate(batch_data['question_char_ids']):
            for char_index in range(len(char_list)):
                if len(char_list[char_index]) >= pad_char_len:
                    char_list[char_index] = char_list[char_index][:self.max_char_len]
                else:
                    char_list[char_index] += [pad_char_id] * (pad_char_len - len(char_list[char_index]))
            batch_data['question_char_ids'][index] = char_list
        batch_data['question_char_ids'] = [(ids + [[pad_char_id] * pad_char_len] * (pad_q_len - len(ids)))[:pad_q_len]
                                           for ids in batch_data['question_char_ids']]

        return batch_data, pad_p_len, pad_q_len

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['question_tokens']:
                    yield token
                for passage in sample['passages']:
                    for token in passage['passage_tokens']:
                        yield token

    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_token_ids'] = vocab.convert_word_to_ids(sample['question_tokens'])
                sample["question_char_ids"] = vocab.convert_char_to_ids(sample['question_tokens'])
                for passage in sample['passages']:
                    passage['passage_token_ids'] = vocab.convert_word_to_ids(passage['passage_tokens'])
                    passage['passage_char_ids'] = vocab.convert_char_to_ids(passage['passage_tokens'])

    def next_batch(self, set_name, batch_size, pad_id, pad_char_id, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id, pad_char_id)

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch2(data, batch_indices, pad_id)

    def sort_passage(self, segmented_paragraphs, paragraphs, segmented_title, segmented_answers, segmented_question,
                     train=False):
        if train:
            if segmented_answers == []:
                return segmented_paragraphs, paragraphs
                # print(sample['segmented_answers'])
                # print('###/n')
        j = 0
        NO_passage = {}
        passage_list = []
        len_paragraphs = 0
        for sentence in paragraphs:
            len_paragraphs += len(sentence)
        # print (len_paragraphs)
        if (len(segmented_title) + len_paragraphs) > self.max_p_len:
            for passage in segmented_paragraphs:
                # print('###')
                # print(doc['segmented_paragraphs'])
                j = j + 1
                if train:
                    sorce = sentence_bleu(passage, word_tokenize(segmented_answers[0]),
                                          weights=(0.25, 0.25, 0.25, 0.25))
                    # print(sorce)
                else:
                    sorce = sentence_bleu(passage, word_tokenize(segmented_question),
                                          weights=(0.25, 0.25, 0.25, 0.25))
                NO_passage[j] = sorce
            #print(NO_passage)
            s_NO = list(NO_passage.items())
            s_NO.sort(key=lambda x: x[1], reverse=True)
            # print(list(s_NO))
            # 提取前3个中原位置最前的段落
            fist_passage = list(s_NO[0])[0]
            for t in s_NO[:3]:
                fist_passage = min(fist_passage, list(t)[0])
            #print(fist_passage)
            # print(segmented_paragraphs[fist_passage - 1])
            passage_list.append(segmented_paragraphs[fist_passage - 1])
            if segmented_paragraphs[fist_passage - 1] != segmented_paragraphs[-1]:
                passage_list.append(segmented_paragraphs[fist_passage])
            passage_idx = 0
            for passage in segmented_paragraphs:
                passage_idx = passage_idx + 1
                if passage_idx != fist_passage and passage_idx != fist_passage + 1:
                    new_passage = []
                    for word in word_tokenize(passage):
                        new_passage.append(word)
                        if word == '。':
                            # print(word)
                            break
                    passage_list.append(new_passage)
            segmented_paragraphs = passage_list
            # paragraph = []
            # for s_p in segmented_paragraphs:
            #     one_s_p = ""
            #     for word in s_p:
            #         one_s_p += word
            #     paragraph.append(one_s_p)
            # paragraphs = paragraph

        return segmented_paragraphs
