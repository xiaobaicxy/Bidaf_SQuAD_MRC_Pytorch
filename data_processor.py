import json
import os
import nltk
import torch

from torchtext.legacy import data
from torchtext import datasets
from torchtext.vocab import GloVe

from pathlib import Path

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


class DataProcessor():
    def __init__(self, config):
        if not os.path.exists(config.torchtexted_dir) and \
            not os.path.exists(config.preprocessed_dir):
            self.preprocess_file(config)

        # torchtext的各种Filed
        # https://pytorch.org/text/_modules/torchtext/data/field.html
        self.RAW = data.RawField()
        self.RAW.is_target = False
        
        self.CHAR_NESTING = data.Field(batch_first=True, tokenize=list, lower=True)
        self.CHAR = data.NestedField(self.CHAR_NESTING, tokenize=word_tokenize) # 先执行word_tokenize再执行list
        self.WORD = data.Field(batch_first=True, tokenize=word_tokenize, lower=True, include_lengths=True)
        self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)

        dict_fields = {
                        'id': ('id', self.RAW),
                        's_idx': ('s_idx', self.LABEL),
                        'e_idx': ('e_idx', self.LABEL),
                        'context': [('c_word', self.WORD), ('c_char', self.CHAR)],
                        'question': [('q_word', self.WORD), ('q_char', self.CHAR)]
                    }

        list_fields = [
                        ('id', self.RAW), 
                        ('s_idx', self.LABEL), 
                        ('e_idx', self.LABEL),
                        ('c_word', self.WORD), 
                        ('c_char', self.CHAR),
                        ('q_word', self.WORD), 
                        ('q_char', self.CHAR)
                    ]

        if os.path.exists(config.torchtexted_dir):
            torchtexted_train_path = Path(config.torchtexted_dir).joinpath('train.pkl')
            torchtexted_dev_path = Path(config.torchtexted_dir).joinpath('dev.pkl')
            train_examples = torch.load(torchtexted_train_path)
            dev_examples = torch.load(torchtexted_dev_path)

            self.train = data.Dataset(examples=train_examples, fields=list_fields)
            self.dev = data.Dataset(examples=dev_examples, fields=list_fields)
        else:
            self.train, self.dev = data.TabularDataset.splits(
                                        config.preprocessed_dir,
                                        train='train.json',
                                        validation='dev.json',
                                        format='json',
                                        fields=dict_fields
                                    )

            os.makedirs(config.torchtexted_dir)
            torchtexted_train_path = Path(config.torchtexted_dir).joinpath('train.pkl')
            torchtexted_dev_path = Path(config.torchtexted_dir).joinpath('dev.pkl')
            torch.save(self.train.examples, torchtexted_train_path)
            torch.save(self.dev.examples, torchtexted_dev_path)

        # 过滤掉训练数据中过长的文本
        if config.context_threshold > 0:
            self.train.examples = [e for e in self.train.examples if len(e.c_word) <= config.context_threshold]

        # 创建字典
        self.CHAR.build_vocab(self.train, self.dev)
        self.WORD.build_vocab(self.train, self.dev, vectors=GloVe(name='6B', dim=config.word_embed_size))

        # 创建iterators
        self.train_iter = data.BucketIterator(
            self.train,
            batch_size=config.train_batch_size,
            device=config.device,
            repeat=True,
            shuffle=True,
            sort_key=lambda x: len(x.c_word)
        )

        self.dev_iter = data.BucketIterator(
            self.dev,
            batch_size=config.dev_batch_size,
            device=config.device,
            repeat=False,
            shuffle=False,
            sort_key=lambda x: len(x.c_word)
        )

      
    def preprocess_file(self, config):
        # 更新span：标注中answer_start和计算得到的answer_end是按字符个数统计的，需要更新为token(word)的个数
        dump = []
        abnormals = [' ', '\n', '\u3000', '\u202f', '\u2009'] # 空格、换行字符

        src_train_path = Path(config.data_dir).joinpath('train-v2.0.json')
        src_dev_path = Path(config.data_dir).joinpath('dev-v2.0.json')
        src_path = {'train': src_train_path, 'dev': src_dev_path}
        
        os.makedirs(config.preprocessed_dir)
        preprocessed_train_path = Path(config.preprocessed_dir).joinpath('train.json')
        preprocessed_dev_path = Path(config.preprocessed_dir).joinpath('dev.json')
        dst_path = {'train': preprocessed_train_path, 'dev': preprocessed_dev_path}

        for data_type in src_path.keys():
            with open(src_path[data_type], 'r', encoding='utf-8') as f:
                data = json.load(f)
                data = data['data']

                for article in data:
                    for paragraph in article['paragraphs']:
                        context = paragraph['context']
                        tokens = word_tokenize(context)
                        for qa in paragraph['qas']:
                            id = qa['id']
                            question = qa['question']
                            for ans in qa['answers']:
                                answer = ans['text']
                                s_idx = ans['answer_start']
                                e_idx = s_idx + len(answer)

                                l = 0
                                s_found = False
                                for i, t in enumerate(tokens):
                                    # 在文本中遇到控格与换行符，需要将指向文本字符的指针右移
                                    while l < len(context): 
                                        if context[l] in abnormals:
                                            l += 1
                                        else:
                                            break

                                    # 特殊字符：文本中的"''"和"``"会被处理成'"'，为了统计token对应到原始文本的长度，需要将相应的token处理回去
                                    if t == '"':
                                        if context[l:l + 2] == '\'\'':
                                            t = '\'\''
                                        elif context[l:l+2] == '``':
                                            t = '``'
                                    elif t[0] == '"':
                                        if context[l:l + 2] == '\'\'':
                                            t = '\'\'' + t[1:]
                                        elif context[l:l + 2] == '``':
                                            t = '``' + t[1:]
                                        
                                    l += len(t) # 指向原始文本字符的指针右移token长度个位置

                                    # 当指向原始文本的指针第一次跳到标注的answer_start的右边时，更新answer_start，i从左到当前位置token的个数
                                    if l > s_idx and s_found == False: # 用>的原因：如果A是开始的token，当tokens[i] == "A"时， 标注的s_idx指向的是A，而此时l已经指向了A的下一个字符
                                        s_idx = i
                                        s_found = True
                                    if l >= e_idx: # 用>=的原因：如果XXX是结束的token，当tokens[i] == "XXX"时，标注的e_idx指向的是最后一个X的下一个字符，l也是指向最后一个X的下一个字符，因此可以取=
                                        e_idx = i
                                        break

                                dump.append(dict([
                                                    ('id', id),
                                                    ('context', context),
                                                    ('question', question),
                                                    ('answer', answer),
                                                    ('s_idx', s_idx),
                                                    ('e_idx', e_idx)
                                                ])
                                            )
                    break # for debug
                    
            with open(dst_path[data_type], 'w', encoding='utf-8') as f:
                for line in dump:
                    json.dump(line, f)
                    print('', file=f)

if __name__ == '__main__':
    class Config:
        def __init__(self):
            self.data_dir = "/home/fuyong/workspace/dataset/SQuAD"
            self.preprocessed_dir = "./preprocessed"
            self.torchtexted_dir = "./torchtexted"
            self.context_threshold = 512

            self.train_batch_size = 1
            self.dev_batch_size = 1
            self.device = "cpu"
            self.word_embed_size = 100
    
    config = Config()
    data_processor = DataProcessor(config)
    iters = data_processor.train_iter
    for it in iters:
        cur_epoch = iters.epoch
        print(cur_epoch)

        c_char = it.c_char
        print("char shape: ", c_char.shape)
        c_word = it.c_word[0]
        c_lens = it.c_word[1]
        print("word shape: ", c_word.shape)
        print("seq_len shape: ", c_lens.shape)
        
        # q_char = it.q_char
        # print(q_char.shape)
        # q_word = it.q_word[0]
        # print(q_word.shape)
        s_idx = it.s_idx
        print(s_idx)
        e_idx = it.e_idx
        print(e_idx)
        break

    # data_processor.preprocess_file(config)
