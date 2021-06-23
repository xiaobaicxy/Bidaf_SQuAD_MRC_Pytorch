# -*- coding: utf-8 -*-
import torch

class DataConfig:
    def __init__(self):
        # 
        self.data_dir = "/home/fuyong/workspace/dataset/SQuAD"
        self.preprocessed_dir = "./preprocessed"
        self.torchtexted_dir = "./torchtexted"
        self.context_threshold = 400

        self.train_batch_size = 2 # 受资源限制（原文60）
        self.dev_batch_size = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word_embed_size = 100

class ModelConfig(DataConfig):
    def __init__(self, data):
        super(ModelConfig, self).__init__()
        self.data = data
        self.char_vocab_size = len(data.CHAR.vocab)
        self.word_vocab_size = len(data.WORD.vocab)
        self.pretrained = data.WORD.vocab.vectors

        self.char_embed_size = 100
        self.cnn_out_channels = self.word_embed_size
        self.embed_size = self.word_embed_size + self.cnn_out_channels
        self.hidden_size = 100
        self.lr = 0.5
        self.epochs = 20 # batch_size太小，收敛应该会更困难，所以多跑几个epoch（原文12）
        self.print_freq = 500

        self.dev_dataset_file = "/home/fuyong/workspace/dataset/SQuAD/dev.json"
        self.prediction_file = "./prediction.json"
