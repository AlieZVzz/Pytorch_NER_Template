# -*- encoding: utf-8 -*-
# Author: Sparkling Deng
# Time: 2022/4/1
# Email: Sparklingdeng@outlook.com

from time import strftime, localtime
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random
import logging
from transformers import BertTokenizer
from torch.utils import data

from model.bert_lstm_crf import Bert_BiLSTM_CRF
from utils.dataset import NerDataset, TAGS, idx2tag, tag2idx
from utils.utils import pad
from utils.config_parser import ConfigArgumentParser, save_args
from metrics.ner_f1 import f1_score

formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)


class Named_Entity_Recognition():
    def __init__(self, opt):
        self.opt = opt
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.cuda.empty_cache()
        self.model = Bert_BiLSTM_CRF(self.opt.model_name, tag2idx, train_type=opt.train_type).to(self.opt.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.opt.model_name)
        logger.info('Initial model Done')
        self._print_args()
        self.train_dataset = NerDataset(opt.trainset, self.tokenizer)
        self.eval_dataset = NerDataset(opt.validset, self.tokenizer)

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info(
            '> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _train(self, iterator, optimizer, criterion, device):
        self.model.train()
        for i, batch in enumerate(iterator):
            # clear gradient accumulators
            optimizer.zero_grad()
            words, x, is_heads, tags, y, seqlens = batch

            x = x.to(device)
            y = y.to(device)
            _y = y  # for monitoring

            loss = self.model.neg_log_likelihood(x, y)  # logits: (N, T, VOCAB), y: (N, T)

            # logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
            # y = y.view(-1)  # (N*T,)
            # writer.add_scalar('data/loss', loss.item(), )

            # loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            if i == 0:
                logger.info("=====sanity check======")
                # logger.info("words:", words[0])
                logger.info("input_ids:%s", x.cpu().numpy()[0][:seqlens[0]])
                # logger.info("attention_mask:%s", x['attention_mask'].cpu().numpy()[0][:seqlens[0]])
                # # logger.info("tokens:", tokenizer.convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
                logger.info("y:%s", _y.cpu().numpy()[0][:seqlens[0]])
                logger.info("tags:%s", tags[0])
                logger.info("seqlen:%s", seqlens[0])
                logger.info("=======================")

            if i % 10 == 0:  # monitoring
                logger.info("step %s, loss %s", i, loss.item())

    def _eval(self, iterator, f, device):
        self.model.eval()
        recall_list, precision_list, f1_list = [], [], []
        Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                words, x, is_heads, tags, y, seqlens = batch
                x = x.to(device)
                # y = y.to(device)

                _, y_hat = self.model(x)  # y_hat: (N, T)

                Words.extend(words)
                Is_heads.extend(is_heads)
                Tags.extend(tags)
                Y.extend(y.numpy().tolist())
                Y_hat.extend(y_hat.cpu().numpy().tolist())

        for tag in TAGS:
            recall, precision, f1 = f1_score(y, y_hat, tag, tag2idx)
            recall_list.append(recall)
            precision_list.append(precision)
            f1_list.append(f1)
            logger.info("tag:%s  Recall:%s  Precision:%s  f1:%s", tag, recall, precision, f1)

        # gets results and save
        with open("temp", 'w', encoding='utf-8') as fout:
            for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
                y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
                preds = [idx2tag[hat] for hat in y_hat]
                assert len(preds) == len(words.split()) == len(tags.split())
                for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                    fout.write(f"{w} {t} {p}\n")
                fout.write("\n")

        # calculate metric
        y_true = np.array(
            [tag2idx[line.split()[1]] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if
             len(line) > 0])
        y_pred = np.array(
            [tag2idx[line.split()[2]] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if
             len(line) > 0])
        num_proposed = len(y_pred[y_pred > 1])
        num_correct = (np.logical_and(y_true == y_pred, y_true > 1)).astype(int).sum()
        num_gold = len(y_true[y_true > 1])

        logger.info("num_proposed:%s", num_proposed)
        logger.info("num_correct:%s", num_correct)
        logger.info("num_gold:%s", num_gold)

        test_pre = float(np.mean(precision_list))
        test_recall = float(np.mean(recall_list))
        test_f1 = float(np.mean(f1_list))
        final = f + ".P%.2f_R%.2f_F%.2f.txt" % (test_pre, test_recall, test_f1)
        with open(final, 'w', encoding='utf-8') as fout:
            result = open("temp", "r", encoding='utf-8').read()
            fout.write(f"{result}\n")

            fout.write(f"precision={test_pre}\n")
            fout.write(f"recall={test_recall}\n")
            fout.write(f"f1={test_f1}\n")

        os.remove("temp")

        logger.info("precision=%.4f", test_pre)
        logger.info("recall=%.4f", test_recall)
        logger.info("f1=%.4f", test_f1)
        return test_pre, test_recall, test_f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = optim.Adam(self.model.parameters(), lr=self.opt.lr)
        logger.info('Load Data Done')

        train_iter = data.DataLoader(dataset=self.train_dataset, batch_size=self.opt.batch_size, shuffle=True,
                                     num_workers=0,
                                     collate_fn=pad)
        eval_iter = data.DataLoader(dataset=self.eval_dataset, batch_size=self.opt.batch_size, shuffle=False,
                                    num_workers=0,
                                    collate_fn=pad)

        logger.info('Start Train...,')
        for epoch in range(1, self.opt.n_epochs + 1):
            if not os.path.exists(self.opt.logdir):
                os.makedirs(self.opt.logdir)
            fname = os.path.join(self.opt.logdir, str(epoch))

            logger.info(f"=========train at epoch={epoch}=========")
            self._train(train_iter, optimizer, criterion, self.opt.device)

            logger.info(f"=========eval at epoch={epoch}=========")
            precision, recall, f1 = self._eval(eval_iter, fname, self.opt.device)
            self.opt.early_stopping(f1, self.model)
            if self.opt.early_stopping.early_stop:
                logger.info("Early stopping")
                break


def main():
    parser = ConfigArgumentParser(description="")
    parser.add_argument("--work_dir", type=str, default="log", metavar="DIR",
                        help="Directory to save args.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2022, help="random seed")
    parser.add_argument("--lr", type=float, default=0.001, help="optimizer learning rate")
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--model_name", type=str, default='bert-base-uncased')
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/target_predict")
    parser.add_argument("--modeldir", type=str, default="checkpoints")
    parser.add_argument("--trainset", type=str, default="data/train_bmes.txt")
    parser.add_argument("--validset", type=str, default="data/valid_bmes.txt")
    parser.add_argument("--train_type", type=str, default='PLM-BiLSTM-CRF')  # bert_bilstm_crf bilstm_crf bert_crf
    opt = parser.parse_args()
    save_args(opt, os.path.join(opt.work_dir, "runtime_config.yaml"))

    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.train_type, strftime("%y%m%d-%H%M", localtime()))
    fh = logging.FileHandler('log/' + log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # os.environ['PYTHONHASHSEED'] = str(opt.seed)

    class EarlyStopping:
        """Early stops the training if validation loss doesn't improve after a given patience."""

        def __init__(self, patience=5, verbose=False, delta=0):
            """
            Args:
                patience (int): How long to wait after last time validation loss improved.
                                Default: 7
                verbose (bool): If True, prints a message for each validation loss improvement.
                                Default: False
                delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                                Default: 0
            """
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_f1_min = np.Inf
            self.delta = delta
            self.filename = None

        def __call__(self, val_f1, model):

            score = val_f1

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_f1, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_f1, model)
                self.counter = 0

        def save_checkpoint(self, val_f1, model):
            """Saves model when validation loss decrease."""
            if self.verbose:
                logger.info(f'Validation f1 increased ({self.val_f1_min:.6f} --> {val_f1:.6f}).  Saving model ...')

            # torch.save(model.state_dict(), 'checkpoint.pt')	# 这里会存储迄今最优模型的参数
            if self.filename is not None:
                os.remove(self.filename)

            self.filename = os.path.join(opt.modeldir, opt.model_name + '_' + opt.train_type + '_' + str(val_f1) + '_params.pth')
            torch.save(model, self.filename)

            self.val_f1_min = val_f1

    opt.early_stopping = EarlyStopping(opt.patience, verbose=True)
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.criterion = nn.CrossEntropyLoss(ignore_index=0)

    NER = Named_Entity_Recognition(opt)
    NER.run()


if __name__ == '__main__':
    main()
    # add_argument("-c", "--config", default=None, metavar="FILE",
    #                                         help="Where to load YAML configuration.")
