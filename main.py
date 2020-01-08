# -*- encoding: utf-8 -*-
import argparse
import logging
import torch
import torch.nn as nn
import random
import glob
import os
from model import Summary
from pyrouge import Rouge155

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description="extrative summary")

# device
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# model param
parser.add_argument("-data_path", type=str, default="./onlstm_data")
parser.add_argument("-save_path", type=str, default="./sum_model")
parser.add_argument("-lr", type=float, default=0.00001)
parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-max_len", type=int, default=10, help="The max length of a document.")
parser.add_argument("-epoch", type=int, default=10)
parser.add_argument("-model_name", type=str, default="")
parser.add_argument("-topk", type=int, default=3)
parser.add_argument("-rouge_dir", type=str, default="./rouge")
parser.add_argument("-mode", type=str, default="train")

args = parser.parse_args()


def load_data(corpus_type):
    logging.info("Loading Data")
    pts = sorted(glob.glob(args.data_path + '.' + corpus_type + '.[0-9]*.pt'))

    def _lazy_dataset_loader(pt_file):
        with open(pt_file) as f:
            dataset = torch.load(f)
            logging.info('Loading %s dataset from %s, number of examples: %d' % (corpus_type,pt_file, len(dataset)))
            return dataset

    random.shuffle(pts)
    for pt in pts:
        yield _lazy_dataset_loader(pt)


def bachify_data(corpus_type):
    batch_df = []
    batch_label = []
    batch_target = []
    batch_src = []
    for data_list in load_data(corpus_type):
        for sample in data_list:
            batch_df.append(sample['df'])
            batch_label.append(sample['label'])
            batch_target.append(sample['tgt_txt'])
            batch_src.append(sample['src_txt'])
            if len(batch_df) == args.batch_size:
                yield batch_df, batch_label, batch_target, batch_src
                batch_df = []
                batch_label = []
                batch_target = []
                batch_src = []


def train():
    """
    A simple Neural Network
    :return:
    """
    logging.info("Start Training!")
    corpus_type = 'train'
    summary = Summary(args.batch_size, args.max_len)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(summary.parameters(), lr=args.lr)
    summary.train()

    start_epoch = 0
    if args.model_name:
        checkpoint = torch.load(args.load_model)
        summary = checkpoint['model']
        start_epoch = checkpoint['epochs']

    start_epoch += 1 if start_epoch != 0 else start_epoch

    for epoch in range(start_epoch, args.epoch):
        epoch_loss = 0
        batch_num = 0
        for i, batch in enumerate(bachify_data(corpus_type)):
            batch_df, batch_label, _, _ = batch
            batch_df = torch.tensor(batch_df)
            batch_label = torch.tensor(batch_label)
            binary_output = summary(batch_df)

            # calculate loss
            loss = criterion(binary_output, batch_label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            batch_num += 1

        logging.info("Epoch {}: Total loss is {}, Avg loss is {}".format(epoch, epoch_loss, epoch_loss/batch_num))
        # store model
        model_name = "{}_epoch_model.tar".format(epoch)
        directory = os.path.join(args.save_path, model_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'model': summary.state_dict(),
            'loss': epoch_loss / batch_num,
            "epochs": epoch
        }, directory)

    logging.info("Finish Training!")


# calculate rouge
def cal_rouge(rouge_dir):
    tmp_dir = "/home/huqian/anaconda3/envs/drqa_env/lib/python3.6/site-packages/pyrouge"
    r = Rouge155(rouge_dir=tmp_dir)
    r.model_dir = rouge_dir + "/reference"
    r.system_dir = rouge_dir + "/candidate"
    r.model_filename_pattern = 'ref.(\d+).txt'
    r.system_filename_pattern = 'cand.(\d+).txt'
    rouge_results = r.convert_and_evaluate()
    result_dict = r.output_to_dict(rouge_results)

    return ">> Rouge - F(1/2//l): {:.2f}/{:.2f}/{:.2f}\n ROUGE- R(1/2//l): {:.2f}/{:.2f}/{:.2f} \n".format(
        result_dict["rouge_1_f_score"] * 100,
        result_dict["rouge_2_f_score"] * 100,
        result_dict["rouge_l_f_score"] * 100,
        result_dict["rouge_1_recall"] * 100,
        result_dict["rouge_2_recall"] * 100,
        result_dict["rouge_l_recall"] * 100
    )


def predict(model_name):
    """
    predict summary and calculate rouge
    :return:
    """
    # load model
    checkpoint = torch.load(model_name)
    summary = checkpoint['model']

    if not os.path.isdir(args.rouge_dir):
        os.mkdir(args.rouge_dir)
        os.mkdir(args.rouge_dir + "/candidate")
        os.mkdir(args.rouge_dir + "/reference")

    corpus_type = 'test'
    for i, batch in enumerate(bachify_data(corpus_type)):
        batch_df, _, batch_target, batch_src = batch
        predict_output = summary(batch_df)

        # 排序，随后选择topk个，找到对应index的sentence组成summary
        topk_indices = predict_output.topk(args.topk)[1].cpu().data().numpy()
        selected_sent = [batch_src[index] for index in topk_indices]
        hyp = ". ".join(selected_sent)
        ground = ". ".join(batch_target)

        # 将起存储成rouge可判断
        with open(args.rouge_dir + "/reference/ref.{}.txt".format(i + 1), 'w') as f:
            f.write(ground)

        with open(args.rouge_dir + "/candidate/cand.{}.txt".format(i + 1), 'w') as f:
            f.write(hyp)

    # 调用Rouge计算函数
    logging.info("Calculating Rouge score!")
    rouge_res = cal_rouge(args.rouge_dir)
    logging.info("Rouge result is : {}".format(rouge_res))


if __name__ == '__main__':
    if args.mode == "test":
        predict(args.model_name)
    else:
        train()
