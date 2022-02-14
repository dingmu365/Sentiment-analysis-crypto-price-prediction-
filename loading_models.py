import csv

import numpy as np
import os
import pandas as pd
import re
import time
import datetime
import seaborn as sns
from dataclasses import make_dataclass
from transformers import BertModel, BertTokenizer, DistilBertTokenizer, RobertaModel, RobertaTokenizer
from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn, optim
from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from BERTFamily import *

RANDOM_SEED = 177
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
MAX_LENGTH = 64
BATCH_SIZE = 16
NUM_CLASSES = 3  # neutral, positive, negative
EPOCHS = 5
DROPOUT_PROB = 0.1
WEIGHT_DECAY = 0.01
NFOLDS = 10
LEARNING_RATE = 2e-5
le = LabelEncoder()
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_SEED)

##variables##
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dir = f"./model/model_2021_08_03"
keys = ["Date", "ID", "headline", "tags"]
statement = "headline"
data_path = f"./data/NASDAQ_headlines.csv"
paths = os.listdir(dir)

Point = make_dataclass("Point", [("date", str), ("ID", str), ('preds', int), ('prob_0', float), ('prob1', float),
                                 ('prob2', float)])
###################

for path in paths:
    print(path)
    if "distilbert_fold_9"in path:
        continue

    if "bert-base-uncased" in path:
        model_type = "bert-base-uncased"
        continue
    elif "bert-base-cased" in path:
        model_type = "bert-base-cased"
    elif "roberta" in path:
        model_type = "roberta"

    elif "distilbert" in path:
        model_type = "distilbert"
    else:
        print("wrong file", path)
        continue

    model_path = dir + "/" + path
    model, tokenizer = load_model(model_type, model_path)
    model = model.eval()
    file_path = f"./data/predictions/" + path.replace(".bin", ".csv")
    n = 0
    with open(data_path, 'r') as files:

        file = csv.DictReader((l.replace('\0', '') for l in files), fieldnames=keys)
        for row in file:
            rows = pd.DataFrame(columns=keys)
            predictions, pre_prob = predict_model(input_data=row[statement],
                                                  model=model,
                                                  tokenizer=tokenizer,
                                                  outpath=model_path.replace(".bin", "") + ".csv")
            print(n, predictions)
            n = n + 1
            pre_prob = pd.Series(pre_prob.flatten())
            statements = pd.DataFrame(
                [Point(row["Date"], row["ID"], predictions.__int__(), pre_prob[0], pre_prob[1], pre_prob[2])])

            statements.to_csv(
                file_path,
                index=False,
                header=False,
                mode='a',
                chunksize=1)
            del statements

        files.close()

# model_type='bert-base-cased'

# df = pd.read_csv('./data NASDAQ/all-data.csv', encoding='latin-1',
#                            header=None)
#
# df=pd.read_csv('./data/NASDAQ_headlines.csv', encoding='latin-1',
#                            header=None, nrows=100, names=["Date","id","statement","tags"])
# statement_df=pd.DataFrame(None,columns=['sentiment', 'statement'])
# statement_df["statement"]= df["statement"]
# statement_df["sentiment"]= -1
# statement_df = statement_df.drop_duplicates()
# statement_df['statement'] = statement_df['statement'].apply(clean_statements)

# (predictions,pre_prob)

# le = LabelEncoder()
# statement_df['sentiment'] = le.fit_transform(statement_df['sentiment'])
# model_type = 'bert-base-cased'

# return statement_texts, predictions, prediction_probs
