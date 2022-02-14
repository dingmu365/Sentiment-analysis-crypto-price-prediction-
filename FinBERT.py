## Load required packages
import csv
import BERTFamily as fn
import pandas as pd
from dataclasses import make_dataclass
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold


# = """A tinyurl link takes users to a scamming site promising that users can earn thousands of dollars by becoming a Google ( NASDAQ : GOOG ) Cash advertiser ."""


RANDOM_SEED = 177
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
MAX_LENGTH = 64
BATCH_SIZE = 16
NUM_CLASSES = 3  # neutral, positive, negative
EPOCHS = 25
DROPOUT_PROB = 0.1
WEIGHT_DECAY = 0.01
NFOLDS = 5
LEARNING_RATE = 1e-4
le = LabelEncoder()
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_SEED)

class FinBERT():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ipuneetrathore/bert-base-cased-finetuned-finBERT")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "ipuneetrathore/bert-base-cased-finetuned-finBERT")
        self.max_len = 160
        self.class_names = ['0', '1', '2']
        # self.train=train
        # if self.train:
        #     self.drop = nn.Dropout(DROPOUT_PROB)
        #     self.fc= nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        #     self.output = nn.Linear(self.model.config.hidden_size, len(self.class_names))

    # Add the encoded sentence to the list.
    def predict(self, statement):
        self.encoded_new = self.tokenizer.encode_plus(
            statement,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.max_len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )
        input_idst = (self.encoded_new['input_ids'])
        attention_maskst = (self.encoded_new['attention_mask'])

        input_idst = torch.cat([input_idst], dim=0)
        attention_maskst = torch.cat([attention_maskst], dim=0)

        new_test_output = self.model(input_idst, token_type_ids=None,
                                     attention_mask=attention_maskst)

        logits = new_test_output[0]
        predicted = logits.detach().numpy()
        # Store predictions
        flat_predictions = np.concatenate(predicted, axis=0)
        # For each sample, pick the label (0 or 1) with the higher score.
        new_predictions = np.argmax(flat_predictions).flatten()
        prediction = self.class_names[new_predictions[0]]


        return  prediction,flat_predictions


def FinBERT_prediction(data, statement="statement", FinBert=FinBERT()):
    results = pd.DataFrame(columns=["prediction","preds"])
    for index, item in data.iterrows():
        flat_predictions, pred= FinBert.predict(item[statement])
        # positive-2, neutral-1,negative-0
        results = results.append(pd.Series({"prediction": flat_predictions,"preds":pred}, name=index))
    return results


# RANDOM_SEED=177
# statement_df = pd.read_csv('./data/all-data.csv', encoding='latin-1',
#                            header=None)
# statement_df.columns = ['sentiment', 'statement']
# statement_df = statement_df.drop_duplicates()
# statement_df['statement'] = statement_df['statement'].apply(fn.clean_statements)
# le = LabelEncoder()
# statement_df['sentiment'] = le.fit_transform(statement_df['sentiment'])
# #positive-2, neutral-1,negative-0
# df_train, df_test = train_test_split(statement_df,
#                                      test_size=0.2,
#                                      random_state=RANDOM_SEED,
#                                      stratify=statement_df['sentiment'].values)
#
# df_val, df_test = train_test_split(df_test,
#                                    test_size=0.5,
#                                    random_state=RANDOM_SEED,
#                                    stratify=df_test['sentiment'].values)
#
# df_train_full = pd.concat([df_train, df_val])
# keys=["sentiment","statement"]
# preds=FinBERT_prediction(data=df_test,statement="statement")
# actual = [x for y in [df_test["sentiment"]] for x in y]
# pres = [x for y in [preds["prediction"]] for x in y]
# accuracy_score(y_true=actual, y_pred=pres)
