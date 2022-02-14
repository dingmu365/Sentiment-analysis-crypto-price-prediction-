import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import  BertTokenizer,DistilBertTokenizer,RobertaTokenizer,AutoTokenizer
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import BERTFamily as fn

####global variable
RANDOM_SEED = 177
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
MAX_LENGTH = 64
BATCH_SIZE = 16
NUM_CLASSES = 3 # neutral, positive, negative
EPOCHS = 5
DROPOUT_PROB = 0.1
WEIGHT_DECAY = 0.01
NFOLDS = 10
LEARNING_RATE = 2e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
########

loss_function = nn.CrossEntropyLoss().to(device)

statement_df = pd.read_csv('./data NASDAQ/all-data.csv', encoding='latin-1',
                           header=None)
statement_df.columns = ['sentiment', 'statement']
statement_df = statement_df.drop_duplicates()
statement_df['statement'] = statement_df['statement'].apply(fn.clean_statements)
le = LabelEncoder()
statement_df['sentiment'] = le.fit_transform(statement_df['sentiment'])
#positive-2, neutral-1,negative-0
df_train, df_test = train_test_split(statement_df,
                                     test_size=0.1,
                                     random_state=RANDOM_SEED,
                                     stratify=statement_df['sentiment'].values)

df_val, df_test = train_test_split(df_test,
                                   test_size=0.5,
                                   random_state=RANDOM_SEED,
                                   stratify=df_test['sentiment'].values)

df_train_full = pd.concat([df_train, df_val])
# bert_train_ds = fn.create_dataset(df_train, bert_tokenizer, MAX_LENGTH)
# bert_test_ds = fn.create_dataset(df_test, bert_tokenizer, MAX_LENGTH)
# bert_val_ds = fn.create_dataset(df_val, bert_tokenizer, MAX_LENGTH)
#
# bert_train_dataloader = fn.create_dataloader(bert_train_ds, BATCH_SIZE)
# bert_test_dataloader = fn.create_dataloader(bert_test_ds, BATCH_SIZE)
# bert_val_dataloader = fn.create_dataloader(bert_val_ds, BATCH_SIZE)


############################################################################################################
#
#You need to call fit(...) or fit_transform(...) on your LabelEncoder before you try an access classes_,
# or you will get this error. The attribute is created by fitting.

# bert_history, bert_test_outputs = fn.get_oof_and_test_preds(model_type='bert-base-cased',
#                                                             tokenizer=bert_tokenizer,
#                                                             train_df=df_train_full,
#                                                             test_df=df_test,
#                                                             single_model=False,
#                                                             loss_fn=loss_function)
#
# print("bert history",bert_history)
# print("bert test outputs",bert_test_outputs)
import json
import datetime
# a = [1,2,3]
# with open('test.txt', 'w') as f:
#     f.write(json.dumps(a))
#
# #Now read the file back into a Python list object
# with open('test.txt', 'r') as f:
#     a = json.loads(f.read())
# finbert_tokenizer = AutoTokenizer.from_pretrained("ipuneetrathore/bert-base-cased-finetuned-finBERT")
# finbert_history, finbert_test_outputs = fn.get_oof_and_test_preds(model_type='bert-base-cased-finetuned-finBERT',
#                                                             tokenizer=finbert_tokenizer,
#                                                             train_df=df_train_full,
#                                                             test_df=df_test,
#                                                             single_model=False,
#                                                             loss_fn=loss_function)

# path = './model/model_' + datetime.datetime.today().strftime('%Y_%m_%d')
#
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
# bert_history, bert_test_outputs = fn.get_oof_and_test_preds(model_type='bert-base-uncased',
#                                                             tokenizer=bert_tokenizer,
#                                                             train_df=df_train_full,
#                                                             test_df=df_test,
#                                                             single_model=False,
#                                                             loss_fn=loss_function)
# with open(path + '/bert_history.txt','w') as f:
#     f.write(json.dumps(bert_history))
#
# print("distilbert")
# distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# distilbert_history, distilbert_test_outputs = fn.get_oof_and_test_preds(model_type='distilbert',
#                                                                      tokenizer=distilbert_tokenizer,
#                                                                      train_df=df_train_full,
#                                                                      test_df=df_test,
#                                                                      single_model=False,
#                                                                      loss_fn=loss_function)
# with open(path +'/distil_history.txt','w') as f:
#     f.write(json.dumps(distilbert_history))
# print("robert")
# roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# roberta_history, roberta_test_outputs = fn.get_oof_and_test_preds(model_type='roberta',
#                                                                tokenizer=roberta_tokenizer,
#                                                                train_df=df_train_full,
#                                                                test_df=df_test,
#                                                                single_model=False,
#                                                                loss_fn=loss_function)
# with open(path+'/roberta_history.txt','w') as f:
#     f.write(json.dumps(roberta_history))
# df_test.to_csv(path +"/31_07_test_data.csv")
# num of sentiments
# fig, ax = plt.subplots(figsize=(12, 8))
# sns.countplot(x='sentiment', data=statement_df)
# plt.xlabel('')
# plt.ylabel('num of statement')
# plt.title('num of Sentiments')
# plt.show()
# # number of characters by class
# fig, ax = plt.subplots(figsize=(12, 8))
# sns.boxplot(x='sentiment', y='num_char', data=statement_df)
# plt.xlabel('')
# plt.ylabel('# of Characters')
# plt.title('# of Characters by Sentiment')
# plt.show()

# bert_model = fn.BERTSentimentClassifier(NUM_CLASSES)
# bert_model = bert_model.to(device)
# training_steps = len(bert_train_dataloader.dataset) * EPOCHS
#
# bert_optimizer = AdamW(bert_model.parameters(),
#                        lr=LEARNING_RATE,
#                        weight_decay=WEIGHT_DECAY,
#                        correct_bias=True)
#
# warmup_steps = int(0.1 * training_steps)
# bert_scheduler = get_linear_schedule_with_warmup(bert_optimizer,
#                                                  num_warmup_steps=warmup_steps,
#                                                  num_training_steps=training_steps)
# bert_single_model_items = fn.train_fold(epochs=EPOCHS,
#                                         model=bert_model,
#                                         device=device,
#                                         train_dataloader=bert_train_dataloader,
#                                         val_dataloader=bert_val_dataloader,
#                                         test_dataloader=bert_test_dataloader,
#                                         loss_fn=loss_function,
#                                         optimizer=bert_optimizer,
#                                         scheduler=bert_scheduler,
#                                         model_save_name="./model/bert_best_model.bin",
#                                         n_train=len(df_train),
#                                         n_val=len(df_val),
#                                         single_model=True
#                                         )

# bert_model=torch.load('./model/bert_best_model.bin')