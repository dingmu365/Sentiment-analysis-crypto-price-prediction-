import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn

plt.style.use('ggplot')
from transformers import BertModel, BertTokenizer, DistilBertTokenizer, RobertaModel, RobertaTokenizer
from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup
import torch
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
loss_function = nn.CrossEntropyLoss().to(device)

########
statement_df = pd.read_csv('./data NASDAQ/all-data.csv', encoding='latin-1',
                           header=None)
statement_df.columns = ['sentiment', 'statement']
statement_df = statement_df.drop_duplicates()
statement_df['statement'] = statement_df['statement'].apply(fn.clean_statements)
statement_df['num_char'] = statement_df['statement'].apply(len)
statement_df['num_words'] = statement_df['statement'].apply(lambda x: len(x.split()))
le = LabelEncoder()
statement_df['sentiment'] = le.fit_transform(statement_df['sentiment'])
df_train, df_test = train_test_split(statement_df,
                                     test_size=0.1,
                                     random_state=RANDOM_SEED,
                                     stratify=statement_df['sentiment'].values)

df_val, df_test = train_test_split(df_test,
                                   test_size=0.5,
                                   random_state=RANDOM_SEED,
                                   stratify=df_test['sentiment'].values)

df_train_full = pd.concat([df_train, df_val])
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# roberta_train_ds = fn.create_dataset(df_train, roberta_tokenizer, MAX_LENGTH)
# roberta_test_ds = fn.create_dataset(df_test, roberta_tokenizer, MAX_LENGTH)
# roberta_val_ds = fn.create_dataset(df_val, roberta_tokenizer, MAX_LENGTH)
#
# roberta_train_dataloader = fn.create_dataloader(roberta_train_ds, BATCH_SIZE)
# roberta_test_dataloader = fn.create_dataloader(roberta_test_ds, BATCH_SIZE)
# roberta_val_dataloader = fn.create_dataloader(roberta_val_ds, BATCH_SIZE)
# roberta_model = fn.RobertaSentimentClassifier(n_classes=NUM_CLASSES)
# roberta_model = roberta_model.to(device)
#
# training_steps = len(roberta_train_dataloader.dataset) * EPOCHS
#
# roberta_optimizer = AdamW(roberta_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, correct_bias=True)
# roberta_scheduler = get_linear_schedule_with_warmup(roberta_optimizer, num_warmup_steps=int(0.1 * training_steps), num_training_steps=training_steps)
#
#\ roberta_single_model_items = fn.train_fold(epochs=EPOCHS,
#                                            model=roberta_model,
#                                            device=device,
#                                            train_dataloader=roberta_train_dataloader,
#                                            val_dataloader=roberta_val_dataloader,
#                                            test_dataloader=roberta_test_dataloader,
#                                            loss_fn=loss_function,
#                                            optimizer=roberta_optimizer,
#                                            scheduler=roberta_scheduler,
#                                            model_save_name='./model/roberta_best_model.bin',
#                                            n_train=len(df_train),
#                                            n_val=len(df_val),
#                                            single_model=True
#                                            )
#
# ############################################################################################################
# TODO: debug following part
# 10-Fold CV
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_history, roberta_test_outputs = fn.get_oof_and_test_preds(model_type='roberta',
                                                               tokenizer=roberta_tokenizer,
                                                               train_df=df_train_full,
                                                               test_df=df_test,
                                                               single_model=False,
                                                               loss_fn=loss_function)


#fn.cv_ensemble_performance(roberta_test_outputs, df_test['sentiment'].values)