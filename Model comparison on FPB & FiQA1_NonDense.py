#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
import pandas as pd
import numpy as np
import BERTFamily as fn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

plt.style.use('ggplot')
from transformers import  BertTokenizer,DistilBertTokenizer, RobertaModel, RobertaTokenizer
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch

from torch import nn
from sklearn.model_selection import train_test_split
import os
import sys

from dataclasses import make_dataclass
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score, confusion_matrix, classification_report


# # Comparison on Financial PhrraseBank

# In[2]:



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
#device = torch.device('cpu')
loss_function = nn.CrossEntropyLoss().to(device)


# ## data loading

# In[3]:


statement_df = pd.read_csv('./data/all-data.csv', encoding='latin-1',
                           header=None)
statement_df.columns = ['sentiment', 'statement']
statement_df = statement_df.drop_duplicates()
statement_df['statement'] = statement_df['statement'].apply(fn.clean_statements)
le = LabelEncoder()
statement_df['sentiment'] = le.fit_transform(statement_df['sentiment'])
#positive-2, neutral-1,negative-0
df_train, df_test = train_test_split(statement_df,
                                     test_size=0.2,
                                     random_state=RANDOM_SEED,
                                     stratify=statement_df['sentiment'].values)

df_val, df_test = train_test_split(df_test,
                                   test_size=0.5,
                                   random_state=RANDOM_SEED,
                                   stratify=df_test['sentiment'].values)

df_train_full = pd.concat([df_train, df_val])
keys=["sentiment","statement"]


# In[4]:


statement_df.head(5)


# In[5]:


import re
def model_name(string):
    string=string.replace("_fold_","")
    string=string.replace("_","")
    string=re.sub(r'\d+', "",string, count=0, flags=0)
    return string


# In[6]:



def load_prediction(path,model_path,file,statement,device):
        
    if "roberta" in path:
        model_type = "roberta"
    elif "distilbert" in path:
        model_type = "distilbert"
    elif "bert-base-cased" in path:
        model_type = "bert-base-cased"
    elif "bert-base-uncased" in path:
        model_type = "bert-base-uncased"
    elif "bert-large-cased" in path:
        model_type = "bert-large-cased"
    elif "bert-large-uncased" in path:
        model_type = "bert-large-uncased"
    else:
        print("wrong model type")
        return None
   
    model, tokenizer = fn.load_model(model_type, model_path)
    model=model.to(device)
    model = model.eval()
    
    predictions,preds = fn.pred_model(input_data=file[statement],
                             model=model,
                             tokenizer=tokenizer)
    return predictions,preds


# In[7]:


dir = f"./model/model_2021_08_03/best"
paths = os.listdir(dir)


# In[8]:


#micro-F1 = micro-precision = micro-recall = accuracy
# cm = confusion_matrix(labels, preds)
# recall = np.diag(cm) / np.sum(cm, axis = 1)
# precision = np.diag(cm) / np.sum(cm, axis = 0)
# print(cm)
# print(recall,precision)
# print(classification_report(labels,preds))


# In[9]:


def scores(preds,predictions,lables,name):
    labels=df_test["sentiment"]
    record=pd.DataFrame(columns=["model","accuracy","MSE"])
    
    accuracy=accuracy_score(predictions,labels)
    loss=loss_function(preds,
               torch.from_numpy(labels.values).type(torch.LongTensor))
    record=record.append({"model":name,"accuracy":accuracy,"MSE":loss.item()},ignore_index=True)
    return record


# In[11]:


paths


# In[12]:


# %reload_ext autoreload
# %autoreload 1
# %aimport BERTFamily
# import BERTFamily as fn

model_preds=pd.DataFrame()
total_model_record=pd.DataFrame(columns=["model","MSE","accuracy"])
labels=df_test["sentiment"]
file=df_test
statement="statement"
for path in paths:
    model_path = dir + "/" + path 
    prediction,preds=load_prediction(path,model_path,file,statement,device)
    name=model_name(path.replace(".bin",""))
    prediction["model"]=name
    record=scores(preds,prediction[0],labels,name)
   
    total_model_record=total_model_record.append(record,ignore_index=True)
     
    model_preds=model_preds.append(prediction,ignore_index=True)


# ## Baseline: LM_dictionary

# In[ ]:


from LM_dictionary import LM_prediction

LM_preds = LM_prediction(df_test)


# In[ ]:


accuracy=accuracy_score(LM_preds.values.astype('float') ,labels)

total_model_record=total_model_record.append({"model":"LM","MSE":None,"accuracy":accuracy},ignore_index=True)


# ## case-base-BERT

# In[ ]:


paths


# In[ ]:


labels=df_test["sentiment"]
record=pd.DataFrame()
Point = make_dataclass("Point", [("model", str), ("accuracy")])
model=model_name("_bert-base-cased_fold_0")
print(model)
preds=model_preds[model_preds["model"]==model][0]
print(accuracy_score(preds,labels))
print(classification_report(preds,labels))


# ## uncase-base-BERT

# In[ ]:


model="_bert-base-uncased_fold_3"
model=model_name(model)
preds=model_preds[model_preds["model"]==model][0]
print(accuracy_score(preds,labels))
print(classification_report(preds,labels))


# ## case-large-BERT

# In[ ]:


model="_bert-large-cased_fold_0"
model=model_name(model)
print(model)
preds=model_preds[model_preds["model"]==model][0]
print(accuracy_score(preds,labels))
print(classification_report(preds,labels))


# ## uncase-large-BERT

# In[ ]:


model="_bert-large-uncased_fold_2"
model=model_name(model)
print(model)
preds=model_preds[model_preds["model"]==model][0]
print(accuracy_score(preds,labels))
print(classification_report(preds,labels))


# ## RoBERTa
# 

# In[ ]:


model="_roberta_fold_4"
model=model_name(model)
print(model)
preds=model_preds[model_preds["model"]==model][0]
print(accuracy_score(preds,labels))
print(classification_report(preds,labels))


# ## DistilBERT

# In[ ]:


model="_distilbert_fold_2"
model=model_name(model)
print(model)
preds=model_preds[model_preds["model"]==model][0]
print(accuracy_score(preds,labels))
print(classification_report(preds,labels))


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')
get_ipython().run_line_magic('aimport', 'FinBERT')

from FinBERT import *
result=FinBERT_prediction(data=df_test,statement="statement")


# In[ ]:


actual = [x for y in [df_test["sentiment"]] for x in y]
pres = [x for y in [result["prediction"]] for x in y]
preds=[x for y in [result["preds"]] for x in y]
accuracy_score(y_true=actual, y_pred=pres)
record=scores(torch.as_tensor(np.array(preds).astype('float')),pres,labels,name="finbert")
total_model_record=total_model_record.append(record,ignore_index=True)
print(classification_report(pres,labels))


# In[ ]:


total_model_record.sort_values("MSE")


# In[ ]:


total_model_record.sort_values("accuracy")


# # Comparison on FiQA1

# In[ ]:



import json


path_read=f"./data/task1_headline_ABSA_train.json"
QA_data=pd.DataFrame(columns=["statement","sentiment"])
with open(path_read) as f:
    data = json.load(f)
    data_items = data.items()
    data_list = list(data_items)
    for i,content in data_items:
        data_items = data.items()
        data_list = list(data_items)
        QA_data=QA_data.append({"statement":content["sentence"],"sentiment":content["info"][0]['sentiment_score']},ignore_index=True)

f.close()
QA_data


# In[ ]:





# In[ ]:




