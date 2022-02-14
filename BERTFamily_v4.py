import numpy as np
import os
import pandas as pd
import re
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
plt.interactive(True)
from collections import Counter, defaultdict
from transformers import BertModel, BertTokenizer, DistilBertTokenizer, RobertaModel, RobertaTokenizer, AutoTokenizer, \
    AutoModelForSequenceClassification
from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

RANDOM_SEED = 177
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
MAX_LENGTH = 64
BATCH_SIZE = 16
NUM_CLASSES = 3  # neutral, positive, negative
EPOCHS = 5
DROPOUT_PROB = 0.1
WEIGHT_DECAY = 0.01
NFOLDS = 5
LEARNING_RATE = 1e-4
le = LabelEncoder()
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_SEED)
#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clean_statements(statement):
    statement = re.sub(" '", "'", statement)
    statement = re.sub(" 's", "'s", statement)
    statement = re.sub('\( ', '(', statement)
    statement = re.sub(' \)', ')', statement)
    statement = re.sub('``', '"', statement)
    statement = re.sub("''", '"', statement)
    statement = re.sub(r'\s([?.,%:!"](?:\s|$))', r'\1', statement)
    #     statement = statement.translate(str.maketrans('', '', string.punctuation))
    return statement


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


class GPReviewDataset(Dataset):

    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        reviews=df.content.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


class StatementDataset(Dataset):

    def __init__(self, statements, labels, tokenizer, max_length):
        self.statements = statements
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.statements)

    def __getitem__(self, idx):
        statement = str(self.statements[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            statement,
            max_length=self.max_length,
            padding='max_length',
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'statement_text': statement,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_dataset(df, tokenizer, max_length):
    ds = StatementDataset(statements=df['statement'].to_numpy(),
                          labels=df['sentiment'].to_numpy(),
                          tokenizer=tokenizer,
                          max_length=max_length)
    return ds


def create_dataloader(ds, batch_size):
    return DataLoader(ds, batch_size, num_workers=0)


class BERTSentimentClassifier(nn.Module):
    def __init__(self, n_classes, type):
        super(BERTSentimentClassifier, self).__init__()
        self.model = BertModel.from_pretrained(type)
        self.drop = nn.Dropout(DROPOUT_PROB)
        ########
        self.fc= nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        ########
        self.output = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        ########
        pooled_output = self.fc(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.drop(pooled_output)
        outputs = F.log_softmax(self.output(pooled_output))
        return outputs


class RobertaSentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super(RobertaSentimentClassifier, self).__init__()
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.drop = nn.Dropout(DROPOUT_PROB)
        ########

        self.fc = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        ########
        self.output = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = self.fc(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.drop(pooled_output)
        outputs = F.log_softmax(self.output(pooled_output))
        return outputs


class DistilBertForSequenceClassification(nn.Module):

    def __init__(self, pretrained_model_name, num_classes):
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained_model_name, num_labels=num_classes)
        self.distilbert = AutoModel.from_pretrained(pretrained_model_name, config=config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, num_classes)
        self.drop = nn.Dropout(config.seq_classif_dropout)

    def forward(self, input_ids, attention_mask=None, head_mask=None):
        assert attention_mask is not None, "No Attention Mask"
        distilbert_output = self.distilbert(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            head_mask=head_mask)

        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.drop(pooled_output)
        outputs = F.log_softmax(self.classifier(pooled_output))

        return outputs



def cv_ensemble_performance(preds, labels):
    preds = np.array(preds)
    summed = np.sum(preds, axis=0)
    preds = np.argmax(summed, axis=1)
    print(confusion_matrix(y_true=labels, y_pred=preds))
    print('')
    print(classification_report(y_true=labels, y_pred=preds, digits=3, target_names=le.classes_))


def single_model_performance(preds, labels):
    print(confusion_matrix(y_true=labels, y_pred=preds))
    print('')
    print(classification_report(y_true=labels, y_pred=preds, digits=3, target_names=le.classes_))


def train_model(model, device, data_loader, loss_function,
                optimizer, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_preds = 0
    complete_preds = []
    complete_labels = []

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_function(outputs, labels)
        complete_preds.append(preds.data.cpu().numpy().tolist())
        complete_labels.append(labels.data.cpu().numpy().tolist())
        correct_preds += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    complete_preds_flat = [x for y in complete_preds for x in y]
    complete_labels_flat = [x for y in complete_labels for x in y]
    acc_score = accuracy_score(y_true=complete_labels_flat,
                               y_pred=complete_preds_flat)
    return acc_score, np.mean(losses)




def eval_model(model, device, data_loader, loss_function, n_examples):
    model = model.eval()

    losses = []
    correct_preds = 0
    complete_preds = []
    complete_labels = []
    complete_outputs = []

    with torch.no_grad():
        for item in data_loader:
            input_ids = item['input_ids'].to(device)
            attention_mask = item['attention_mask'].to(device)
            labels = item['labels'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = loss_function(outputs, labels)

            correct_preds += torch.sum(preds == labels)
            complete_preds.append(preds.data.cpu().numpy().tolist())
            complete_labels.append(labels.data.cpu().numpy().tolist())
            complete_outputs.append(outputs.tolist())
            losses.append(loss.item())

        accuracy = correct_preds.double() / n_examples
        complete_preds_flat = [x for y in complete_preds for x in y]
        complete_labels_flat = [x for y in complete_labels for x in y]
        complete_outputs_flat = [x for y in complete_outputs for x in y]

        acc_score = accuracy_score(y_true=complete_labels_flat,
                                   y_pred=complete_preds_flat)

        return_items = (acc_score,
                        np.mean(losses),
                        complete_preds_flat,
                        complete_outputs_flat)

        return return_items


def train_fold(model_statement, epochs, model, device, train_dataloader,
               val_dataloader, test_dataloader, loss_fn, optimizer,
               scheduler, model_save_name, n_train, n_val, single_model=True):
    start_time = time.time()
    history = defaultdict(list)
    best_accuracy = 0
    mark=0
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print('Epoch ', epoch + 1, '/', epochs)
        print('-' * 50)

        training_output = train_model(model,
                                      device,
                                      train_dataloader,
                                      loss_fn,
                                      optimizer,
                                      scheduler,
                                      n_train)

        train_acc, train_loss = training_output

        val_output = eval_model(model,
                                device,
                                val_dataloader,
                                loss_fn,
                                n_val)

        val_acc, val_loss, val_preds, val_outputs = val_output
        history['model_statement'] = model_statement
        history['epoch'].append(epoch)
        history['train_accuracy'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_accuracy'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['val_preds'].append(val_preds)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), model_save_name)
            best_accuracy = val_acc
            best_preds = val_preds
            best_outputs = val_outputs
            record_loss = val_loss
        elif val_acc == best_accuracy:
            mark = mark + 1
        elif val_loss < record_loss:
            mark = 0
        else:
            mark = mark + 1
        print('Train Loss: ',
              train_loss,
              ' | ',
              'Train Accuracy: ',
              train_acc)
        print('Val Loss: ',
              val_loss,
              ' | ',
              'Val Accuracy: ',
              val_acc)
        print('Epoch Train Time: ',
              format_time(time.time() - epoch_start_time))
        print('\n')
        if mark >= 3:
            print("early stop")
            break

    print('Finished Training.')
    print('Fold Train Time: ', format_time(time.time() - start_time))
    print('\n')
    if single_model:
        _, _, test_preds, test_outputs = eval_model(model,
                                                    device,
                                                    test_dataloader,
                                                    loss_fn,
                                                    len(test_dataloader))

        single_model_performance(test_preds, test_dataloader.dataset.labels)
    # single_model_performance(test_preds, test_dataloader['sentiment'].values)

    return history, best_preds, best_outputs


def load_model(model_type, PATH):
    if model_type == 'bert-base-uncased':
        model = BERTSentimentClassifier(NUM_CLASSES, type=model_type)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    elif model_type == 'bert-base-cased':
        model = BERTSentimentClassifier(NUM_CLASSES, type=model_type)
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    elif model_type == 'bert-large-uncased':
        model = BERTSentimentClassifier(n_classes=NUM_CLASSES, type=model_type)
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

    elif model_type == 'bert-large-cased':
        model = BERTSentimentClassifier(n_classes=NUM_CLASSES,type=model_type)
        tokenizer = BertTokenizer.from_pretrained("bert-large-cased")

    elif model_type == 'distilbert':
        model = DistilBertForSequenceClassification(pretrained_model_name='distilbert-base-uncased',
                                                    num_classes=NUM_CLASSES)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    elif model_type == 'roberta':
        model = RobertaSentimentClassifier(n_classes=NUM_CLASSES)
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    else:
        print("given wrong model type")
        None
    model.load_state_dict(torch.load(PATH, map_location=device))

    return model, tokenizer


def pred_model(input_data, model, tokenizer,device=device):
    data = pd.DataFrame(columns=['statement', 'sentiment'])
    # data = data.append({'statement': input_data, 'sentiment': pd.Series(np.ones(len(input_data)))}, ignore_index=True)
    data['statement'] = input_data
    data['sentiment'] = 1
    ds = create_dataset(data, tokenizer, MAX_LENGTH)
    data_loader = create_dataloader(ds, len(input_data))
    for item in data_loader:
        input_ids = item['input_ids'].to(device)
        attention_mask = item['attention_mask'].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        result=pd.DataFrame(torch.cat((torch.reshape(preds,(len(input_data),1)),outputs),1).detach().numpy())

    return result,outputs

def predict_model(input_data, model, tokenizer, outpath):
    data = pd.DataFrame(columns=['statement', 'sentiment'])
    # data['statement']=input_data
    # data['sentiment']=-1
    data = data.append({'statement': input_data, 'sentiment': -1}, ignore_index=True)
    ds = create_dataset(data, tokenizer, MAX_LENGTH)
    data_loader = create_dataloader(ds, BATCH_SIZE)
    # model = model.eval()
    statement_texts = []
    predictions = []
    prediction_probs = []

    # with torch.no_grad():
    for item in data_loader:
        texts = item['statement_text']
        input_ids = item['input_ids'].to(device)
        attention_mask = item['attention_mask'].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        statement_texts.extend(texts)
        predictions.extend(preds)
        prediction_probs.extend(outputs)

    predictions = torch.stack(predictions)
    prediction_probs = torch.stack(prediction_probs)

    # data["preds"]=predictions

    return predictions.detach().numpy(), prediction_probs.detach().numpy()


def get_oof_and_test_preds(model_type, tokenizer,
                           train_df, test_df, loss_fn, single_model=False):
    oof_preds = []
    oof_outputs = []
    oof_preds_indices = []
    test_preds_list = []
    test_outputs_list = []
    history_list = []
    start_time = time.time()

    fold = 0

    x_train = train_df['statement']
    y_train = train_df['sentiment']

    for train_index, val_index in skf.split(x_train, y_train):
        print('Fold: {}'.format(fold + 1))

        x_tr = x_train.iloc[train_index]
        y_tr = y_train.iloc[train_index]
        x_va = x_train.iloc[val_index]
        y_va = y_train.iloc[val_index]

        train = pd.DataFrame(list(zip(x_tr, y_tr)),
                             columns=['statement', 'sentiment'])
        val = pd.DataFrame(list(zip(x_va, y_va)),
                           columns=['statement', 'sentiment'])

        train_ds = create_dataset(train, tokenizer, MAX_LENGTH)
        val_ds = create_dataset(val, tokenizer, MAX_LENGTH)
        test_ds = create_dataset(test_df, tokenizer, MAX_LENGTH)

        if model_type == 'bert-base-cased':
            model = BERTSentimentClassifier(type=model_type, n_classes=NUM_CLASSES)
            model = model.to(device)
        elif model_type == 'bert-base-uncased':
            model = BERTSentimentClassifier(type=model_type, n_classes=NUM_CLASSES)
            model = model.to(device)
        elif model_type == 'bert-large-uncased':
            model = BERTSentimentClassifier(type=model_type, n_classes=NUM_CLASSES)
            model = model.to(device)
        elif model_type == 'bert-large-cased':
            model = BERTSentimentClassifier(type=model_type, n_classes=NUM_CLASSES)
            model = model.to(device)
        elif model_type == 'distilbert':
            model = DistilBertForSequenceClassification(pretrained_model_name='distilbert-base-uncased',
                                                        num_classes=NUM_CLASSES)
            model = model.to(device)
        elif model_type == 'roberta':
            model = RobertaSentimentClassifier(n_classes=NUM_CLASSES)
            model = model.to(device)

        else:
            print("given wrong model type")

        train_loader = create_dataloader(train_ds, BATCH_SIZE)
        val_loader = create_dataloader(val_ds, BATCH_SIZE)
        test_loader = create_dataloader(test_ds, BATCH_SIZE)

        training_steps = len(train_loader.dataset) * EPOCHS
        warmup_steps = int(0.1 * training_steps)
        optimizer = AdamW(model.parameters(),
                          lr=LEARNING_RATE,
                          weight_decay=WEIGHT_DECAY,
                          correct_bias=True)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=training_steps)
        path = './model/model_dense_' + datetime.datetime.today().strftime('%Y_%m_%d')
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        model_save_name = path + '/_{}_fold_{}.bin'.format(model_type, fold)

        history, preds, outputs = train_fold(model_statement=model_type + str(fold),
                                             epochs=EPOCHS,
                                             model=model,
                                             device=device,
                                             train_dataloader=train_loader,
                                             val_dataloader=val_loader,
                                             test_dataloader=test_loader,
                                             loss_fn=loss_fn,
                                             optimizer=optimizer,
                                             scheduler=scheduler,
                                             model_save_name=model_save_name,
                                             n_train=len(train),
                                             n_val=len(val),
                                             single_model=False
                                             )

        history_list.append(history)
        oof_preds.append(preds)
        oof_outputs.append(outputs)
        oof_preds_indices.append(val_index)
        _, _, test_preds, test_outputs = eval_model(model,
                                                    device,
                                                    test_loader,
                                                    loss_fn,
                                                    len(test_df))
        test_preds_list.append(test_preds)
        test_outputs_list.append(test_outputs)

        fold += 1

    print(str(NFOLDS), 'Fold CV Train Time: ', format_time(time.time() - start_time))

    return history_list, test_outputs_list

####################################################################################################################


#
# #train dataset:0.9, test dataset:0.05, val dataset:0.05
# RANDOM_SEED = 177
# np.random.seed(RANDOM_SEED)
# torch.manual_seed(RANDOM_SEED)
# MAX_LENGTH = 64
# BATCH_SIZE = 16
# NUM_CLASSES = 3 # neutral, positive, negative
# EPOCHS = 5
# DROPOUT_PROB = 0.1
# WEIGHT_DECAY = 0.01
# NFOLDS = 10
# LEARNING_RATE = 2e-5
# #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #device =torch.device('cpu')
# loss_function = nn.CrossEntropyLoss().to(device)
# statement_df = pd.read_csv('./data NASDAQ/all-data.csv', encoding='latin-1',
#                            header=None)
# statement_df.columns = ['sentiment', 'statement']
# statement_df = statement_df.drop_duplicates()
# statement_df['statement'] = statement_df['statement'].apply(clean_statements)
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
#
# data_loader = create_dataloader(df_test, 16)
# total_model_record = pd.DataFrame(columns=["model", "acc", "loss"])
#
# dir = f"./model/model_dense_2021_09_15/"
# paths = os.listdir(dir)
# for path in paths:
#     if "roberta" in path:
#         model_type = "roberta"
#     elif "distilbert" in path:
#         model_type = "distilbert"
#     elif "bert-base-cased" in path:
#         model_type = "bert-base-cased"
#     elif "bert-base-uncased" in path:
#         model_type = "bert-base-uncased"
#     elif "bert-large-cased" in path:
#         model_type = "bert-large-cased"
#     elif "bert-large-uncased" in path:
#         model_type = "bert-large-uncased"
#     else:
#         print("wrong file", path)
#         continue
#     if not ".bin" in path:
#         continue
#     model_path = dir + "/" + path
#     model, tokenizer = load_model(model_type, model_path)
#     model = model.to(device)
#     result = eval_model(model, device, data_loader, loss_function, len(df_test))
#     acc, loss, __ = result
#     model_name = path.replace(".bin", "")
#     total_model_record = total_model_record.append({"model": model_name, "acc": acc, "loss": loss}, ignore_index=True)


# #
# path = './model/model_dense_' + datetime.datetime.today().strftime('%Y_%m_%d')
#
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
#
# bert_history, bert_test_outputs = get_oof_and_test_preds(model_type='bert-base-cased',
#                                                             tokenizer=bert_tokenizer,
#                                                             train_df=df_train_full,
#                                                             test_df=df_test,
#                                                             single_model=False,
#                                                             loss_fn=loss_function)


#
#
