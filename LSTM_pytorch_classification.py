import datetime
import time

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from torch import Tensor
from sklearn.metrics import r2_score
from torch.autograd import Variable


# https://github.com/hadi-gharibi/pytorch-lstm/blob/master/lstm.ipynb
# df[df.columns[~df.columns.isin(['C','D'])]]

def window_data(df, window, target_names):
    X = []
    y = []
    z = []
    for i in range(len(df) - window - 1):
        features = df[df.columns[~df.columns.isin(target_names)]][i:(i + window)]
        # target = df[target_names][i + window]
        target = df.loc[i + window - 1, target_names]
        X.append(features)
        y.append(target)
        z.append((torch.tensor(features.values), torch.tensor(target)))
    return z


def dataloading(keys, batch_size, targets, output_dim,df=None,filepath=None, window_size=7):
    if(filepath != None):
        df = pd.read_csv(filepath, infer_datetime_format=True,
                         parse_dates=True)

    df = df[keys]
    df = df.fillna(value=df.mean(axis=0))

    scaler = MinMaxScaler(feature_range=(0, 1))
    split = int(0.7 * len(df))
    df_scaled_train = scaler.fit_transform(df.iloc[:split, :].values)
    df_scaled_train = pd.DataFrame(df_scaled_train)
    df_scaled_test = scaler.fit_transform(df.iloc[split:, :].values)
    df_scaled_test = pd.DataFrame(df_scaled_test)

    input_dim = len(df_scaled_train.columns) - output_dim

    df_scaled_train.columns = keys
    df_scaled_test.columns = keys
    train_dataset = window_data(df_scaled_train, window_size, target_names=targets)
    test_dataset = window_data(df_scaled_test, window_size, target_names=targets)

    split = int(0.7 * len(train_dataset))
    val_dataset = train_dataset[split:]
    train_dataset = train_dataset[: split - 1]
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

    return input_dim, train_dataset, train_loader, val_dataset, val_loader, test_dataset, scaler


def dataloading_total(keys, batch_size, targets, output_dim,df=None,filepath=None, window_size=7):
    if(filepath != None):
        df = pd.read_csv(filepath, infer_datetime_format=True,
                         parse_dates=True)

    df = df[keys]
    df = df.fillna(value=df.mean(axis=0))

    scaler = MinMaxScaler(feature_range=(0, 1))

    df_scaled = scaler.fit_transform(df.values)
    df_scaled = pd.DataFrame(df_scaled)

    input_dim = len(df_scaled.columns) - output_dim

    df_scaled.columns = keys
    df_dataset = window_data(df_scaled, window_size, target_names=targets)


    return input_dim,df_dataset, scaler


class LSTMCell(nn.Module):
    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' cell.
    http://www.bioinf.jku.at/publications/older/2604.pdf

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.c2c = Tensor(hidden_size * 3)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        # pdb.set_trace()
        hx, cx = hidden
        x = x.view(-1, x.size(1))
        gates = self.x2h(x) + self.h2h(hx)
        gates = gates.squeeze()
        c2c = self.c2c.unsqueeze(0)
        ci, cf, co = c2c.chunk(3, 1)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate + ci * cx)
        forgetgate = torch.sigmoid(forgetgate + cf * cx)
        cellgate = forgetgate * cx + ingate * torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate + co * cellgate)
        hm = outgate * F.tanh(cellgate)
        return (hm, cellgate)


class LSTMModel(nn.Module):
    # if overfit. then add parameter dropout.
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, bias=True):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.layer_dim = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTMCell(input_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.relu=nn.relu()
        # self.fc_1 = nn.Linear(hidden_dim, 128)  # fully connected 1
        # # self.fc = nn.Linear(128, output_dim)  # fully connected last layer
        # self.relu = nn.ReLU()

    def forward(self, x):
        h0 = Tensor(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        c0 = Tensor(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        outs = []
        cn = c0[0, :, :]
        hn = h0[0, :, :]
        for seq in range(x.size(1)):
            hn, cn = self.lstm(x[:, seq, :].float(), (hn, cn))
            outs.append(hn)
        out = outs[-1].squeeze()
        # out = self.relu(hn)
        # out = self.fc_1(out)
        # out = self.relu(out)
        # out = self.fc(out)
        out = self.fc(out)

        return torch.sigmoid(out)


def LSTM(input_dim, train_loader, val_loader, hidden_dim, layer_dim, output_dim, num_epochs=50, seq_dim=7):
    criterion = nn.BCELoss()
    start_time = time.time()
    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=layer_dim, output_dim=output_dim)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    iter = 0
    Train_Loss = []
    Val_Loss = []
    for epoch in range(num_epochs):
        train_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            _, preds = torch.max(outputs, dim=1)
            labels = labels.float().clone().detach()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            iter += 1
        train_loss = train_loss / len(train_loader)
        Train_Loss.append(train_loss)
        if epoch != num_epochs:
            val_loss = 0
            # Iterate through test dataset
            for variables, labels in val_loader:
                with torch.no_grad():
                    #  outputs = model(variables)
                    predicted = model(variables)
                    # Get predictions from the maximum value
                    # _, predicted = torch.max(outputs.data, 1)
                    val_loss += criterion(predicted, labels).item()

            val_loss = val_loss / len(val_loader)

        Val_Loss.append(val_loss)
        return model, Train_Loss, Val_Loss




def LSTM_predict(dataloader, model):
    loss = 0
    # Iterate through test dataset
    criterion = nn.MSELoss()
    prediction = []
    targets = []
    model.eval()
    for variables, labels in dataloader:
        with torch.no_grad():
            predicted = model(variables)
        prediction.append(predicted.item())
        targets.append(labels.item())
    mse = criterion(Variable(torch.tensor(targets)),Variable(torch.tensor(prediction)))
    rmse = math.sqrt(loss)
    r = r2_score(prediction, targets)
    return prediction, targets, mse, r, rmse


from DataFun import plot_error, plot_data
import LSTM_pytorch

#==================================================================================
def train_LSTM(model_name, df, keys, batch_size, targets, output_dim, seq_dim):
    (input_dim, train_dataset, train_loader, val_dataset,
     val_loader, test_dataset, scaler) = LSTM_pytorch.dataloading(df=df,
                                                                  keys=keys,
                                                                  batch_size=batch_size,
                                                                  targets=targets,
                                                                  output_dim=output_dim,
                                                                  window_size=seq_dim)
    training_record = []
    train_l = 0
    val_l = 0
    for i in range(5):
        model, train_loss, val_loss = LSTM(input_dim=input_dim,
                                                        seq_dim=seq_dim,
                                                        train_loader=train_loader,
                                                        val_loader=val_loader,
                                                        hidden_dim=hidden_dim,
                                                        layer_dim=layer_dim,
                                                        output_dim=output_dim,
                                                        num_epochs=50)
        training_record.append([model, train_loss, val_loss])
        train_l = +train_loss[-1]
        val_l = +val_loss[-1]
    train_l = train_l / 5
    val_l = val_l / 5
    print(f"single layer LSTM train_l: {train_l}, val_l: {val_l}")
    return training_record, test_dataset


def test_LSTM(model_name, test_dataset, training_record, total_record):
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=True)
    prediction_record = pd.DataFrame(columns=["model_name", "model", "MSE", "RMSE", "R2"])

    # fig, axs = plt.subplots(5,2)
    # #plt.xlim(0, 0.8)
    # fig.set_size_inches(18.5, 14.5)
    for index, record in enumerate(training_record):
        model, _, _ = record
        predicted, actual, mse, r, rmse = LSTM_pytorch.LSTM_predict(test_loader, model)
        prediction_record = prediction_record.append({"model_name": model_name,
                                                      "model": index,
                                                      "MSE": mse.item(),
                                                      "RMSE": rmse, "R2": r}, ignore_index=True)

    total_record = total_record.append(prediction_record, ignore_index=True)

    return total_record, prediction_record


def total_performance_LSTM(model_name, df, total_model_record, keys, batch_size, targets, output_dim, seq_dim,
                           prediction_record, training_record):
    index_min_mse = min(range(len(prediction_record["MSE"])), key=prediction_record["MSE"].__getitem__)
    index_max_r = max(range(len(prediction_record["R2"])), key=prediction_record["R2"].__getitem__)
    modle_min_mse, _, _ = training_record[index_min_mse]
    model_max_r2, _, _ = training_record[index_max_r]
    scaler = MinMaxScaler()
    (input_dim, df_dataset, scaler) = LSTM_pytorch.dataloading_total(df=df,
                                                                     keys=keys,
                                                                     batch_size=batch_size,
                                                                     targets=targets,
                                                                     output_dim=output_dim,
                                                                     window_size=seq_dim)
    # LSTM(input_dim,train_loader, val_loader, hidden_dim, layer_dim, output_dim, seq_dim=7):
    data_loader = torch.utils.data.DataLoader(dataset=df_dataset,
                                              batch_size=1,
                                              shuffle=True)
    predicted, actual, mse, r, rmse = LSTM_pytorch.LSTM_predict(data_loader, modle_min_mse)
    print('MSE = {}'.format(mse))
    print('RMSE = {}'.format(rmse))
    print('R-Squared Score = {}'.format(r))

    plot_data(actual, predicted)

    score = "MSE"
    total_model_record = total_model_record.append(
        {"model": model_name, "score": score, "sentiment_method": "no sentiment",
         "MSE": mse.item(), "RMSE": rmse, "R2": r}, ignore_index=True)
    return total_model_record


# LSTM strucure:
hidden_dim = 10
layer_dim = 1 # ONLY CHANGE IS HERE FROM ONE LAYER TO N LAYER
output_dim = 1  # 2
seq_dim = 3
batch_size = 7

filepath = f"./data/BTC_Data.csv"
BTC_df = pd.read_csv(filepath, infer_datetime_format=True,
                         parse_dates=True)
BTC_df=BTC_df.fillna(0)
BTC_df=BTC_df[:-1]
keys=['High', 'Low', 'Open', 'Close', 'Volume', 'tweets',
      'google_trends','profitability','Marketcap','price','change']
targets = ["change"]

direction=f'./model/baseline'
sentiment_direction=f'./data/model_dense'
num_epochs = 50
model_name="single_LSTM-3"
df=BTC_df
training_record,test_dataset=train_LSTM(model_name,df,keys,batch_size,targets,output_dim,seq_dim)

# total_record,prediction_record=test_LSTM(model_name,test_dataset,training_record,total_record)
# model_record= total_performance_LSTM(model_name,df,total_model_record,keys,batch_size,targets,output_dim,seq_dim,prediction_record,training_record)
# total_model_record=model_record
# total_model_record