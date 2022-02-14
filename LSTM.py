import numpy as np
np.random.seed(177)

import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from math import sqrt
import datetime as dt
import time
plt.style.use('ggplot')

def window_data(df, window=7, target_names=['fluction']):
    X = []
    y = []
    z = []
    for i in range(len(df) - window - 1):
        features = df[df.columns[~df.columns.isin(target_names)]][i:(i + window)]
        # target = df[target_names][i + window]
        target = df.loc[i + window, target_names]
        X.append(features)
        y.append(target)
        z.append((features.values), (target))
    return X,y

# Build and train the model
def fit_model(train, val, timesteps, hl, learning_rate, batch, epochs):
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    # Extracting the series
    # Setting up an early stop
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=80, verbose=1, mode='min')
    callbacks_list = [earlystop]
    # Loop for training data
    # for i in range(timesteps, train.shape[0]):
    #     X_train.append(train[i - timesteps:i][:-1])
    #     Y_train.append(train[i][-1])
    # X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_train, Y_train=window_data(train,window=timesteps)
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    # Loop for val data
    # for i in range(timesteps, val.shape[0]):
    #     X_val.append(val[i - timesteps:i][:-1])
    #     Y_val.append(val[i][-1])
    # X_val, Y_val = np.array(X_val), np.array(Y_val)
    X_val, Y_val=window_data(val)
    X_val, Y_val = np.array(X_val), np.array(Y_val)
    # Adding Layers to the model
    model = Sequential()
    model.add(LSTM(X_train.shape[2], input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True,
                   activation='relu'))
    for i in range(len(hl) - 1):
        model.add(LSTM(hl[i], activation='relu', return_sequences=True))
    model.add(LSTM(hl[-1], activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss='mean_squared_error')
    # print(model.summary())

    # Training the data
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch, validation_data=(X_val, Y_val), verbose=0,
                        shuffle=False, callbacks=callbacks_list)
    model.reset_states()
    return model, history.history['loss'], history.history['val_loss']


# Evaluating the model
def evaluate_model(model, test, timesteps):
    X_test = []
    Y_test = []

    # Loop for testing data
    for i in range(timesteps, test.shape[0]):
        X_test.append(test[i - timesteps:i][:-1])
        Y_test.append(test[i][-1])
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    # print(X_test.shape,Y_test.shape)

    # Prediction Time !!!!
    Y_hat = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_hat)
    rmse = sqrt(mse)
    r = r2_score(Y_test, Y_hat)
    return mse, rmse, r, Y_test, Y_hat

# Plotting the predictions
def plot_data(Y_test,Y_hat):
    plt.plot(Y_test,c = 'r')
    plt.plot(Y_hat,c = 'y')
    plt.xlabel('Day')
    plt.ylabel('Fluctuation')
    plt.title('Price Fluctuation Prediction Graph using Multivariate-LSTM model')
    plt.legend(['Actual','Predicted'],loc = 'lower right')
    plt.show()

# Plotting the training errors
def plot_error(train_loss,val_loss):
    plt.plot(train_loss,c = 'r')
    plt.plot(val_loss,c = 'b')
    plt.ylabel('Loss')
    plt.legend(['train','val'],loc = 'upper right')
    plt.show()


filepath = f"./data/BTC_Data.csv"
keys = ['High', 'Low', 'Open', 'Close', 'Volume', 'tweets','Marketcap', 'google_trends', 'profitability', 'transaction_fee','fluction']
targets = ["fluction"]
BTC_df = pd.read_csv(filepath, infer_datetime_format=True,
                         parse_dates=True)
BTC_df=BTC_df.fillna(value=BTC_df.mean(axis=0,skipna=True))

BTC_df=BTC_df.set_index("Date")
keys=['Close', 'Volume', 'tweets', 'google_trends', 'profitability', 'transaction_fee', 'fluction']
series = BTC_df[keys]
# Picking the series with high correlation
print(series.shape)
series.tail(2)

timesteps = 7
hl = [40,35]
learning_rate = 1e-4
batch_size = 4
num_epochs = 10
# Train Val Test Split
split1=int(len(BTC_df)*0.8)
split2=int(len(BTC_df)*0.9)
train_data = series[:split1]
val_data = series[split1:split2]
test_data = series[split2:]
#Normalisation
scaler = MinMaxScaler()
BTC_train = pd.DataFrame(scaler.fit_transform(train_data),columns=keys)
BTC_val = scaler.transform(val_data)
BTC_test = scaler.transform(test_data)
print(BTC_train.shape,BTC_val.shape,BTC_test.shape)
BTC_model,train_error,val_error = fit_model(BTC_train,BTC_val,timesteps,hl,learning_rate,batch_size,num_epochs)
plot_error(train_error,val_error)