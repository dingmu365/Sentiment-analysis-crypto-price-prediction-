import pandas as pd
import numpy as np
import tkinter
# import statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic


# Use Augmented Dickey-Fuller(ADF)test
def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic': round(r[0], 4), 'pvalue': round(r[1], 4), 'n_lags': round(r[2], 4), 'n_obs': r[3]}
    p_value = output['pvalue']

    def adjust(val, length=6):
        return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-' * 47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key, val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")

BTC_df = pd.read_csv('./data/BTC_Data.csv', encoding='latin-1').iloc[:, 1:]
BTC_df = BTC_df.set_index(['Date'])
tweets_mean = BTC_df["tweets"].mean()
BTC_df["tweets"] = BTC_df["tweets"].fillna(tweets_mean)


# for name, column in BTC_df.iteritems():
#     adfuller_test(column, name=column.name)
#     print('\n')
#
# print("first stationary test done.")
# # Non-stationary: High,Low,Open,Close,Volume,Marketcap
# # stationary:tweets,google_trends,profitability,transaction_fee,change,fluction
# df_differenced = BTC_df.diff().dropna()
#
# for name, column in df_differenced.iteritems():
#     adfuller_test(column, name=column.name)
#     print('\n')

##Non-stationary:none
# stationary:High,Low,Open,Close,Volume,Marketcap,tweets,google_trends,profitability,transaction_fee,change,fluction
# keys = ['Close', 'Volume', 'tweets', 'google_trends', 'profitability', 'transaction_fee', 'fluction']
# targets = ["fluction"]
# #BTC_df = df_differenced[keys]#'change',
# split = int(0.7 * len(BTC_df))
# train_dataset = BTC_df[0: split - 1]
# test_dataset = BTC_df[split:]
# VAR_model = VAR(train_dataset)
# # to check which lag order is the best
# # maxlags = 10
# # val_model_lag_select=VAR_model.select_order(maxlags=maxlags)
# # print(val_model_lag_select.summary())
# #AIC: lag order 7: 106.1227558574435
# #BIC: lag order 2: 108.56648183096402
# #FPE: lag order 7: 1.2340789089904682e+46
# #HQIC: lag order 3: 107.5*
# #==> use lag order 2 and 7 as our VAR model
#
# VAR_model_fitted_2=VAR_model.fit(2)
# VAR_model_fitted_7=VAR_model.fit(7)

####forcast#######


def VAR_forecast(fitted_model,data,predict_steps=1):
    # empty list for our predictions
    prediction = []
    lag_order=fitted_model.k_ar
    # for loop to iterate fitted_model over data
    for i in range(lag_order, len(data)+1):
        # window of lagged data that the model uses to predict next observation
        window = data.iloc[i - lag_order: i].copy()
        # results of fitted_model being applied to window
        if(i==lag_order):
            results = fitted_model.forecast(y=window.values, steps=lag_order)
            prediction=list(results)
        else:
            results = fitted_model.forecast(y=window.values, steps=predict_steps)
            # append results to prediction list

            prediction.append(results)

    # convert prediction (which is a list of numpy arrays) to a dataframe
    df = np.vstack(prediction)
    df = pd.DataFrame(df)
    # df column names from data
    df.columns = list(data.columns)
    # df index from data
    df.index = data.iloc[len(data) - len(prediction):].index

    # return df
    return df

# result=VAR_forecast(VAR_model_fitted_7,test_dataset)
#
# flu_prediction=result["fluction"]
# flu_actual=test_dataset["fluction"]
# a=pd.concat([flu_prediction>0,flu_actual>0],axis=1)
# a.column=['col1','col1']