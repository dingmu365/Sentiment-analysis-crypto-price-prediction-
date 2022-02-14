import csv
import os

import numpy as np
import pysentiment2 as ps
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from dataclasses import make_dataclass


def LM_prediction(data,LM):
    result = pd.DataFrame(columns=["prediction"])
    for index, item in data.iterrows():
        token = LM.tokenize(item['statement'])
        score = LM.get_score(token)
        # positive-2, neutral-1,negative-0
        Point = make_dataclass("Point", [("prediction", int)])

        if score["Positive"] > score["Negative"]:
            prediction = 2
        elif score["Positive"] == score["Negative"]:
            prediction = 1
        else:
            prediction = 0

        result = result.append(pd.Series({"prediction": prediction}, name=index))
    return result



#
#
LM= ps.LM()

# path_read="./data/NASDAQ_headlines.csv"
# keys=["Date","Nid","statement","tags"]
# path_write= "./data/LM_sentiment_pres.csv"
# with open(path_read, 'r', encoding="utf_8_sig") as file:
#     i=0
#     file = csv.DictReader(file,fieldnames=keys)
#     for line in file:
#         token = LM.tokenize(line['statement'])
#         score = LM.get_score(token)
#         Point = make_dataclass("Point", [("date", str), ("ID", str), ("Positive",float),("Negative",float),("Polarity",float),('Subjectivity',float)])
#         statement = pd.DataFrame([Point(line["Date"],
#                                         line["Nid"],
#                                         score["Positive"],
#                                         score["Negative"],
#                                         score["Polarity"],
#                                         score["Subjectivity"])])
#         print(i)
#         i += 1
#         statement.to_csv(
#             path_write,
#             index=False,
#             header=False,
#             mode='a',
#             chunksize=1)

filename = f"./data/LM_sentiment_pres.csv"
data = pd.read_csv(filename)
data["sentiment"]=100000
data["sentiment"]=np.where(data["Positive"]>data["Negative"],1,data["sentiment"])
data["sentiment"]=np.where(data["Positive"]==data["Negative"],0,data["sentiment"])
data["sentiment"]=np.where(data["Positive"]<data["Negative"],-1,data["sentiment"])
data["Neutral"]=np.where(data["sentiment"]==0,1,0)
g = data.groupby(['Date'])
sentiment_agg = g.sum() / g.count()
write_path = f"./data/LM_dic_sentiment.csv"
data=sentiment_agg[["Negative","Neutral","Positive","sentiment"]]
data.to_csv(write_path,header=True)
# print(write_path)
