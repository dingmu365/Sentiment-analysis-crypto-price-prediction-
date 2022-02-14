import csv

import numpy as np
import operator
import os
import pandas as pd


def sort_csv(filename):
    with open(filename) as fp:
        crdr = csv.reader(fp)
        # set key1 and key2 to column numbers of keys
        filedata = sorted(crdr, key=lambda row: (row[0]))
    return filedata


'''
dir = f'./data/predictions'
    paths=os.listdir(dir)
    for path in paths:
        filename=dir+"/"+path
        mylist = sort_csv(filename)
        write_path=path.replace(".csv","_sorted.csv")
        folder = os.path.exists(write_path)
        if not folder:
            os.makedirs(write_path)
        with open(dir+"/"+write_path,'w',newline='') as f:
            writer=csv.writer(f)
            writer.writerows(mylist)
            print(write_path)
        f.close()
'''



def sentiment(filename):

   #keys2=["date", "sentiment_pred", "sentiment_count"]
    data=pd.read_csv(filename,header=None)
    if len(data.columns)==5:
        keys = ["date", "id", "pred0", "pred1", "pred2"]
        data.columns=keys
        data["sentiment"] =0
        data["sentiment"] = np.where(((data["pred0"]>data["pred1"])&(data["pred0"]>data["pred2"])) ,0,
                                 np.where(((data["pred1"]>data["pred0"])&(data["pred1"]>data["pred2"])), 1, 2))

    else:
        keys = ["date", "id","sentiment", "pred0", "pred1", "pred2"]
        data.columns = keys
    data['preds'] = 0
    data["sentiment"] = data["sentiment"] - 1
    data["negative"] = 0
    data["neutral"] = 0
    data["positive"] = 0

    data["preds"]=np.where(data["sentiment"]==-1,data["pred0"],
                           np.where(data["sentiment"]==0,data["pred1"],
                                    data["pred2"]))

    data["negative"]=np.where(data["sentiment"]==-1,1,0)
    data["neutral"] = np.where(data["sentiment"]==0,1,0)
    data["positive"] = np.where(data["sentiment"]==1,1, 0)
    data["preds"]=data["preds"]*data["sentiment"]
    g = data.groupby(['date'])
    sentiment_agg=g.sum()/g.count()

    return sentiment_agg[["sentiment","preds","negative","neutral","positive"]]

def sentiment_indicator(dir):
    paths = os.listdir(dir)

    for path in paths:
        filename = dir + "/" + path
        mylist = sort_csv(filename)
        write_path = path.replace(".csv", "_sorted.csv")
        with open(dir + "/" + write_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(mylist)
            print(write_path)
        f.close()
    paths = os.listdir(dir)
    for path in paths:
        if "sorted" not in path:
            continue
        filename = dir + "/" + path
        mylist = sentiment(filename)
        write_path = path.replace("_sorted", "_sentiment")
        mylist.to_csv(dir + "/" + write_path, header=True)
        print(write_path)


if __name__ == '__main__':
    dir = f'./data/model_dense/roberta'
    sentiment_indicator(dir)