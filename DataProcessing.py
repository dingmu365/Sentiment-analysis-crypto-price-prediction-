import pandas as pd
import nltk.data
import json
import csv
import re
from pygrok import Grok
import datetime
from dataclasses import make_dataclass

dictfilt = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])


def clean_statements(statement):
    statement = re.sub(" '", "'", statement)
    statement = re.sub(" 's", "'s", statement)
    statement = re.sub('\( ', '(', statement)
    statement = re.sub(' \)', ')', statement)
    statement = re.sub('``', '"', statement)
    statement = re.sub("''", '"', statement)
    statement = re.sub(r'\s([?.,%:!"](?:\s|$))', r'\1', statement)
    return statement

def ConvertJson2Csv(path_read,path_write,keys,dictfilt=dictfilt):
    csv_file = open(path_write,'w', newline='',encoding="utf_8_sig")
    writer = csv.DictWriter(csv_file, fieldnames=keys)
    writer.writeheader()
    #Z=False
    with open(path_read,'r') as file:
        for line in file:
            line = json.loads(line)
            if("news_tag" in line):
                None
            else:
                line["news_tag"]=None

            if("news_time" in line and "_id" in line and 'news_title' in line ):
                if("$date" in line["news_time"]):
                                   line["news_time"]=line["news_time"]["$date"]

                writer.writerow(dictfilt(line, keys))

    file.close()
    csv_file.close()
    print("convert finished")

# keys = ['_id', 'news_title', 'news_time', 'news_tag']
# read_path=r"./data NASDAQ/NASDAQ_headlines.json"
# write_path=r"./data NASDAQ/NASDAQ_headlines_next.csv"
# ConvertJson2Csv(read_path ,write_path, keys)
path=r'./data/Headline_Trainingdata.json'


def TitleSpilter(input_path, file_path,keys):
   # tokenizers = nltk.data.load('tokenizers/punkt/english.pickle')
    res = False
    date_pattern = "%{YEAR:year}-%{MONTHNUM:month}-%{MONTHDAY:day}"
    grok = Grok(date_pattern)
    i = 0
    tags = ["Inversting", "Stocks", "Personal Finance"]

    with open(input_path, 'r') as files:
            file = csv.DictReader((l.replace('\0', '') for l in files), fieldnames=keys)
                #
            for row in file:
                    if i == 0:
                        i += 1
                        print("begin")
                        continue
                    else:
                        print(i)
                        i += 1
                        obj = json.loads(row["_id"].replace("'", "\""))
                        date = row["news_time"].replace("'", "\"")
                        date = date.replace("T", " ")
                        date = date.replace("Z", " ")

                        date = date.replace("'", "\"")
                        date = grok.match(date)
                        try:
                            date = datetime.date(year=int(date['year']), month=int(date['month']), day=int(date['day'])).isoformat()
                        except ValueError:
                            continue
                        except TypeError:
                            continue

                        Point = make_dataclass("Point", [("date",str), ("ID",str),('title',str),('news_tag',str)])
                        statement=pd.DataFrame([Point(date,obj['$oid'],row['news_title'],row['news_tag'])])

                        statement.to_csv(
                            file_path,
                            index=False,
                            header=False,
                            mode='a',
                            chunksize=1)

    files.close()
    res=True
    return (res)
# tags=["Inversting","Stocks","Personal Finance"]
# keys = ['_id', 'news_title', 'news_time', 'news_tag']
# print(TitleSpilter(f'./data NASDAQ/NASDAQ_headlines.csv',
#                    f'./data/NASDAQ_headlines_final.csv',
#                    keys))

def TagFilter(read_path,write_path,tags,keys):
    i=0
    with open(read_path,'r',encoding='utf_8_sig') as files:
        file=csv.DictReader(files,fieldnames=keys)
        for line in file:
            if any(x in line["news_tag"] for x in tags):
                None
            else:
                continue
            Point = make_dataclass("Point", [("date", str), ("ID", str), ('title', str), ('news_tag', str)])
            statement = pd.DataFrame([Point(line["Date"],line["id"],line["Title"],line["news_tag"])])
            print(i)
            i += 1
            statement.to_csv(
                write_path,
                index=False,
                header=False,
                mode='a',
                chunksize=1)
    files.close()
    print("done!")
# tags=["Inversting","Stocks","Personal Finance"]
# keys = ['Date', 'id', 'Title', 'news_tag']
# read_path=f"./data/NASDAQ_headlines.csv"
# write_path=f"./data/NASDAQ_headlines_filtered.csv"
# TagFilter(read_path,write_path,tags,keys)


def NewsSpilter(input_path, file_path):
    tokenizers = nltk.data.load('tokenizers/punkt/english.pickle')
    res = None
    date_pattern = "%{YEAR:year}-%{MONTHNUM:month}-%{MONTHDAY:day}"
    grok = Grok(date_pattern)
    i = 0
    keys = ['_id', 'article_link', 'article_title', 'article_time', 'author_name', 'author_link',
            'article_content', 'appears_in', 'symbols', 'related_articles', 'related_articles']

    with open(input_path, 'r') as contents:
        contents = csv.DictReader(contents, fieldnames=keys)
        for row in contents:
            if i == 0:
                i += 1
                continue

            else:
                obj = json.loads(row["_id"].replace("'", "\""))
                statement = pd.DataFrame(tokenizers.tokenize(row["article_content"]))
                # statement=statement[0].lower()
                # statement=re.sub('[^a-zA-z\s]', '',statement) # remove 0-9
                date = json.loads(row["article_time"].replace("'", "\""))
                date = grok.match(date["$date"])

                date = datetime.date(year=int(date['year']), month=int(date['month']), day=int(date['day'])).isoformat()
                statement['date'] = date
                statement['ID'] = obj["$oid"]
                statement['sentences'] = statement[0]
                statement.to_csv(  # file_path,
                    f"./data NASDAQ/exp.csv",
                    index=False,
                    header=True,
                    mode='a',
                    chunksize=len(statement),
                    encoding='uft-8')
    contents.close()

    return (res)
    
def Convert_into_timeinterval(data, TimeIntervalDown, TimeIntervalUp):
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data['date'] = data['Date'].dt.date

    data = data.set_index('date')
    data = data[pd.to_datetime(TimeIntervalDown, format='%Y-%m-%d'):
                pd.to_datetime(TimeIntervalUp, format='%Y-%m-%d')]
    return data



# def convert_binary:

def convert_fluction(up_down):
    up_down['change'] = None
    up_down['fluction'] = None
    up_down['price'] = None
    up_down['Date']=up_down.index
    up_down.index=range(len(up_down))
    n= len(up_down)
    for idx in up_down.index:
        if idx ==n-1:
            up_down["change"][idx] = 0
            up_down["fluction"][idx] = 0
            up_down["price"][idx] = 0

        else:
            up_down["price"][idx] =up_down["Close"][idx + 1]
            x=up_down["Close"][idx+1]-up_down["Close"][idx]
            #x = up_down.loc[idx, "Close"]-up_down.loc[idx-1, "Close"]
                #up_down["Close"][idx] - up_down["Close"][idx - 1]
            # 0 is increading. 0 is decreasing
            if x > 0:
                up_down["change"][idx] = 1
                up_down.loc[idx, "fluction"] = abs(round(x / up_down["Close"][idx], 5))

            else:
                up_down["change"][idx] = 0
                up_down.loc[idx, "fluction"] = abs(round(x / up_down["Close"][idx], 5)) * -1

            print(up_down.loc[idx, "fluction"])

    print("finished")
    return up_down

##################### combine different table data############################

##combine all the data together
## use price fluaction as target
## target1: if the price increase then 1, otherwise 0
## target2: how many will it change in precent
# begin= '2018-03-12'
# end='2020-12-14'
# BTC_df = pd.read_csv('./data NASDAQ/coin_Bitcoin.csv').dropna()
# BTC_df = Convert_into_timeinterval(BTC_df, begin, end)
# BTC_df = BTC_df[["High", "Low", "Open", "Close", "Volume", "Marketcap"]]
#
# ETH_df = pd.read_csv('./data NASDAQ/coin_Ethereum.csv').dropna()
# ETH_df = Convert_into_timeinterval(ETH_df, begin, end)
# ETH_df = ETH_df[["High", "Low", "Open", "Close", "Volume", "Marketcap"]]
#
# tweets = pd.read_csv('./data NASDAQ/tweets.csv')
# tweets = Convert_into_timeinterval(tweets, begin, end)
#
# google_trends = pd.read_csv('./data NASDAQ/google_trend.csv')
# google_trends['Date'] = google_trends['Date'].map(
#     lambda x: datetime.datetime.strptime(x, '%m/%d/%Y').strftime('%Y-%m-%d %I:%M:%S'))
# google_trends = Convert_into_timeinterval(google_trends, begin, end)
#
# profitability = pd.read_csv('./data NASDAQ/profitability.csv')
# profitability = Convert_into_timeinterval(profitability, begin, end)
#
# transaction_fee = pd.read_csv('./data NASDAQ/profitability.csv')
# transaction_fee = Convert_into_timeinterval(transaction_fee, begin, end)
# BTC_df = pd.concat([BTC_df, tweets.Bitcoin, google_trends.Bitcoin, profitability.Bitcoin, transaction_fee.Bitcoin],
#                    axis=1)
# BTC_df.columns = ["High", "Low", "Open", "Close", "Volume", "Marketcap", 'tweets', 'google_trends', 'profitability',
#                   'transaction_fee']
# ETH_df = pd.concat([ETH_df, tweets.Ethereum, google_trends.Ethereum, profitability.Ethereum, transaction_fee.Ethereum],
#                    axis=1)
# ETH_df.columns = ["High", "Low", "Open", "Close", "Volume", 'tweets', "Marketcap", 'google_trends', 'profitability',
#                   'transaction_fee']
#
# #BTC_df=pd.read_csv(f'./data/BTC_data.csv')
# BTC_df=BTC_df[["High", "Low", "Open", "Close", "Volume", 'tweets', "Marketcap", 'google_trends', 'profitability',
#                   'transaction_fee']]
# BTC_df=convert_fluction(BTC_df)
#
# ETH_df=ETH_df[["High", "Low", "Open", "Close", "Volume", 'tweets', "Marketcap", 'google_trends', 'profitability',
#                   'transaction_fee']]
# ETH_df = convert_fluction(ETH_df)
#
# #
# BTC_df.to_csv("./data/BTC_Data.csv")
# ETH_df.to_csv("./data/ETH_Data.csv")


