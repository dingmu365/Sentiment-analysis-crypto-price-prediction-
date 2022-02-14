import DataProcessing
import numpy as np
import pandas as pd

## TODO: coding BERT,DistilBERT and RoBERTa for senteiment analysis of statements
##          using financial_statement_sentiment_analysis
##          bilt LSTM for price predictionW



"""
## programming logic
1. split the article to a new database with a_id and date
2. prediction BERT sentiment base on sentence and aggregate result for article 
3. predict sentiment base on lexicon model
4. create daily sentiment indicate base on mean(top5 max+ top5 min) 
5. put all these sentiment into LSTM
6. put all these sentiment into VAR

content_sentence=[]
for article in database:
    
    sentence=DataFrame(spliter(article))
    df=pd.DataFrame()
    contence_sentence.append(sentence)
pre_sentiment=[]
for a_id in database:
    if len(score)>10:
        agg_score=avrage(max(score,5)+min(score,5))
    else:
        agg_score=mean(agg_score)
pre_sentiment.append(agg_score)

# after get all the article sentiment
# aggregate them into daily basis
for day in news:
    sentiment_d_i= pre_sentiment(where pre_sentiment.day==day)
"""
## Data Processing
# c_filepath = f""
# df = pd.read_csv(c_filepath)
# df.head()
#
# df.isnull().sum().sum()
#

## model loading



## sentiment prediction
# content=pd.read_csv('./resample_NASDAQ_News.csv')
# result=DataProcessing.NewsSpilter()
# result.to_csv("./covert_sample.csv")
# print(result)

