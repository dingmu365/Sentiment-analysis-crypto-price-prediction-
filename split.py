import torch
from dataclasses import make_dataclass
from BERTFamily_v2 import *

RANDOM_SEED = 177
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
MAX_LENGTH = 64
BATCH_SIZE = 16
BATCH_SIZE = 16
NUM_CLASSES = 3  # neutral, positive, negative
EPOCHS = 5
DROPOUT_PROB = 0.1
WEIGHT_DECAY = 0.01
NFOLDS = 10
LEARNING_RATE = 2e-5
le = LabelEncoder()
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_SEED)


##variables##
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device=torch.device('cpu')

def model_name(string):
    string = string.replace("_fold_", "")
    string = string.replace("_", "")
    string = re.sub(r'\d+', "", string, count=0, flags=0)
    return string


dir = f"./model/model_dense"
keys = ["Date", "ID", "headline", "tags"]
statement = "headline"

data_path = f"./data/NASDAQ_headlines.csv"

paths = os.listdir(dir)

Point = make_dataclass("Point", [("date", str), ("ID", str), ('sentiment', int), ('prob_0', float), ('prob1', float),
                                 ('prob2', float)])
###################
'''

'''

for path in paths:
    if ".bin" not in path:
        continue
    # elif "roberta-base" in path:
    #     model_type = "roberta-base"
    elif "roberta-large" in path:
        model_type = "roberta-large"
    # elif "distilbert" in path:
    #     model_type = "distilbert"
    # elif "bert-base-cased" in path:
    #     model_type = "bert-base-cased"
    # elif "bert-base-uncased" in path:
    #     model_type = "bert-base-uncased"
    # elif "bert-large-cased" in path:
    #     model_type = "bert-large-cased"
    # elif "bert-large-uncased" in path:
    #     model_type = "bert-large-uncased"
    else:
        # print("wrong file", path)
        continue


    model_path = dir + "/" + path
    model, tokenizer = load_model(model_type, model_path)
    model = model.eval()
    model = model.to(device)
    file_path = f"./data/model_dense/" + model_name(path.replace(".bin", ".csv"))
    n=0
    files = pd.read_csv(data_path, chunksize=2000, header=None)
    print(model_name(path.replace(".bin", ".csv")))
    for file in files:
        file.columns = keys
        predictions = pred_model(input_data=file[statement],
                                 model=model,
                                 tokenizer=tokenizer)
        n+=1
        print(n)

        # statements=pd.concat([file.reset_index()[['Date','ID']],predictions],1)
        statements = pd.concat([file.reset_index()[['Date', 'ID']], pd.DataFrame(predictions[1]).astype(float)], 1)
        statements.to_csv(
            file_path,
            index=False,
            header=False,
            mode='a',
            chunksize=file.__len__())
        del statements
    print("finished", model_name(path.replace(".bin", ".csv")))
