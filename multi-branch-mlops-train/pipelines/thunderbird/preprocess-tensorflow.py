"""Feature engineers the thunderbird dataset."""
import argparse
import logging
import os
import pathlib

import boto3
import numpy as np
import pandas as pd
from keras.utils import pad_sequences
import transformers
import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# PRE_TRAINED_MODEL_NAME = "roberta-base"

# tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/thunderbird_up_to_01012006.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    ### ab hier ausführen
    logger.debug("Reading downloaded data.")
    df = pd.read_csv(fn)
    # df = pd.read_csv('C:/Users/SEPA/Lighthouse_Projekt/thesis_katie/backend/old_data.csv')
    os.unlink(fn)

    logger.debug("Defining transformers.")
    
    def func_tokenizer(tokenizer_name, input_text):
        features = []
        for input_text in tqdm.tqdm(input_text, desc = 'converting titles to features'):
            tokens = tokenizer_name.tokenize(input_text)
            ids = tokenizer_name.convert_tokens_to_ids(tokens)
            features.append(ids)
        return features

    # X, y = news['text'], news['target']
    X, y = df['Title'], df['Component']
    lst = list(y)
    d = {x: i for i, x in enumerate(set(lst))}
    y = [d[x] for x in lst]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    roberta_tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base-openai-detector')

    roberta_train_features = func_tokenizer(roberta_tokenizer, X) # sind eigentl. nur die 

    roberta_trg = pad_sequences(roberta_train_features, maxlen = 50)
    #y_train = y
    # X = roberta_trg, y= y_train

    #text_features = ["Title", "Description"]
    #text_transformer = Pipeline(
        #steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    #)

    #categorical_features = ["Component"]
    #categorical_transformer = Pipeline(
     #   steps=[
     #       ("onehot", OneHotEncoder(handle_unknown="ignore")),
     #   ]
    #)

    #preprocess = ColumnTransformer(
     #   transformers=[
     #       ("txt", text_transformer, text_features),
            #("cat", categorical_transformer, categorical_features),
     #   ]
    # )

    logger.info("Applying transforms.")
    # y = df.pop("rings")
    # mit preprocess function, welche als ColumnTransformer definiert wurde
    # X_pre = preprocess.fit_transform(df)
    y = np.array(y)
    y_pre = y.reshape(len(y), 1)
    X_pre = roberta_trg
    X = np.concatenate((y_pre, X_pre), axis=1)

    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
