import pandas as pd
import ruamel.yaml
import json
import os
import math

from keybert import KeyBERT
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer


def extract_relevant_words(directory_in: str,
                           output: str,
                           file_format: str = "yaml",
                           model: str = "msmarco-distilbert-cos-v5",
                           percent: float = 0.20) -> pd.DataFrame:
    """
    Extracts relevant words from a dataset of OpenAPI documents (YAML or JSON)
    :param directory_in: Name of the directory of the dataset
    :param output: Name of the output file (ends with .csv)
    :param file_format: File format of the OpenAPI documents (YAML or JSON)
    :param model: Sentence-transformer model to use with KeyBERT
    :param percent: Percent of words to extract
    :return: DataFrame containing the extracted relevant words
    """
    with open("ignored_terms.txt", "r", encoding="utf-8") as file:
        stops = file.read().split("\n")
    all_stops = list(text.ENGLISH_STOP_WORDS.union(stops))

    if file_format == "yaml":
        parser = ruamel.yaml.YAML(typ="safe")
    elif file_format == "json":
        parser = json

    keybert = KeyBERT(model=model)
    count = CountVectorizer(stop_words=all_stops)

    res = {
        "title": [],
        "labels": [],
        "keywords": [],
        "gpt": []
    }

    for file in os.scandir(directory_in):
        try:
            with open(file.path, "r", encoding="utf-8") as f:
                data = parser.load(f)
                res["title"].append(data["info"]["title"])
                res["labels"].append(data["info"].pop("x-apisguru-categories", ""))

                str_data = str(data)
                kw_n = math.ceil(percent * len(count.fit([str_data]).get_feature_names_out()))
                keywords = keybert.extract_keywords(str_data, top_n=kw_n, stop_words=all_stops)
                res["keywords"].append(','.join([word[0] for word in keywords]))
        except Exception as e:
            print(e)

    df = pd.DataFrame(res)
    df.to_csv(output, index=False, sep=";")
    return pd.DataFrame(res)
