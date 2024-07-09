import time

import pandas as pd

from prompt_construction import create_prompt
from query_classifier import query_gpt


def main():
    dataset = pd.read_csv("../relevant_words/apisguru.csv", sep=";")
    target_labels = dataset.labels.unique().tolist()

    query_multiple(dataset, target_labels=target_labels)

    dataset.to_csv("../results/apisguru/gpt-35.csv", index=False, sep=";")


def query_multiple(dataset: pd.DataFrame, target_labels: list[str], model: str = "gpt-3.5-turbo", timer: int = 3):
    """
    Queries the classifier multiple times using the provided dataset
    :param dataset: Dataframe of the dataset to be queried
    :param target_labels: Labels used as choices for the classifier
    :param model: GPT model to use
    :param timer: Time to wait between requests (in seconds)
    :return: None
    """
    for index, _ in dataset.iterrows():
        prompt = create_prompt(dataset.loc[index, "keywords"].split(','), target_labels)
        for _ in range(4):
            try:
                response = query_gpt(prompt, model=model)
                dataset.loc[index, "gpt"] = response
                time.sleep(timer)
            except Exception as e:
                print(e)
                time.sleep(timer)
                continue
            break


if __name__ == "__main__":
    main()
