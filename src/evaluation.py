
import pandas as pd
from utils import clean_dataframe
from classifier import Aspect_Classifier
import os
from os.path import join


def compute_results():
    pipeline = Aspect_Classifier()
    df = pd.read_csv(join(*[os.getcwd(), os.pardir, "data", "test.csv"]))
    df_clean = clean_dataframe(df, "text", "aspect")
    submission = []
    for index, row in df_clean.iterrows():
        output = pipeline(row["text"], row["aspect"])
        submission.append(output["label"])

    df["target"] = submission
    df.to_csv(join(*[os.getcwd(), os.pardir, "data",
              "results", "submission.csv"]), index=False)


if __name__ == "__main__":
    compute_results()
