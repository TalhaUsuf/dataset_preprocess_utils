import pandas as pd
from rich.console import Console
from skmultilearn.model_selection import iterative_train_test_split
import seaborn as sns
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np




p = ArgumentParser(description="Split a dataset into histogram balanced train / test split")
p.add_argument("-c", "--csv", type=str, help="csv file containing label, image and identity columns", required=True)
p.add_argument("-t", "--train", type=str, help="train csv file path to save", required=True)
p.add_argument("-v", "--valid", type=str, help="valid csv file path to save", required=True)
args = p.parse_args()


def main():
    df = pd.read_csv(args.csv, skipinitialspace=True)
    one_hot_classes = pd.get_dummies(df["label"], sparse=True)
    df.drop("label", axis=1, inplace=True)
    df = pd.concat([df, one_hot_classes], axis=1)
    df.to_csv("one_hot_encoded.csv", index=False)
    X = df[["image", "identity"]]
    Y = df.iloc[:, 2:]
    x_train, y_train, x_test, y_test = iterative_train_test_split(X.values, Y.values, test_size = 0.2)
    
    y_train = np.argmax(y_train, axis=1).squeeze()
    y_test = np.argmax(y_test, axis=1).squeeze()
    
    train = {}
    train["image"] = x_train[:, 0]
    train["identity"] = x_train[:, 1]
    train["label"] = y_train
    
    
    test = {}
    test["image"] = x_test[:, 0]
    test["identity"] = x_test[:, 1]
    test["label"] = y_test
    
    train = pd.DataFrame.from_dict(train)
    test = pd.DataFrame.from_dict(test)
    
    common = set(train["identity"].values).intersection(set(test["identity"].values))
    assert len(common) == len(set(train["identity"].values)) and len(common) == len(set(test["identity"].values)), "Dataset identities not balanced"
    
    train.to_csv(args.train, index=False)
    test.to_csv(args.valid, index=False)
    
    
    sns.set_style("darkgrid", {"patch.edgecolor": 'None'})
    f, ax = plt.subplots(1, 2)
    plt.tight_layout()
    f.set_size_inches(40, 15)
    p1=sns.countplot(data=train, x="identity", ax=ax[0])
    p1.set_xticklabels(p1.get_xticklabels(), rotation=90)
    
    p2=sns.countplot(data=test, x="identity", ax=ax[1])
    p2.set_xticklabels(p2.get_xticklabels(), rotation=90)
    
    
    plt.savefig("histogram.png")
    # plt.show()
    
    Console().rule(title=f'[bold cyan]DONE [bold green]{train.shape} [yellow]{test.shape}', characters='-', style='bold yellow')
if __name__ == '__main__':
    main()