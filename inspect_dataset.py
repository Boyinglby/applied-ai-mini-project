'''
Inspect dataset
'''

from dataset import Dataset
import numpy as np
from utils import seperate

def inspect(id):
    # fetch dataset
    dataset = Dataset(id)

    # data (as pandas dataframes)
    X = dataset.X
    y = dataset.y

    metadata = dataset.metadata

    # dataset name
    seperate("Dataset Name")
    print(f"{metadata.uci_id}: {metadata.name}")

    # metadata 
    seperate("Metaname")
    print(dataset.metadata)

    # variable information 
    seperate("Variable Information")
    print(dataset.variables)

    # X
    seperate("X")
    print(X)

    # print X dtypes
    seperate("X dtypes")
    print(X.dtypes)

    seperate("Numeric columns:")
    print(X.select_dtypes(include=np.number).columns.tolist())
    seperate("Non-Numeric columns:")
    print(X.select_dtypes(include=object).columns.tolist())

    # y
    seperate("y")
    print(y)

    # print y dtypes
    seperate("y dtypes")
    print(y.dtypes)

