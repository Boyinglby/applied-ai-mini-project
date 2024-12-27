import numpy as np
from ucimlrepo import fetch_ucirepo

class Dataset:

    def __init__(self, id, target=None):
        self.id = id
        self.target = target
        self.load()

    def load(self):
        result = fetch_ucirepo(id=self.id)
        # data (as pandas dataframes)
        self.X = result.data.features
        self.y = result.data.targets
        
        self.metadata = result.metadata
        self.variables = result.variables
        self.target_cols = result.metadata["target_col"]
        self.y_target = self.__get_target()
        self.target_labels = self.__targe_labels()
        self.categorical_columns = self.__categorical_columns()
        self.numeric_columns = self.__numeric_columns()
    
    def __targe_labels(self):
        return self.y_target.unique()

    def __numeric_columns(self):
        return self.X.select_dtypes(include=np.number).columns.tolist()

    def __categorical_columns(self):
        return self.X.select_dtypes(include=object).columns.tolist()

    def __get_target(self):
        if self.target is None:
            self.target = self.target_cols[0]
        return self.y[self.target]
