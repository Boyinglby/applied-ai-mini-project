from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

'''
Preprocess steps
'''
class Proprocessor:

    def __init__(self, categorical_columns, numeric_columns):
        self.categorical_feature_names = categorical_columns
        self.numeric_feature_names = numeric_columns

    def create(self):
        numeric_preprocessor = Pipeline(
            steps = [
                ("imputer_mean", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]
        )
        categorical_preprocessor = Pipeline(
            steps = [
                ("imputer_frequent", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("categorical", categorical_preprocessor, self.categorical_feature_names),
                ("numeric", numeric_preprocessor, self.numeric_feature_names)
            ]
        )

        return preprocessor

