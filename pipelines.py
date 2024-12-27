
from sklearn.cluster import KMeans, Birch, MeanShift
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from preprocessing import Proprocessor

'''
Pipelines that include steps of preprocessing, training, hyper-parameter tunning
and evaluation.
'''
CLASSIFIERS = {
    "KNN": KNeighborsClassifier(),
    "RandomForest": RandomForestClassifier(),
    "SVC": SVC(),
    "KMeans": KMeans(n_clusters=2),
    "Birch": Birch(n_clusters=2),
    "MeanShift": MeanShift()
}

class Pipelines:

    def __init__(self, categorical_columns, numeric_columns, classfier_name):
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.classifier_name = classfier_name
        self.classifier = self.__get_classifier()
    
    def create(self):
        preprocessor = Proprocessor(self.categorical_columns, self.numeric_columns)
        steps = [
            ("preprocessing", preprocessor.create()),
            ("classifier", self.classifier)
        ]
        pipe = Pipeline(steps)
        return pipe
    
    def __get_classifier(self):
        if self.classifier_name is None:
            raise Exception("classifier_name must be specified")
        try:
            return CLASSIFIERS[self.classifier_name]
        except KeyError as e:
            print(f"Invalid classifier_name: {self.classifier_name}, error: {e}")
            raise