'''
Experiment module
'''
from sklearn.model_selection import KFold

from dataset import Dataset
from evaluation import avg_df, evaluate_scores, rank_models, save_df, save_result
from pipelines import Pipelines
from utils import seperate
from visualization import plot_dataset_score

PATH = "results"
DEMO_PATH = "results/demo"

def run(type, dataset_configs, metrics_config, classifiers_config, demo=False):
    path = PATH if not demo else DEMO_PATH
    show = True if demo else False
    # Train and evaluate models on all datasets
    seperate(f"Type: {type} learning")
    classifiers = classifiers_config[type]
    result = train_predict(dataset_configs, type, classifiers)

    # Save the results for each dataset and metric
    for metric in metrics_config[type]:
        save_result(result, path, metric)
        plot_dataset_score(result, metric, path, show=show)
    
    # Calculate average scores on all datasets
    average = avg_df(result.values())
    seperate("Average scores on all datasets")
    print(average)
    save_df(average, path, f"{type}_average")

    # Rank models based on each metric
    for metric in metrics_config[type]:
        rank = rank_models(average, metric=metric)
        seperate(f"Model ranks based on {metric}")
        print(rank)
        save_df(rank, path, f"rank_{metric}")
    return result

def train_predict(dataset_configs, type, classifiers):
    seperate(f"List of datasets: {dataset_configs}")
    dataset_scores = {}
    for config in dataset_configs:
        id = config["id"]
        dataset_name = config["name"]
        target = config.get("target", None)

        # load dataset
        seperate(f"Dataset: {dataset_name}")
        dataset = Dataset(id, target=target)
        X = dataset.X
        y = dataset.y_target
        categorical_columns = dataset.categorical_columns
        numeric_columns = dataset.numeric_columns

        # use first of unique labels in the target column
        pos_label = y.unique()[0]

        # create pipeline for each classifier
        pipelines = create_pipelines(categorical_columns, numeric_columns, classifiers)
        
        # KFold cross-validation
        cv_scores = cross_validation(type, X, y, pos_label, pipelines)

        # calculate average score for the KFold cross-validation
        avg_score = avg_df(cv_scores)
        dataset_scores[dataset_name] = avg_score
        seperate(f"Average score for dataset {dataset_name}")
        print(avg_score)
    return dataset_scores

def create_pipelines(categorical_columns, numeric_columns, classifiers):
    pipelines = {}
    for name in classifiers:
        pipeline = Pipelines(categorical_columns, numeric_columns, classfier_name=name).create()
        pipelines[name] = pipeline
    return pipelines

def cross_validation(type, X, y, pos_label, pipelines):
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        models = train_pipelines(X_train, y_train, pipelines)
        score_fold = evaluate_scores(X_test, y_test, models, pos_label=pos_label, type=type)
        scores.append(score_fold)
    return scores

def train_pipelines(X_train, y_train, pipelines):
    models = {}
    for name, pipe in pipelines.items():
        model = pipe.fit(X_train, y_train)
        models[name] = model
    return models


