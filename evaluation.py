'''
Evaluate the performance of the model
'''
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    f1_score,
    homogeneity_completeness_v_measure,
    precision_score,
    rand_score,
    recall_score,
)

def evaluate_scores(X_test, y_test, models, pos_label=1, type="supervised"):
    if type == "supervised":
        return evaluate_supervised_scores(X_test, y_test, models, pos_label)
    else:
        return evaluate_unsupervised_scores(X_test, y_test, models)

def evaluate_supervised_scores(X_test, y_test, models, pos_label):
    '''
    Create a dataframe of scores for each model
    Scores include: Accuracy, Precision, Recall, F1
    '''
    scores = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accquracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=pos_label)
        recall = recall_score(y_test, y_pred, pos_label=pos_label)
        f1 = f1_score(y_test, y_pred, pos_label=pos_label)
        scores[name] = [accquracy, precision, recall, f1]
    return pd.DataFrame(scores, index=["Accuracy", "Precision", "Recall", "F1"])

def evaluate_unsupervised_scores(X_test, y_test, models):
    '''
    Create a dataframe of scores for each model
    Scores include: Rand
    '''
    scores = {}
    for name, model in models.items():
        if name == "DBSCAN":
            y_pred = model.fit_predict(X_test)
        else:
            y_pred = model.predict(X_test)
        rand = rand_score(y_test, y_pred)
        ami = adjusted_mutual_info_score(y_test, y_pred)
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_test, y_pred)
        # silhouette = silhouette_score(X_test, y_pred)
        scores[name] = [rand, ami, homogeneity, completeness, v_measure]
    return pd.DataFrame(scores, index=["Rand", "AMI", "Homogeneity", "Completeness", "V Measure"])

def avg_df(df_list):
    return pd.concat(df_list).groupby(level=0).mean()

def rank_models(average, metric=None):
    if metric is None:
        raise Exception("metric must be specified")
    metric_scores = average.loc[metric]
    # Sort the models based on the selected metric in descending order
    ranked_models = metric_scores.sort_values(ascending=False)
    return ranked_models

def save_result(dataset_scores, path, metric):
    '''
    Save the results of the experiment to a file
    '''
    file = f"{path}/{metric}.csv"
    score_list = []
    for score in dataset_scores.values():
        value = score.loc[metric]
        score_list.append(value)
    df = pd.DataFrame(score_list, index=dataset_scores.keys())
    df.to_csv(file)

def save_df(df, path, name):
    file = f"{path}/{name}.csv"
    df.to_csv(file)