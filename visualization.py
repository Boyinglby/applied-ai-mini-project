'''
Visualization module for the project
'''
import matplotlib.pyplot as plt
import pandas as pd

SAVE_PATH = "results/"

def plot_dataset_score(dataset_scores, metric, path, show=True):
    '''
    Plot combined model scores for all datasets
    '''
    # Create a list of Series
    y = []
    for score in dataset_scores.values():
        acc = score.loc[metric]
        y.append(acc)

    # Combine the list of Series into a DataFrame
    combined_scores = pd.DataFrame(y)

    # Set the index to be a range of numbers
    combined_scores.index = list(dataset_scores.keys())
    
    # plt.figure(figsize=(15, 8))
    
    # Plot the DataFrame
    combined_scores.plot(kind='bar')
    
    plt.xlabel("Datasets")
    plt.ylabel(f"{metric} Scores")
    plt.title(f"{metric} Scores for Different Models Across Datasets")
    plt.legend(title="Models")
    plt.tight_layout()
    plt.savefig(f"{SAVE_PATH}{metric}_scores.png")
    plt.show() if show else plt.close()

    
