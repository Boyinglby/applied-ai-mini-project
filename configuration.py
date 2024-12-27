'''
Configuration file for the project
'''

# Configurations for classifiers
CLASSIFIER_CONFIGS = {
    "supervised": [
        "KNN",
        "RandomForest",
        "SVC"
    ],
    "unsupervised": [
        "KMeans",
        "Birch",
        "MeanShift"
    ]
}

DATASET_CONFIGS_DEMO = [
    {
        "id": 176,
        "name": "blood",
    },
]


# Configurations for datasets
DATASET_CONFIGS = [
    {
        "id": 14,
        "name": "breast-cancer",
    },
    {
        "id": 13,
        "name": "balloons",
    },
    {
        "id": 15,
        "name": "bc-wisc",
    },
    {
        "id": 16,
        "name": "bc-wisc-prog",
    },
    {
        "id": 17,
        "name": "bc-wisc-diag",
    },
    {
        "id": 27,
        "name": "credit-approval",
    },
    # {
    #     "id": 38,
    #     "name": "echocardiogram",
    # },
    {
        "id": 47,
        "name": "horse-colic",
    },
    {
        "id": 52,
        "name": "ionosphere",
    },
    # {
    #     "id": 67,
    #     "name": "molec-biol-promoter",
    # },
    {
        "id": 73,
        "name": "mushroom",
    },
    {
        "id": 94,
        "name": "spambase",
    },
    {
        "id": 101,
        "name": "tic-tac-toe",
    },
    {
        "id": 105,
        "name": "congress-voting",
    },
    {
        "id": 151,
        "name": "conn-bench-sonar",
    },
    {
        "id": 159,
        "name": "magic",
    },
    {
        "id": 161,
        "name": "mammographic",
    },
    {
        "id": 172,
        "name": "ozone",
    },
    {
        "id": 176,
        "name": "blood",
    },
    {
        "id": 184,
        "name": "ac-inflam",
        "target": "bladder-inflammation" # specify target column in case of multiple target columns
    },
    {
        "id": 184,
        "name": "acute-nephritis",
        "target": "nephritis" # specify target column in case of multiple target columns
    },
    {
        "id": 244,
        "name": "fertility",
    },
    {
        "id": 225,
        "name": "ilpd-indian-liver",
    }
]

# Metrics for evaluation
METRICS_CONFIG = {
    "supervised": ["Accuracy", "Precision", "Recall", "F1"],
    "unsupervised": ["Rand", "AMI", "Homogeneity", "Completeness", "V Measure"]
}