from collections import Counter
import itertools
import os
import re

from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.inspection import permutation_importance

def load_data(data_subdir):
    dir = os.path.join('articles/feature-selection/data', data_subdir)
    files = [
        os.path.join(dir, x, 'data.csv')
        for x in os.listdir(dir)
        ]
    data = [pd.read_csv(x) for x in files]
    return data

def count_related_features(d):
    return len([x for x in d.columns if re.match('related', x)])

def calculate_proportion_by_type(selections):
    all_selections = list(itertools.chain.from_iterable(selections))
    n_total = len(all_selections)
    n_related = len([x for x in all_selections if re.match('related_', x)])
    n_random = len([x for x in all_selections if re.match('random_', x)])
    n_correlated = n_total - n_related - n_random
    return {
        'related': n_related / n_total,
        'random': n_random / n_total,
        'correlated': n_correlated / n_total,
        }

def get_top_n_features(values, names, n):
    indices = np.argsort(values)[-1*n:]
    top_features = list(names[indices])
    return top_features

def get_model_artifact_selections(data):
    impurity_selections = []
    shap_selections = []
    permutation_selections = []
    for d in data:
        n = count_related_features(d)
        X = d.drop(['outcome'], axis=1)
        y = d['outcome']
        clf = RandomForestRegressor(random_state=0, max_depth=2)
        clf.fit(X, y)

        feature_names = X.columns
        importance_values = clf.feature_importances_
        top_features = get_top_n_features(importance_values, feature_names, n)
        impurity_selections.append(top_features)

        explainer = shap.Explainer(clf)
        shap_values = explainer(X)
        shap_values_ = np.abs(shap_values.values).mean(0)
        top_features = get_top_n_features(shap_values_, feature_names, n)
        shap_selections.append(top_features)

        permutation_values = permutation_importance(clf, X, y, n_repeats=20, random_state=0).get('importances_mean')
        top_features = get_top_n_features(importance_values, feature_names, n)
        permutation_selections.append(top_features)

    return {
        #'impurity': impurity_selections,
        'permutation': permutation_selections,
        'shap': shap_selections
        }

def get_feature_correlation_selections(data):
    def calculate_top_correlations(d, n, method):
        cor = d.corr(method=method) \
            .reset_index()
        cor['outcome'] = cor['outcome'].abs()
        cor = cor.sort_values(['outcome'], ascending=False) \
            .head(n+1)
        features = [x for x in cor['index'] if x != 'outcome']
        return features

    spearman = []
    pearson = []
    for d in data:
        n = count_related_features(d)
        pearson_ = calculate_top_correlations(d, n, 'pearson')
        spearman_ = calculate_top_correlations(d, n, 'spearman')
        pearson.append(pearson_)
        spearman.append(spearman_)
    return {
        #'pearson': pearson,
        'spearman': spearman
        }



datasets = ['simple', 'correlated', 'noise', 'scales', 'cardinality']

metrics = []
for dataset_name in datasets:
    print(f'Calculating features for {dataset_name}')
    data = load_data(dataset_name)

    corr_selections = get_feature_correlation_selections(data)
    model_selections = get_model_artifact_selections(data)
    selections = {**corr_selections, **model_selections}
    dataset_metrics = [
        [dataset_name, k, calculate_proportion_by_type(v).get('related')]
        for k, v in selections.items()
        ]
    metrics.extend(dataset_metrics)

all_metrics = pd.DataFrame(
    metrics,
    columns = ['scenario', 'method', 'score']
    )
all_metrics.to_csv('articles/feature-selection/metrics_by_scenario.csv', index=False)
print(all_metrics)
