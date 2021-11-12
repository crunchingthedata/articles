from collections import Counter
import itertools
import os
import re

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso

def load_data(data_subdir):
    dir = os.path.join('articles/feature-selection/data', data_subdir)
    files = [
        os.path.join(dir, x, 'data.csv')
        for x in os.listdir(dir)
        ]
    data = [pd.read_csv(x) for x in files]
    return data

def count_related_features(d)':
    return len([x for x in d.columns if re.match('related', x)])

def calculate_proportion_by_type(selections):
    all_selections = list(itertools.chain.from_iterable(selections))
    n_total = len(all_selections)
    n_related = count_related_features(all_selections)
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
    variable_importance_selections = []
    shap_selections = []
    for d in data:
        n = count_related_features(d)

        X = d.drop(['outcome'], axis=1)
        clf = RandomForestRegressor(random_state=0, max_depth=2)
        clf.fit(X, d['outcome'])

        feature_names = X.columns
        importance_values = clf.feature_importances_
        top_features = get_top_n_features(importance_values, feature_names, n)
        variable_importance_selections.append(top_features)

        explainer = shap.Explainer(clf)
        shap_values = explainer(X)
        shap_values_ = np.abs(shap_values.values).mean(0)
        top_features = get_top_n_features(shap_values_, feature_names, n)
        shap_selections.append(top_features)

    return variable_importance_selections, shap_selections

def get_feature_correlation_selections(data):
    selections = []
    for d in data:
        n = count_related_features(d)
        cor = d.corr(method='spearman') \
            .reset_index() \
            .sort_values(['outcome'], ascending=False) \
            .head(n+1)
        high_cors = [x for x in cor['index'] if x != 'outcome']
        selections.append(high_cors)
    return selections



datasets = ['simple', 'correlated', 'noise']

metrics = []
for dataset_name in datasets:
    data = load_data(dataset_name)

    correlation = get_feature_correlation_selections(data)
    correlation_p_correct = calculate_proportion_by_type(correlation).get('related')

    feature_importance, shap_ = get_model_artifact_selections(data)
    feature_importance_p_correct = calculate_proportion_by_type(feature_importance).get('related')
    shap_p_correct = calculate_proportion_by_type(shap_).get('related')

    dataset_metrics = [
        [dataset_name, 'correlation', correlation_p_correct],
        [dataset_name, 'feature_importance', feature_importance_p_correct],
        [dataset_name, 'shap', shap_p_correct]
        ]
    metrics.extend(dataset_metrics)

all_metrics = pd.DataFrame(
    metrics,
    columns = ['scenario', 'method', 'score']
    )
all_metrics.to_csv('articles/feature-selection/metrics_by_scenario.csv', index=False)
print(all_metrics)
