import os
import pathlib
import random

import numpy as np
import pandas as pd
import yaml


class Simlator:
    """
    Constraints on parameter yaml file.
        - mean and std multipliers must be present and non-negative
    """
    def __init__(self, parameter_file):
        self.params = self._load_parameters(parameter_file)
        self.n = self.params.get('global').get('n_observations')
        self.variable_groups = list(self.params.get('variables').keys())
        self.outcome_mean = self.params.get('variables').get('outcome').get('mean')
        self.distributions = {}
        self._create_dataframe_with_outcome()

    def _load_parameters(self, parameter_file):
        parameter_path = os.path.join(
            'articles/feature-selection/parameters',
            parameter_file + '.yaml'
            )
        with open(parameter_path, 'r') as file:
            params = yaml.safe_load(file)
        print(f'Parameters loaded from: {parameter_path}')
        return params

    def _create_dataframe_with_outcome(self):
        outcome_std_multiplier = self.params.get('variables').get('outcome').get('std').get('multiplier')
        outcome = np.random.normal(
            loc = self.outcome_mean,
            scale = outcome_std_multiplier*self.outcome_mean,
            size = self.n
            )
        data = pd.DataFrame(outcome, columns=['outcome'])
        self.data = data

    def simulate_data(self, parameter_file='parameters'):
        numeric_groups = [x for x in self.variable_groups if x != 'outcome']
        for group in numeric_groups:
            self._simulate_group_variables(group)
        return self.data, self.distributions

    def _simulate_group_variables(self, group, additive_column=None):
        n = self.params.get('variables').get(group).get('n')
        std_multiplier = self.params.get('variables').get(group).get('std').get('multiplier', 1)
        mean_multiplier = self.params.get('variables').get(group).get('mean').get('multiplier', 0)
        additive_column = self.params.get('variables').get(group).get('additive_column', None)
        max_order = self.params.get('variables').get(group).get('max_order', None)
        n_numeric = self.params.get('variables').get(group).get('n_numeric', np.Inf)
        max_categories = self.params.get('variables').get(group).get('max_categories', 2)

        for i in range(n):
            name = f'{group}_{i}'
            mean = np.random.uniform(low = 0, high = self.outcome_mean*mean_multiplier)
            std_multiplier = np.random.uniform(low = 0, high = mean*std_multiplier)
            std = mean*std_multiplier
            sample = np.random.normal(mean, std, self.n)
            if max_order:
                order = random.randint(1, max_order)
                sample = np.power(sample, order)
            if additive_column:
                sample = sample + self.data[additive_column]
            if i >= n_numeric:
                sample = self._categorize_variable(sample, max_categories, name)
                self.data = pd.concat([self.data, sample], axis=1)
            else:
                self.data[name] = sample
            self.distributions[name] = {'mean': mean, 'stdev': std}

    def _categorize_variable(self, sample, n_categories, name):
        n_categories = random.randint(1, n_categories)
        sample = pd.qcut(sample, q=n_categories, precision=1).astype(str)
        dummies = pd.get_dummies(sample, prefix=name)
        return dummies

    def save_data(self, data_subfolder=''):
        dir = os.path.join('articles/feature-selection/data', data_subfolder)
        data_path = os.path.join(dir, 'data.csv')
        distibution_path = os.path.join(dir, 'params.yaml')
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.data.to_csv(data_path, index=False)
        with open(distibution_path, 'w') as file:
            yaml.dump(self.distributions, file)



n_simuations = 5
param_files = ['simple', 'correlated', 'noise', 'scales', 'cardinality']

np.random.seed(1)
random.seed(1)
for param_file in param_files:
    for i in range(n_simuations):
        s = Simlator(param_file)
        data, distributions = s.simulate_data()
        subfolder = os.path.join(param_file, str(i))
        s.save_data(subfolder)
