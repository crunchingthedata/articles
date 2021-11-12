import os
import random

import numpy as np
import pandas as pd
import yaml

class Simlator:
    """
    Constraints on parameter yaml file.
        - mean and std multipliers must be present and non-negative
    """
    def __init__(self):
        pass

    def simulate_data(self, parameter_path='parameters.yaml'):
        self._load_parameters(parameter_path)
        self._create_dataframe_with_outcome()
        numeric_groups = [x for x in self.variable_groups if x != 'outcome']
        for group in numeric_groups:
            self._simulate_numeric_group_variables(group)
        return self.data, self.distributions

    def _load_parameters(self, path):
        with open(path, 'r') as file:
            params = yaml.safe_load(file)
        self.params = params
        self.n = params.get('global').get('n_observations')
        self.outcome_mean = params.get('variables').get('outcome').get('mean')
        self.outcome_std = params.get('variables').get('outcome').get('std')
        self.variable_groups = list(params.get('variables').keys())
        self.numeric_variable_groups = [
            x
            for x in self.variable_groups
            if params.get('variables').get(x).get('type') == 'numeric'
            ]
        self.distributions = {}

    def _create_dataframe_with_outcome(self):
        outcome = np.random.normal(
            loc = self.outcome_mean,
            scale = self.outcome_std,
            size = self.n
            )
        data = pd.DataFrame(outcome, columns=['outcome'])
        self.data = data

    def _simulate_numeric_group_variables(self, group, additive_column=None):
        group_params = self._get_numeric_group_parameters(group)
        for i in range(group_params['n']):
            name = f'{group}_{i}'
            std = np.random.uniform(
                low=group_params['std_min'],
                high=group_params['std_max']
                )
            mean = np.random.uniform(
                low=group_params['mean_min'],
                high=group_params['mean_max']
                )
            sample = np.random.normal(mean, std, self.n)
            if group_params['max_order']:
                order = random.randint(1, group_params['max_order'])
                sample = np.power(sample, order)
            if group_params['additive_column']:
                sample = sample + self.data[group_params['additive_column']]
            self.data[name] = sample
            self.distributions[name] = {'mean': mean, 'stdev': std}

    def _get_numeric_group_parameters(self, group):
        group_params = {
            'n': self.params.get('variables').get(group).get('n'),
            'std_multiplier': self.params.get('variables').get(group).get('std').get('multiplier', 1),
            'mean_multiplier': self.params.get('variables').get(group).get('mean').get('multiplier', 0),
            'additive_column': self.params.get('variables').get(group).get('additive_column', None),
            'max_order': self.params.get('variables').get(group).get('max_order', None)
            }
        group_params['std_min'] = 0
        group_params['std_max'] = self.outcome_std*group_params['std_multiplier']
        group_params['mean_min'] = 0
        group_params['mean_max'] = self.outcome_mean*group_params['mean_multiplier']
        return group_params

    def save_data(self, data_subfolder=''):
        dir = os.path.join('data', data_subfolder)
        data_path = os.path.join(dir, 'data.csv')
        distibution_path = os.path.join(dir, 'params.yaml')
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.data.to_csv(data_path, index=False)
        with open(distibution_path, 'w') as file:
            yaml.dump(self.distributions, file)



np.random.seed(1)
s = Simlator()
n_simuations = 10
for i in range(n_simuations):
    prefix = 'test'
    subfolder = os.path.join(prefix, str(i))
    data, distributions = s.simulate_data()
    s.save_data(subfolder)
