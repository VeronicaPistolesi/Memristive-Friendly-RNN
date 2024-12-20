from memristive_friendly import * 
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import pickle
import random
from time import time, localtime, strftime
from aeon.datasets import load_classification
from sklearn.model_selection import train_test_split
import itertools

#print('ciao')
print(tf.__version__)

print("Is GPU available:", tf.config.list_physical_devices('GPU'))

# GPU setup
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" # change this to the id of the GPU you can use (e.g., "2")

# Set the seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# common experimental setting:
num_guesses = 5 # number of reservoir guesses for final evaluation after model selection
max_time = 60 * 60 * 10 # number of seconds that a model selection process is allowed to last = 10 h

root_path = './' # set this one to the root folder for the experiments

#datasets = ['SyntheticControl', 'ECG5000', 'Earthquakes', 'FordA']
datasets = ['JapaneseVowels', 'SyntheticControl', 'ECG5000', 'GunPoint', 'Wafer', 'Epilepsy', 'Coffee'] #, 'Earthquakes', 'OSULeaf', 'ShapesAll', 'StandWalkJump', 'HandOutlines']

#datasets = ['JapaneseVowels', 'GunPoint', 'Wafer', 'Epilepsy', 'Coffee', 'OSULeaf', 'ShapesAll', 'StandWalkJump', 'HandOutlines']

# MF parameter grid
param_grid_mf = {
    'num_units': [100], 
    'input_scaling': [0.1, 1, 10],
    'memory_factor': [0.8, 0.9, 0.95, 0.99], 
    'bias_scaling': [0, 0.001, 0.1, 1], 
    'alpha': [1],
    'gamma': [0.5, 0.8, 0.95, 1], 
    'p': [1, 5],
    'kp0': [0.0001],
    'kd0': [0.5],
    'etap': [10],
    'etad': [1],
    'dt': [0.001, 0.01, 0.1] # 0.1
}

param_combinations_mf = list(itertools.product(
    param_grid_mf['num_units'],
    param_grid_mf['input_scaling'],
    param_grid_mf['memory_factor'],
    param_grid_mf['bias_scaling'],
    param_grid_mf['alpha'],
    param_grid_mf['gamma'],
    param_grid_mf['p'],
    param_grid_mf['kp0'],
    param_grid_mf['kd0'],
    param_grid_mf['etap'],
    param_grid_mf['etad'],
    param_grid_mf['dt']
))

# ESN parameter grid
param_grid_esn = {
    'num_units': [100],
    'leaky': [0.1, 0.3, 0.5],
    'input_scaling': [0.1, 1, 10],
    'bias_scaling': [0, 0.001, 0.1, 1],
    'spectral_radius': [0.8, 0.9, 0.95, 0.99]
}

param_combinations_esn = list(itertools.product(
    param_grid_esn['num_units'],
    param_grid_esn['leaky'],
    param_grid_esn['input_scaling'],
    param_grid_esn['bias_scaling'],
    param_grid_esn['spectral_radius']
))

# --- EXPERIMENT STARTS HERE ---

for dataset_name in datasets:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    x_train_all_aeon, y_train_all_aeon = load_classification(name=dataset_name, split="train")
    x_test_aeon, y_test_aeon = load_classification(name=dataset_name, split="test")

    x_train_all = np.transpose(x_train_all_aeon, (0, 2, 1))
    x_test = np.transpose(x_test_aeon, (0, 2, 1))
    _, y_train_all = np.unique(y_train_all_aeon, return_inverse=True)
    _, y_test = np.unique(y_test_aeon, return_inverse=True)

    x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42, stratify=y_train_all)

    print('Dataset:', dataset_name)
    print('x_train_all shape:', x_train_all.shape)
    print('y_train_all shape:', y_train_all.shape)
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_val shape:', x_val.shape)
    print('y_val shape:', y_val.shape)
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)

    print('max input value:', np.max(x_train_all))
    print('min input value:', np.min(x_train_all))

    output_units = int(max(y_train_all)) + 1
    print('number of classes:', output_units)

    for model_type, param_combinations, param_keys in [
        ('MF-ESN', param_combinations_mf, ['num_units', 'input_scaling', 'memory_factor', 'bias_scaling', 'alpha', 'gamma', 'p', 'kp0', 'kd0', 'etap', 'etad', 'dt']),
        ('ESN', param_combinations_esn, ['num_units', 'leaky', 'input_scaling', 'bias_scaling', 'spectral_radius'])
    ]:
        results_path = os.path.join(root_path, 'results_gamma', dataset_name)
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        model_selection_times_filename = os.path.join(results_path, 'model_selection_times_' + model_type + '.p')
        times_filename = os.path.join(results_path, 'times_' + model_type + '.p')
        accuracy_filename = os.path.join(results_path, 'accuracy_' + model_type + '.p')

        results_logger_filename = os.path.join(results_path, 'results_logger_' + model_type + '.txt')

        results_logger = open(results_logger_filename, 'w')
        results_logger.write('Experiment with ' + model_type + ' on dataset ' + dataset_name + ' starting now\n')
        time_string_start = strftime("%Y/%m/%d %H:%M:%S", localtime())
        results_logger.write('** local time = ' + time_string_start + '\n')

        initial_model_selection_time = time()

        best_val_score = 0
        best_params = None

        for params in param_combinations:
            if time() - initial_model_selection_time > max_time:
                print('--> Terminating model selection due to time limit.')
                results_logger.write('--> Terminating model selection due to time limit.\n')
                break

            param_dict = dict(zip(param_keys, params))

            try:
                val_score = 0
                for _ in range(num_guesses):
                    if model_type == 'MF-ESN':
                        model = MF(
                            units=param_dict['num_units'], kp0=param_dict['kp0'], kd0=param_dict['kd0'], 
                            etap=param_dict['etap'], etad=param_dict['etad'], dt=param_dict['dt'], 
                            input_scaling=param_dict['input_scaling'], bias_scaling=param_dict['bias_scaling'], 
                            memory_factor=param_dict['memory_factor'], alpha=param_dict['alpha'], gamma=param_dict['gamma'], p=param_dict['p']
                        )
                    elif model_type == 'ESN':
                        model = ESN(
                            units=param_dict['num_units'], leaky=param_dict['leaky'], 
                            input_scaling=param_dict['input_scaling'], bias_scaling=param_dict['bias_scaling'], 
                            spectral_radius=param_dict['spectral_radius']
                        )
                    model.fit(x_train, y_train)
                    val_score += model.evaluate(x_val, y_val)

                val_score /= num_guesses

                print(f'Score: {val_score} with params: {params}')

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_params = param_dict
                    print(f'New best score: {best_val_score} with params: {best_params}')
            except Exception as e:
                print(f'Error with parameters {params}: {e}')
                continue

        print(f'Model selection completed for {model_type}')
        elapsed_model_selection_time = time() - initial_model_selection_time
        time_string = strftime("%Y/%m/%d %H:%M:%S", localtime())
        results_logger.write(f'Model selection concluded at local time = {time_string}\n')

        if best_params:
            acc_ts = []
            required_time = []
            for i in range(num_guesses):
                initial_time = time()
                if model_type == 'MF-ESN':
                    model = MF(
                        units=best_params['num_units'], kp0=best_params['kp0'], kd0=best_params['kd0'], 
                        etap=best_params['etap'], etad=best_params['etad'], dt=best_params['dt'], 
                        input_scaling=best_params['input_scaling'], bias_scaling=best_params['bias_scaling'], 
                        memory_factor=best_params['memory_factor'], alpha=best_params['alpha'], gamma=best_params['gamma'], p=best_params['p']
                    )
                elif model_type == 'ESN':
                    model = ESN(
                        units=best_params['num_units'], leaky=best_params['leaky'], 
                        input_scaling=best_params['input_scaling'], bias_scaling=best_params['bias_scaling'], 
                        spectral_radius=best_params['spectral_radius']
                    )
                model.fit(x_train_all, y_train_all)
                acc = model.evaluate(x_test, y_test)
                required_time.append(time() - initial_time)
                acc_ts.append(acc)

            time_string_end = strftime("%Y/%m/%d %H:%M:%S", localtime())
            results_logger.write(f'*** Best model assessment concluded at local time = {time_string_end}\n')

            with open(model_selection_times_filename, 'wb') as f:
                pickle.dump(elapsed_model_selection_time, f)
            with open(times_filename, 'wb') as f:
                pickle.dump(required_time, f)
            with open(accuracy_filename, 'wb') as f:
                pickle.dump(acc_ts, f)

            print(f'--{model_type} on {dataset_name}--')
            print(f'Results: MEAN {np.mean(acc_ts)} STD {np.std(acc_ts)}')
            print(f'Required time: MEAN {np.mean(required_time)} STD {np.std(required_time)}')
            print(f'Total model selection time: {elapsed_model_selection_time}')

            results_logger.write('** Results:\n')
            results_logger.write(f'Start time: {time_string_start}\n')
            results_logger.write(f'End time: {time_string_end}\n')
            results_logger.write(f'Accuracy: MEAN {np.mean(acc_ts)} STD {np.std(acc_ts)}\n')
            results_logger.write(f'Model selection time: {elapsed_model_selection_time} seconds = {elapsed_model_selection_time / 60} minutes\n')
            results_logger.write(f'Average time for TR,TS: MEAN {np.mean(required_time)} STD {np.std(required_time)}\n')
            results_logger.write(f'Model summary:\n')
            results_logger.write(f'Trainable parameters = {model.readout.coef_.shape[0] * model.readout.coef_.shape[1] + model.readout.intercept_.shape[0]}\n')
            results_logger.write(f'Reservoir size = {best_params["num_units"]}\n')
            results_logger.write(f'Best parameters: {best_params}\n')
            results_logger.close()
        else:
            print(f'No valid model found during model selection for {model_type}.')
            results_logger.write(f'No valid model found during model selection for {model_type}.\n')
            results_logger.close()
