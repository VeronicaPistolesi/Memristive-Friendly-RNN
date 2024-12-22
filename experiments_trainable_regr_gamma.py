from memristive_friendly import *
import os
import pickle
from time import time, localtime, strftime
import itertools
import numpy as np
import tensorflow as tf
from tensorflow import keras
from aeon.datasets import load_regression
from sklearn.model_selection import train_test_split
import random

# List all available GPUs
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Configure TensorFlow to use all available GPUs
        tf.config.set_visible_devices(gpus, 'GPU')
        
        # Enable memory growth on all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        print(f"{len(gpus)} GPU(s) configured and ready for use.")
        
    except RuntimeError as e:
        print(f"Error during GPU configuration: {e}")


# Impostazioni generali dell'esperimento
num_guesses = 5  # Numero di tentativi per la valutazione finale dopo la selezione del modello
max_trials = 200  # Numero di configurazioni campionate casualmente durante la selezione del modello
num_guesses_ms = 1  # Numero di tentativi necessari per la selezione del modello (in questo caso 1 è sufficiente)
max_time = 60 * 60 * 10  # Tempo massimo consentito per il processo di selezione del modello = 10 ore
num_epochs = 5000  # Numero massimo di epoche per l'addestramento
root_path = './'  # Percorso radice per gli esperimenti
#datasets = ['FloodModeling1', 'FloodModeling2']  # Lista dei dataset da esplorare

datasets = ['Covid3Month', 'AppliancesEnergy', 'FloodModeling1', 'FloodModeling2', 'FloodModeling3']

# Configurazione della GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Cambia questo con l'id della GPU che desideri utilizzare (es. "2")

# Set the seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# Caricamento dei dati del dataset
for dataset_name in datasets:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    x_train_all_aeon, y_train_all_aeon = load_regression(name=dataset_name, split="train")
    x_test_aeon, y_test_aeon = load_regression(name=dataset_name, split="test")

    x_train_all = np.transpose(x_train_all_aeon, (0, 2, 1))
    x_test = np.transpose(x_test_aeon, (0, 2, 1))
    y_train_all = y_train_all_aeon
    y_test = y_test_aeon

    x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

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

    def rmse_loss(y_true, y_pred):
        return tf.keras.backend.sqrt(tf.keras.losses.mean_squared_error(y_true, y_pred))

    output_units = 1
    output_activation = 'linear'
    loss_function = rmse_loss

    results_path = os.path.join(root_path, 'results_gamma', dataset_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    def log_results(results_logger, text):
        results_logger.write(text + '\n')
        print(text)

    for model_type, create_model_fn, param_grid in [
        ('MF-RNN', lambda **kwargs: keras.Sequential([
            keras.layers.RNN(MFCell(units=kwargs['num_units'], kp0=kwargs['kp0'], kd0=kwargs['kd0'], dt=kwargs['dt'], etap=kwargs['etap'], etad=kwargs['etad'], gamma=kwargs['gamma'], p=kwargs['p'])),
            keras.layers.Dense(output_units, activation=output_activation)
        ]), {
            'num_units': [100],
            'kp0': [0.0001],
            'kd0': [0.5],
            'etap': [10],
            'etad': [1],
            'dt': [0.001, 0.01, 0.1],
            'gamma': [0.5, 0.8, 0.95, 1], 
            'p': [1, 5],
            'lr': [0.001, 0.01], 
            'patience': [10],
            'batch_size': [64] 
        }),
        ('RNN', lambda **kwargs: keras.Sequential([
            keras.layers.SimpleRNN(units=kwargs['num_units']),
            keras.layers.Dense(output_units, activation=output_activation)
        ]), {
            'num_units': [100],
            'lr': [0.001, 0.01], 
            'patience': [10],
            'batch_size': [64]
        })
    ]:
        param_combinations = list(itertools.product(*param_grid.values()))

        model_selection_times_filename = os.path.join(results_path, 'model_selection_times_' + model_type + '.p')
        times_filename = os.path.join(results_path, 'times_' + model_type + '.p')
        rmse_filename = os.path.join(results_path, 'rmse_' + model_type + '.p')
        results_logger_filename = os.path.join(results_path, 'results_logger_' + model_type + '.txt')

        with open(results_logger_filename, 'w') as results_logger:
            log_results(results_logger, f'Experiment with {model_type} on dataset {dataset_name} starting now')
            time_string_start = strftime("%Y/%m/%d %H:%M:%S", localtime())
            log_results(results_logger, f'** local time = {time_string_start}')

            initial_model_selection_time = time()
            best_val_score = float('inf')
            best_params = None

            for param_values in param_combinations:
                if time() - initial_model_selection_time > max_time:
                    log_results(results_logger, '--> Terminating model selection due to time limit.')
                    break

                params = dict(zip(param_grid.keys(), param_values))
                val_score = 0

                for _ in range(num_guesses_ms):
                    try:
                        model = create_model_fn(**params)
                        model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['lr']),
                                      loss=loss_function, metrics=[keras.metrics.RootMeanSquaredError()])

                        model.fit(x_train, y_train, verbose=0, epochs=num_epochs,
                                  validation_data=(x_val, y_val), batch_size=params['batch_size'],
                                  callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=params['patience'])])

                        _, val_score_new = model.evaluate(x_val, y_val, verbose=0)
                        val_score += val_score_new
                    except Exception as e:
                        log_results(results_logger, f'Error with parameters {params}: {e}')
                        continue

                val_score /= num_guesses_ms
                log_results(results_logger, f'RUN {param_combinations.index(param_values)}. Score = {val_score} with params: {params}')

                if val_score < best_val_score:
                    best_val_score = val_score
                    best_params = params
                    log_results(results_logger, f'New best score: {best_val_score} with params: {best_params}')

            elapsed_model_selection_time = time() - initial_model_selection_time
            time_string_end = strftime("%Y/%m/%d %H:%M:%S", localtime())
            log_results(results_logger, f'Model selection concluded at local time = {time_string_end} with best_val_score= {best_val_score}')

            if best_params:
                rmse_ts = []
                required_time = []

                for _ in range(num_guesses):
                    start_time = time()
                    model = create_model_fn(**best_params)
                    model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_params['lr']),
                                  loss=loss_function, metrics=[keras.metrics.RootMeanSquaredError()])

                    model.fit(x_train_all, y_train_all, verbose=0, epochs=num_epochs,
                              batch_size=best_params['batch_size'],
                              callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=best_params['patience'])])

                    _, rmse = model.evaluate(x_test, y_test, verbose=0)
                    required_time.append(time() - start_time)
                    rmse_ts.append(rmse)

                log_results(results_logger, f'-- {model_type} on {dataset_name} --')
                log_results(results_logger, f'Results: MEAN {np.nanmean(rmse_ts)} STD {np.nanstd(rmse_ts)}')
                log_results(results_logger, f'Required time: MEAN {np.mean(required_time)} STD {np.std(required_time)}')
                log_results(results_logger, f'Total model selection time: {elapsed_model_selection_time}')

                with open(model_selection_times_filename, 'wb') as f:
                    pickle.dump(elapsed_model_selection_time, f)
                with open(times_filename, 'wb') as f:
                    pickle.dump(required_time, f)
                with open(rmse_filename, 'wb') as f:
                    pickle.dump(rmse_ts, f)

                model.summary(print_fn=lambda x: log_results(results_logger, x))
            else:
                log_results(results_logger, 'No valid model found during the selection process.')
