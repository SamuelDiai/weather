from trainer import Trainer
from dataloader import Dataset
import torch
import numpy as np
import pandas as pd

class Rolling:
    def __init__(self, rolling_dict, params_dataset, params_data_generation, params_model):
            self.__dict__.update((k, v) for k, v in rolling_dict.items())
            self.params_dataset = params_dataset
            self.params_data_generation = params_data_generation
            self.params_model = params_model
            self.params_dataset_complete = self.params_dataset.copy()
            self.list_train_loss = []
            self.list_test_loss = []
            self.list_val_loss = []


    def update_params_dataset(self, i):
        b = self.len_buffer + i * (self.len_training + self.len_test + self.len_val + 3 * self.len_buffer)
        self.params_dataset_complete['beginning_train'] = b
        self.params_dataset_complete['end_train'] = b + self.len_training
        self.params_dataset_complete['beginning_val'] = b + self.len_training + self.len_buffer
        self.params_dataset_complete['end_val'] = b + self.len_training + self.len_buffer + self.len_val
        self.params_dataset_complete['beginning_test'] = b + self.len_training + self.len_buffer + self.len_val + self.len_buffer
        self.params_dataset_complete['end_test'] = b + self.len_training + self.len_buffer + self.len_val + self.len_test

    def create_generator(self):
    # Generators
        training_set = Dataset('train', self.params_dataset_complete)
        training_generator = torch.utils.data.DataLoader(training_set, **self.params_data_generation)

        test_set = Dataset('test', self.params_dataset_complete)
        test_generator = torch.utils.data.DataLoader(test_set, **self.params_data_generation)

        val_set = Dataset('val', self.params_dataset_complete)
        val_generator = torch.utils.data.DataLoader(val_set, **self.params_data_generation)

        dict_generators = {
            'training_generator' : training_generator,
            'test_generator' : test_generator,
            'val_generator' : val_generator
            }

        return dict_generators

    def rolling_training(self):

        for i in range(self.n_fold):
            print('Fold n°{}'.format(i+1))
            # Create generators
            self.update_params_dataset(i)
            dict_generators = self.create_generator()

            # Create model
            mod = Trainer(self.params_model, dict_generators)
            if self.params_model['model_name'] != 'naive_last_step' or self.params_model['model_name'] != 'naive_rolling_average':
                self.nb_parameters = sum(p.numel() for p in mod.model.parameters())
                # Train model
                mod.training()

            # Do forecast
            mod.complete_forecast()

            # Compute losses :
            mod.compute_loss('training')
            mod.compute_loss('test')
            mod.compute_loss('val')

            # Update Loss
            self.list_train_loss.append(mod.result_loss_training)
            self.list_test_loss.append(mod.result_loss_test)
            self.list_val_loss.append(mod.result_loss_val)
            print("TRAIN LOSS ", mod.result_loss_training, "VAL : ", mod.result_loss_val, "Test : ", mod.result_loss_test)

    def compute_loss(self):
        L_train = np.zeros((len(self.list_train_loss[0]), len(self.list_train_loss)))
        L_test = np.zeros((len(self.list_test_loss[0]), len(self.list_test_loss)))
        L_val = np.zeros((len(self.list_val_loss[0]), len(self.list_val_loss)))
        for i, key in enumerate(self.list_train_loss[0]):
            for j in range(len(self.list_train_loss)):
                L_train[i, j] = self.list_train_loss[j][key]
                L_test[i, j] = self.list_test_loss[j][key]
                L_val[i, j] = self.list_val_loss[j][key]
        mean_train = np.mean(L_train, axis = 1)
        mean_test = np.mean(L_test, axis = 1)
        mean_val = np.mean(L_val, axis = 1)
        std_train = np.std(L_train, axis = 1)
        std_val = np.std(L_val, axis = 1)
        std_test = np.std(L_test, axis = 1)
        dict_train = self.list_train_loss[0].copy()
        dict_test = self.list_test_loss[0].copy()
        dict_val = self.list_val_loss[0].copy()
        for i, key in enumerate(dict_train):
            dict_train[key] = [mean_train[i], std_train[i]]
            dict_test[key] = [mean_test[i], std_test[i]]
            dict_val[key] = [mean_val[i], std_val[i]]

        self.train_loss = dict_train
        self.test_loss = dict_test
        self.val_loss = dict_val

    def return_df(self):
        columns = ['model_name', 'nb parameters', 'n_past', 'target', 'time_shift', 'n_hidden', 'num_layers', 'learning_rate', 'dropout', 'n_epoch', 'len_buffer', 'len_training', 'len_test', 'len_val', 'n_fold', 'MAE_train', \
               'MAE_test', 'MAE_val', 'RMSE_train', 'RMSE_test', 'RMSE_val', 'sMAPE_train', 'sMAPE_test', 'sMAPE_val', 'MASE_train', 'MASE_test', 'MASE_val']
        if self.params_model['model_name'] == 'naive_last_step' or self.params_model['model_name'] == 'naive_rolling_average':
            self.nb_parameters = 0
        data = [[self.params_model['model_name'], self.nb_parameters, self.params_dataset['n_past'], self.params_dataset['target'], self.params_dataset['time_shift'], self.params_model['n_hidden'], \
                 self.params_model['num_layers'], self.params_model['learning_rate'], self.params_model['dropout'], self.params_model['n_epoch'], self.len_buffer, self.len_training, \
                 self.len_test, self.len_val, self.n_fold, self.train_loss['MAE'], self.test_loss['MAE'], self.val_loss['MAE'], self.train_loss['RMSE'], self.test_loss['RMSE'], self.val_loss['RMSE'], self.train_loss['sMAPE'], self.test_loss['sMAPE'], self.val_loss['sMAPE'],\
                 self.train_loss['MASE'], self.test_loss['MASE'], self.val_loss['MASE']]]
        return pd.DataFrame(data=data, columns = columns)
