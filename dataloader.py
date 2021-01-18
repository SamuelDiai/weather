import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch import Tensor

from sklearn import preprocessing
import pandas as pd
import numpy as np

class Dataset(torch.utils.data.Dataset):
    """ DataLoader
        Args:
            list_IDs:
            target: output regression target
            n_past:
            time_shift: number of timesteps ahead we can predict
            df: dataframe
        Raises:
          ValueError: when target is not on the list or when time_shift is not strictly positive
        """
    'Characterizes a dataset for PyTorch'
    def __init__(self, step, params_dataset):
        'Defines which column of the dataset is the target :'
        self.__dict__.update((k, v) for k, v in params_dataset.items())
        self.df_norm = self.normalize(self.df)

        if self.target not in ['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
                               'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
                               'H2OC (mmol/mol)', 'rho (g/m**3)', 'Wx', 'Wy', 'max Wx', 'max Wy']:
            raise ValueError("Target must be on the lsit of targets ! ")
        if self.time_shift <= 0 :
            raise ValueError("Time shift must be strictly positive !")
        self.labels = {}
        for index, value in self.df_norm[self.target].shift(- self.time_shift).copy().items():
            self.labels[index] = value

        if step == 'train':
            self.list_IDs = np.arange(self.beginning_train, self.end_train)
        elif step == 'test':
            self.list_IDs = np.arange(self.beginning_test, self.end_test)
        elif step == 'val':
            self.list_IDs = np.arange(self.beginning_val, self.end_val)
        else :
          raise ValueError('The training step is not valid ! ')

    def normalize(self, df):
      df_train = df.iloc[self.beginning_train - self.n_past:self.end_train]
      scaler = preprocessing.RobustScaler().fit(df_train)

      df_norm = pd.DataFrame(scaler.transform(df))
      df_norm.columns = df.columns

      return df_norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.FloatTensor([np.array(self.df_norm.iloc[ID + 1 - self.n_past:ID + 1])]).squeeze().to(device=self.device)
        y = torch.FloatTensor([[self.labels[ID]]]).to(device=self.device)

        return X, y
