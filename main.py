from dataloader import Dataset
from models import naive_rolling_average, model_LSTM, naive_last_time_step, DA_RNN, Conv_Model
import argparse
import torch
from data import load_data
from rolling import Rolling
from utils import save_or_append


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_results = 'results.csv'

params_data_generation = {
    'batch_size': 64,
    'shuffle': False,
    'num_workers': 0,
}

params_dataset = {
    'target' : 'T (degC)',
    'device' : device
}

rolling_dict = {
    'len_buffer' : 100,
    'len_training' : 15000,
    'len_test' : 2000,
    'len_val' : 2000,
    'n_fold' : 10,
}

params_model = {
    #"model_name" : 'DA-RNN',
    #"n_hidden" : 64,
    "num_layers" : 2,
    "dropout" : 0.15,
    "learning_rate" : 1e-3,
    "adam_eps" : 1e-8,
    "n_epoch" : 25,
    #"name" : df.columns,
    "device" : device,
    "verbose" : True,
    "n_epochs_stop" : 5
}


def main():
    ap = argparse.ArgumentParser("Weather Forecast")

    ap.add_argument("model", choices=["DA-RNN", "LSTM", "naive_last_step", "naive_rolling_average", "conv"], help="chose which model to train")

    ap.add_argument("n_hidden", type=int, help="dimension of the hidden layer")

    ap.add_argument("n_past", type=int, help="number of past step as inputs")

    ap.add_argument("time_shift", type=int, help="prediction horizon")
    args = ap.parse_args()
    print("Loading Data")
    df = load_data()
    print("Data Loaded")

    params_dataset['n_past'] = args.n_past
    params_dataset['time_shift'] = args.time_shift
    params_dataset['df'] = df

    params_model['n_hidden'] = args.n_hidden
    params_model['model_name'] = args.model_name
    params_model['name'] = df.columns


    Roll = Rolling(rolling_dict, params_dataset, params_data_generation, params_model)
    Roll.rolling_training()
    Roll.compute_loss()
    results = Roll.return_df()
    save_or_append(results, path_results)


if __name__ == "__main__":
    main()