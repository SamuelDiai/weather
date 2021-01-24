from models import naive_rolling_average, model_LSTM, naive_last_time_step, DA_RNN, Conv_Model
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self, params, dict_generators):

        self.__dict__.update((k, v) for k, v in params.items())
        self.__dict__.update((k, v) for k, v in dict_generators.items())
        self.idx_target = self.name.get_loc(self.training_generator.dataset.target)
        self.n_past = self.training_generator.dataset.n_past
        self.input_size = self.training_generator.dataset.df.shape[1]

        if self.model_name == 'LSTM':
            self.model = model_LSTM(self.input_size, self.n_hidden, self.num_layers, self.dropout).to(self.device)
        elif self.model_name == 'naive_last_step':
            self.model = naive_last_time_step(self.idx_target).to(self.device)
        elif self.model_name == 'naive_rolling_average':
            self.model = naive_rolling_average(self.idx_target).to(self.device)
        elif self.model_name == 'DA-RNN' :
            self.model = DA_RNN(input_size = self.input_size - 1, encoder_hidden_size = self.n_hidden, decoder_hidden_size = self.n_hidden, T = self.n_past+1, idx_target = self.idx_target, device = self.device, out_feats=1)
        elif self.model_name == 'conv':
            self.model = Conv_Model(input_size = self.input_size, n_past = self.n_past, device=self.device)

        if  self.model_name != 'naive_last_step' or self.model_name != 'naive_rolling_average':
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate, eps = self.adam_eps)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.MSE = []
        self.MAE = []
        self.eval_epoch = []

    def train(self):
        self.model.training = True
        mse_tot, mae_tot = 0, 0
        for batch_idx, (local_batch, local_labels) in enumerate(self.training_generator):
            self.optimizer.zero_grad()
            output = self.model(local_batch)
            loss = F.mse_loss(output.reshape(-1, 1), local_labels.reshape(-1, 1), reduction = 'sum')
            mae_loss = F.l1_loss(output.reshape(-1, 1), local_labels.reshape(-1, 1), reduction = 'sum')
            loss.backward()
            self.optimizer.step()
            mse_tot += loss
            mae_tot += mae_loss
        nb_element = self.training_generator.dataset.__len__()
        return mse_tot/nb_element, mae_tot/nb_element

    def evaluate(self, generator):
        self.model.training = False
        with torch.no_grad():
            mse_tot, mae_tot = 0, 0
            for batch_idx, (local_batch, local_labels) in enumerate(generator):
                output = self.model(local_batch)
                mse_loss = F.mse_loss(output.reshape(-1, 1), local_labels.reshape(-1, 1), reduction = 'sum')
                mae_loss = F.l1_loss(output.reshape(-1, 1), local_labels.reshape(-1, 1), reduction = 'sum')
                mse_tot += mse_loss
                mae_tot += mae_loss
            nb_element = generator.dataset.__len__()
        return mse_tot/nb_element, mae_tot/nb_element

    def training(self):
        min_val_loss = 1e10
        for i in range(1, self.n_epoch + 1):
            mse_train_tot, mae_train_tot = self.train()
            mse_test_tot, mae_test_tot = self.evaluate(self.test_generator)
            mse_val_tot, mae_val_tot = self.evaluate(self.val_generator)
            if mse_val_tot < min_val_loss:
                min_val_loss = mse_val_tot
                epochs_no_improve = 0
            else :
                epochs_no_improve += 1
            if self.verbose:
                print('Epoch : {:02d} | Loss Train {:.3f} | Loss Val {:.3f} | Loss Test {:.3f}'.format(i, mse_train_tot, mse_val_tot, mse_test_tot))
            self.MSE.append([mse_train_tot, mse_val_tot, mse_test_tot])
            self.MAE.append([mae_train_tot, mae_val_tot, mae_test_tot])
            self.scheduler.step(mse_val_tot)
            self.eval_epoch.append(i)
            if epochs_no_improve >= self.n_epochs_stop:
                break

    def forecast(self, generator):
        self.model.training = False
        forecast = torch.Tensor().to(device = device)
        truth = torch.Tensor().to(device = device)
        with torch.no_grad():
            for batch_idx, (local_batch, local_labels) in enumerate(generator):
                output = self.model(local_batch)
                forecast = torch.cat((forecast, output.reshape(-1)))
                truth = torch.cat((truth, local_labels.reshape(-1)))
        return forecast.cpu().detach(), truth.cpu().detach()

    def complete_forecast(self):
        self.training_forecast, self.training_truth = self.forecast(self.training_generator)
        self.test_forecast, self.test_truth = self.forecast(self.test_generator)
        self.val_forecast, self.val_truth = self.forecast(self.val_generator)

    def plot_forecast(self, dataset, idx_beginning, idx_end):
        if dataset == 'training':
            plt.plot(self.training_forecast.numpy()[idx_beginning:idx_end], label = 'Forecast')
            plt.plot(self.training_truth.numpy()[idx_beginning:idx_end], label = 'Truth')
            plt.title("Differences between Truth and forecast from {} to {} on {} dataset".format(idx_beginning, idx_end, dataset))
            plt.legend()
        elif dataset == 'test':
            plt.plot(self.test_forecast.numpy()[idx_beginning:idx_end], label = 'Forecast')
            plt.plot(self.test_truth.numpy()[idx_beginning:idx_end], label = 'Truth')
            plt.title("Differences between Truth and forecast from {} to {} on {} dataset".format(idx_beginning, idx_end, dataset))
            plt.legend()
        elif dataset == 'val':
            plt.plot(self.val_forecast.numpy()[idx_beginning:idx_end], label = 'Forecast')
            plt.plot(self.val_truth.numpy()[idx_beginning:idx_end], label = 'Truth')
            plt.title("Differences between Truth and forecast from {} to {} on {} dataset".format(idx_beginning, idx_end, dataset))
            plt.legend()

    def compute_loss(self, dataset):
        if dataset == 'training':
            forecast = self.training_forecast.numpy()
            truth = self.training_truth.numpy()
            shift = self.training_generator.dataset.time_shift
        elif dataset == 'test':
            forecast = self.test_forecast.numpy()
            truth = self.test_truth.numpy()
            shift = self.test_generator.dataset.time_shift
        elif dataset == 'val':
            forecast = self.val_forecast.numpy()
            truth = self.val_truth.numpy()
            shift = self.val_generator.dataset.time_shift
        else :
            raise ValueError(' Not valid training step')

        # Err
        err = np.abs(forecast - truth)
        # MAE:
        mae = np.mean(err)
        # RMSE
        rmse = np.mean(err**2)
        # MAPE
        #mape = np.mean(np.abs(100*err/truth))
        # sMAPE
        smape = np.mean(200*err/(forecast + truth))
        # MASE
        scale = np.mean(np.abs(np.roll(truth, shift = shift + 24*6) - truth)[shift + 24*6:])
        mase = np.mean(err /scale)

        if dataset == 'training':
            self.result_loss_training = {"MAE" : mae, "RMSE" : rmse, "sMAPE" : smape, "MASE" : mase}
        elif dataset == 'test':
            self.result_loss_test = {"MAE" : mae, "RMSE" : rmse, "sMAPE" : smape, "MASE" : mase}
        elif dataset == 'val' :
            self.result_loss_val = {"MAE" : mae, "RMSE" : rmse, "sMAPE" : smape, "MASE" : mase}

    def plot_loss(self):

        plt.figure(figsize = (7, 14))
        plt.subplot(211)
        plt.plot(self.eval_epoch, np.array(self.MSE)[:,0], label = 'Train')
        plt.plot(self.eval_epoch, np.array(self.MSE)[:,1], label = 'Val')
        plt.plot(self.eval_epoch, np.array(self.MSE)[:,2], label = 'Test')
        plt.xlabel('Epochs')
        plt.title("Evolution de la MSE")
        plt.legend()

        plt.subplot(212)
        plt.plot(self.eval_epoch, np.array(self.MAE)[:,0], label = 'Train')
        plt.plot(self.eval_epoch, np.array(self.MAE)[:,1], label = 'Val')
        plt.plot(self.eval_epoch, np.array(self.MAE)[:,2], label = 'Test')
        plt.xlabel('Epochs')
        plt.title("Evolution de la MAE")
        plt.legend()

        plt.show()

    def plot_attention_weight(self, idx, gen='test'):
        if gen == 'train':
            X, y = self.training_generator.dataset.__getitem__(idx)
        elif gen == 'test':
            X, y = self.test_generator.dataset.__getitem__(idx)
        elif gen == 'val':
            X, y = self.val_generator.dataset.__getitem__(idx)
        X = X.unsqueeze(0)

        spatial_weights, temporal_weights = self.model.return_weights(X)
        spatial_array = spatial_weights.cpu().detach().squeeze().view(-1,self.n_past).numpy()
        temporal_array = temporal_weights.cpu().detach().squeeze().view(-1, self.n_past, self.n_past).numpy()

        fig, ax = plt.subplots(figsize = (12, 10))
        im = ax.imshow(spatial_array)
        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(spatial_array.shape[1]))
        ax.set_yticks(np.arange(spatial_array.shape[0]))
        # ax.set_xticklabels(col_labels)
        ax.set_yticklabels(self.name[[i for i in range(len(self.name)) if i != self.idx_target]])
        ax.set_xlabel('Past time')


        fig, ax = plt.subplots(figsize = (12, 10))
        im = ax.imshow(spatial_array)
        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(temporal_array.shape[1]))
        ax.set_yticks(np.arange(temporal_array.shape[0]))
        # ax.set_xticklabels(col_labels)
        # ax.set_yticklabels(self.name[[i for i in range(len(self.name)) if i != self.idx_target]])
        ax.set_xlabel('Past time')

        plt.show()
