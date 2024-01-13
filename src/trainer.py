import numpy as np
import torch
from sklearn.metrics import accuracy_score, r2_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda'

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return (np.abs((y_true - y_pred) / (y_true + 10e-18))).mean() * 100


class Trainer:
    def __init__(self, network, train_dataloader, eval_dataloader, criterion, optimizer,
                 config, device):
        self.device = device
        self.config = config
        self.network = network
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.best_error = 999999999

    def train_epoch(self, epoch, total_epoch):
        self.network.train()
        running_loss = []
        accuracy = []
        for idx, data in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            data = data.to(device)
            output = self.network(data.x, data.edge_index, data.edge_features,
                                  data.slices, data.batch)  # Perform a single forward pass.

            
            loss = self.criterion(output.float(), data.y.long())
            loss.backward()

            accuracy.append(accuracy_score( data.y.cpu(), output.cpu().argmax(dim=1)))

            self.optimizer.step()
            running_loss.append(loss.item())
        print("[TRAIN] Epoch {}/{}, Accuracy is {} %, Loss is {}".format(epoch, total_epoch, np.mean(accuracy),
                                                                     np.mean(running_loss)))
        print()

        return np.mean(accuracy), np.mean(running_loss)

    def eval_net(self):
        running_eval_loss = []
        self.network.eval()
        accuracy = []
        with torch.no_grad():
            for idx, data in enumerate(self.eval_dataloader):
                self.optimizer.zero_grad()
                data = data.to(device)
                output =self.network(data.x, data.edge_index, data.edge_features,
                                  data.slices, data.batch) # Perform a single forward pass.
                loss_eval = self.criterion(output.float(), data.y.long())

                accuracy.append(accuracy_score( data.y.cpu(), output.cpu().argmax(dim=1)))
                running_eval_loss.append(loss_eval.item())
            print("[EVAL]  Epoch {}/{}, Accuracy is {} %, Loss is {}".format(0, 15000, np.mean(accuracy),
                                                                         np.mean(running_eval_loss)), sep='')
            print(classification_report(output.cpu().argmax(dim=1), data.y.cpu()))
            if np.mean(accuracy) < self.best_error:
                self.best_error = np.mean(accuracy)
                torch.save(self.network.state_dict(), 'best_model.pt')
            return np.mean(accuracy), np.mean(running_eval_loss)


    def eval_net_raw(self):
        running_eval_loss = []
        self.network.eval()
        accuracy = []
        with torch.no_grad():
            for idx, data in enumerate(self.eval_dataloader):
                self.optimizer.zero_grad()
                data = data.to(device)
                output =self.network(data.x, data.edge_index, data.edge_features,
                                  data.slices, data.batch) # Perform a single forward pass.
                return output.float(), data.y.long()

          
    def train(self):
        training_loss = []
        validation_loss = []

        training_accuracy = []
        validation_accuracy = []

        acc, loss = self.eval_net()
        validation_loss.append(loss)
        validation_accuracy.append(acc)

        for i in range(0, self.config['epochs'] + 1):
            print(os.system("!nvidia-smi"))
            print("\n\n\n\n")


            acc, loss = self.train_epoch(i, self.config['epochs'])
            training_loss.append(loss)
            training_accuracy.append(acc)
            writer_train.add_scalar("Loss", loss, i)
            writer_train.add_scalar("Accuracy", acc, i)

            if i % 5 == 0:
                acc, loss = self.eval_net()
                validation_loss.append(loss)
                validation_accuracy.append(acc)
                writer_test.add_scalar("Loss", loss, i)
                writer_test.add_scalar("Accuracy", acc, i)
        return training_accuracy, training_loss, validation_accuracy, validation_loss