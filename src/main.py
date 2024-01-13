import json
import os
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from trainer import Trainer
from generate_data import get_data, generate_test_graphs, generate_training_graphs


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run():
    
    data = get_data()
    generate_training_graphs()
    generate_test_graphs()

    train_dataset = [torch.load(path) for path in os.listdir(".") if "train_graph" in path]
    test_dataset = [torch.load(path) for path in os.listdir(".") if "test_graph" in path]
    
    model = FusionModel(200, 128, 16, 0.01).to(device)
    
    print("Training on {}, batch_size is {}, lr is {}".format(device, batch_size, lr))
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(test_dataset, batch_size=150)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)
    train_acc, train_loss, val_acc, val_loss = trainer.train()


if __name__ == '__main__':
    run()