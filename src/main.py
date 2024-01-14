import json
import os
import torch
from fusion_model import FusionModel
import torch.optim as optim
from torch_geometric.loader import DataLoader
from trainer import Trainer
from generate_data import load_data, generate_test_graphs, generate_training_graphs


device = 'cuda' if torch.cuda.is_available() else 'cpu'



def run():
    
    with open('config.json') as f:
        config = json.load(f)

    batch_size = config["batch_size"]
    lr = config["lr"]
    nfeat = config["nfeat"]
    nhid = config["nhid"]
    nclass = config["nclass"]
    depth = config["depth"]
    image_size = config["image_size"]
    patch_size = config["patch_size"]
    dropout = config["dropout"]
    graph_data_path = config["graph_data_path"]
    data_path = config["data_path"]
    label_path = config["label_path"]
    heads = config["heads"]
    epochs = config["epochs"]
    mat, mat_gt, train_indx, test_indx = load_data(data_path, label_path)

    #generate_training_graphs(mat, mat_gt, train_indx, graph_data_path)
    #generate_test_graphs(mat, mat_gt, test_indx, graph_data_path)

    train_dataset = [torch.load(os.path.join(graph_data_path, path)) for path in os.listdir(graph_data_path) if "train_graph" in path]
    test_dataset = [torch.load(os.path.join(graph_data_path, path)) for path in os.listdir(graph_data_path) if "test_graph" in path]
    
    model = FusionModel(nfeat, nhid, nclass, depth, image_size, patch_size, heads, dropout).to(device)
    
    print("Training on {}, batch_size is {}, lr is {}".format(device, batch_size, lr))
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)
    train_acc, train_loss, val_acc, val_loss = trainer.train(epochs)
    
    return train_acc, train_loss, val_acc, val_loss

if __name__ == '__main__':
    run()