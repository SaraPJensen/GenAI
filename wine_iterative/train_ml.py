from dataloaders import dataset_and_loader, combined_dataloader, complete_dataset_loader
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import os

torch.manual_seed(1)


def get_dataloader(datatype, percentage):
        if datatype == 'real':
            real_train_dataset, real_test_dataset, real_train_loader, real_test_loader = dataset_and_loader('real', percentage)
            return real_train_dataset, real_test_dataset, real_train_loader, real_test_loader

        if datatype == 'real_complete':
            complete_dataset, complete_loader = complete_dataset_loader()
            return complete_dataset, complete_loader 

        elif datatype == 'gaussian':
            gaussian_train_dataset, gaussian_test_dataset, gaussian_train_loader, gaussian_test_loader = dataset_and_loader('gaussian', percentage)
            return gaussian_train_dataset, gaussian_test_dataset, gaussian_train_loader, gaussian_test_loader 

        elif datatype == 'ctgan':
            ctgan_train_dataset, ctgan_test_dataset, ctgan_train_loader, ctgan_test_loader = dataset_and_loader('ctgan', percentage)
            return ctgan_train_dataset, ctgan_test_dataset, ctgan_train_loader, ctgan_test_loader 

        elif datatype == 'copula':
            copula_train_dataset, copula_test_dataset, copula_train_loader, copula_test_loader = dataset_and_loader('copula', percentage)
            return copula_train_dataset, copula_test_dataset, copula_train_loader, copula_test_loader 

        elif datatype == 'tvae':
            tvae_train_dataset, tvae_test_dataset, tvae_train_loader, tvae_test_loader = dataset_and_loader('tvae', percentage)
            return tvae_train_dataset, tvae_test_dataset, tvae_train_loader, tvae_test_loader



class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_model(datatype, percentage):

    # Training loop with validation
    num_epochs = 50
    train_losses = []
    train_accuracy = []
    test_losses = []
    test_accuracy = []
    real_test_losses = []
    real_test_accuracy = []
    complete_test_losses = []
    complete_test_accuracy = []

    input_shape = 13
    learning_rate = 0.001

    progress_file_path = f'training_progress/{percentage}_train/{datatype}_progress.csv' 

    os.makedirs(f'training_progress/{percentage}_train', exist_ok=True)
    os.makedirs(f'wine_models/{percentage}_train', exist_ok=True) 

    with open(progress_file_path, "w") as file:
        file.write('epoch,training_loss,train_accuracy,test_loss,test_accuracy,real_test_loss,real_test_accuracy,real_complete_loss,real_complete_accuracy\n')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(input_dim=input_shape).to(device)

    train_loader, test_loader, real_test_loader, complete_loader = dataset_and_loader(datatype, percentage)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        running_train_accuracy = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, {datatype} data"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs) 

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

            preds = torch.round(outputs.detach())
            correct = (preds == targets).sum()
            accuracy = correct / targets.size(0)
            running_train_accuracy += accuracy.item()

        avg_train_loss = running_train_loss / len(train_loader)
        avg_train_accuracy = running_train_accuracy / len(train_loader)

        train_losses.append(avg_train_loss)
        train_accuracy.append(avg_train_accuracy)

        # Testing
        model.eval()
        running_test_loss = 0.0
        running_test_accuracy = 0.0 

        running_real_test_loss = 0.0
        running_real_test_accuracy = 0.0 

        running_complete_test_loss = 0.0
        running_complete_test_accuracy = 0.0 

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_test_loss += loss.item()

                preds = torch.round(outputs)
                correct = (preds == targets).sum()
                accuracy = correct / targets.size(0)
                running_test_accuracy += accuracy.item()

                #print('correct from test loader', correct)


            for inputs, targets in real_test_loader:
                inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_real_test_loss += loss.item()

                preds = torch.round(outputs)
                correct = (preds == targets).sum()
                accuracy = correct / targets.size(0)
                running_real_test_accuracy += accuracy.item()


            for inputs, targets in complete_loader:
                inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_complete_test_loss += loss.item()

                preds = torch.round(outputs)
                correct = (preds == targets).sum()
                accuracy = correct / targets.size(0)
                running_complete_test_accuracy += accuracy.item()

                #print('correct from complete loader', correct)


        avg_test_loss = running_test_loss / len(test_loader)
        avg_test_acc = running_test_accuracy / len(test_loader)

        avg_real_test_loss = running_real_test_loss / len(real_test_loader)
        avg_real_test_acc = running_real_test_accuracy / len(real_test_loader)

        avg_complete_test_loss = running_complete_test_loss / len(complete_loader)
        avg_complete_test_acc = running_complete_test_accuracy / len(complete_loader)

        test_losses.append(avg_test_loss)
        test_accuracy.append(avg_test_acc)

        real_test_losses.append(avg_real_test_loss)
        real_test_accuracy.append(avg_real_test_acc)

        complete_test_losses.append(avg_complete_test_loss)
        complete_test_accuracy.append(avg_complete_test_acc)

        print(f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Accuracy: {avg_test_acc:.4f} | Real test accuracy: {avg_real_test_acc:.4f}")

        file = open(progress_file_path, "a")
        file.write(f'{epoch},{avg_train_loss},{avg_train_accuracy},{avg_test_loss},{avg_test_acc},{avg_real_test_loss},{avg_real_test_acc},{avg_complete_test_loss},{avg_complete_test_acc}\n')
        file.close()


    torch.save(model.state_dict(), f'wine_models/{percentage}_train/{datatype}_final_model.pt')


datatypes = ['real', 'gaussian', 'ctgan', 'tvae', 'copula']
percentages = range(10, 100, 10)


for p in range(10, 100, 10): 
    for datatype in datatypes: 

        print('Datatype: ', datatype)
        print('Percentage: ', p)

        train_model(datatype, p)

