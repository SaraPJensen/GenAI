from dataloaders import dataset_and_loader, combined_dataloader
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import os

torch.manual_seed(1)

'''
#Define all the datasets and dataloaders
real_train_dataset, real_val_dataset, real_test_dataset, real_train_loader, real_val_loader, real_test_loader = dataset_and_loader('real')
gaussian_train_dataset, gaussian_val_dataset, gaussian_test_dataset, gaussian_train_loader, gaussian_val_loader, gaussian_test_loader = dataset_and_loader('gaussian')
ctgan_train_dataset, ctgan_val_dataset, ctgan_test_dataset, ctgan_train_loader, ctgan_val_loader, ctgan_test_loader = dataset_and_loader('ctgan')
copula_train_dataset, copula_val_dataset, copula_test_dataset, copula_train_loader, copula_val_loader, copula_test_loader = dataset_and_loader('copula')
tvae_train_dataset, tvae_val_dataset, tvae_test_dataset, tvae_train_loader, tvae_val_loader, tvae_test_loader = dataset_and_loader('tvae')


#Define the combined datasets and dataloaders
real_gaussian_train_dataset, real_gaussian_val_dataset, real_gaussian_test_dataset, real_gaussian_train_loader, real_gaussian_val_loader, real_gaussian_test_loader = combined_dataloader(real_train_dataset, real_val_dataset, real_test_dataset, gaussian_train_dataset, gaussian_val_dataset, gaussian_test_dataset)
real_ctgan_train_dataset, real_ctgan_val_dataset, real_ctgan_test_dataset, real_ctgan_train_loader, real_ctgan_val_loader, real_ctgan_test_loader = combined_dataloader(real_train_dataset, real_val_dataset, real_test_dataset, ctgan_train_dataset, ctgan_val_dataset, ctgan_test_dataset)
real_copula_train_dataset, real_copula_val_dataset, real_copula_test_dataset, real_copula_train_loader, real_copula_val_loader, real_copula_test_loader = combined_dataloader(real_train_dataset, real_val_dataset, real_test_dataset, copula_train_dataset, copula_val_dataset, copula_test_dataset)
real_tvae_train_dataset, real_tvae_val_dataset, real_tvae_test_dataset, real_tvae_train_loader, real_tvae_val_loader, real_tvae_test_loader = combined_dataloader(real_train_dataset, real_val_dataset, real_test_dataset, tvae_train_dataset, tvae_val_dataset, tvae_test_dataset)
'''

def get_dataloader(datatype):
        if datatype == 'real':
            real_train_dataset, real_val_dataset, real_test_dataset, real_train_loader, real_val_loader, real_test_loader = dataset_and_loader('real')
            return real_train_dataset, real_val_dataset, real_test_dataset, real_train_loader, real_val_loader, real_test_loader

        elif datatype == 'gaussian':
            gaussian_train_dataset, gaussian_val_dataset, gaussian_test_dataset, gaussian_train_loader, gaussian_val_loader, gaussian_test_loader = dataset_and_loader('gaussian')
            return gaussian_train_dataset, gaussian_val_dataset, gaussian_test_dataset, gaussian_train_loader, gaussian_val_loader, gaussian_test_loader 

        elif datatype == 'ctgan':
            ctgan_train_dataset, ctgan_val_dataset, ctgan_test_dataset, ctgan_train_loader, ctgan_val_loader, ctgan_test_loader = dataset_and_loader('ctgan')
            return ctgan_train_dataset, ctgan_val_dataset, ctgan_test_dataset, ctgan_train_loader, ctgan_val_loader, ctgan_test_loader 

        elif datatype == 'copula':
            copula_train_dataset, copula_val_dataset, copula_test_dataset, copula_train_loader, copula_val_loader, copula_test_loader = dataset_and_loader('copula')
            return copula_train_dataset, copula_val_dataset, copula_test_dataset, copula_train_loader, copula_val_loader, copula_test_loader 

        elif datatype == 'tvae':
            tvae_train_dataset, tvae_val_dataset, tvae_test_dataset, tvae_train_loader, tvae_val_loader, tvae_test_loader = dataset_and_loader('tvae')
            return tvae_train_dataset, tvae_val_dataset, tvae_test_dataset, tvae_train_loader, tvae_val_loader, tvae_test_loader



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
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)



# Training loop with validation
num_epochs = 50
best_val_loss = float('inf')
train_losses = []
train_accuracy = []
val_losses = []
val_accuracy = []
input_shape = 30
learning_rate = 0.001

datatype = 'copula'
progress_file_path = f'training_progress/{datatype}_progress.csv' 

os.makedirs('training_progress', exist_ok=True)

with open(progress_file_path, "w") as file:
    file.write('epoch,training_loss,train_accuracy,validation_loss,validation_accuracy\n')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(input_dim=input_shape).to(device)

_, _, _, train_loader, val_loader, test_loader = get_dataloader(datatype)

# Loss and optimizer
criterion = nn.BCELoss()
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

    # Validation
    model.eval()
    running_val_loss = 0.0
    running_val_accuracy = 0.0 

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_val_loss += loss.item()

            preds = torch.round(outputs)
            correct = (preds == targets).sum()
            accuracy = correct / targets.size(0)
            running_val_accuracy += accuracy.item()


    avg_val_loss = running_val_loss / len(val_loader)
    avg_val_acc = running_val_accuracy / len(val_loader)

    val_losses.append(avg_val_loss)
    val_accuracy.append(avg_val_acc)

    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {avg_val_acc:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f'cancer_models/{datatype}_best_model.pt')

    file = open(progress_file_path, "a")
    file.write(f'{epoch},{avg_train_loss},{avg_train_accuracy},{avg_val_loss},{avg_val_acc}\n')
    file.close()

    #input()


'''
# Plot loss curves
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("loss_curve.png")
plt.show()
'''