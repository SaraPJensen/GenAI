import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from sklearn.preprocessing import StandardScaler
torch.manual_seed(1)

class Wine_Dataset(Dataset):
    def __init__(self, datatype):
        self.datatype = datatype

        if self.datatype == 'real':
            self.filepath = 'datasets/real_wine_data.csv'

        elif self.datatype == 'gaussian':
            self.filepath = 'datasets/gaussian_wine_data.csv'

        elif self.datatype == 'ctgan':
            self.filepath = 'datasets/ctgan_data.csv'

        elif self.datatype == 'copula':
            self.filepath = 'datasets/copula_data.csv'

        elif self.datatype == 'tvae':
            self.filepath = 'datasets/tvae_data.csv'

        else: 
            print("Wrong datatype")
            exit()

        self.data = pd.read_csv(self.filepath)

        self.inputs = torch.tensor(self.data.iloc[:, :-1].values)
        self.targets = torch.tensor(self.data.iloc[:, -1].values)


    def fit_transform_scaling(self, training_data):
        inputs, _ = training_data[:]
        scaler = StandardScaler()
        scaled_training_data = scaler.fit(inputs)
        self.inputs = torch.from_numpy(scaler.transform(self.inputs))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.inputs[idx].float()
        label = self.targets[idx].float() #.reshape(-1)

        return features, label



def dataset_and_loader(datatype):
    dataset = Wine_Dataset(datatype)
    total_size = len(dataset)

    # Split sizes
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size  # ensures full coverage

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    dataset.fit_transform_scaling(train_dataset)

    # print(train_dataset[0])
    # print(val_dataset[0])
    # print(test_dataset[0])
    # exit()

    train_loader = DataLoader(train_dataset,
                        batch_size = 16,
                        shuffle = True,
                        num_workers = 4,
                        drop_last = True
                        )

    val_loader = DataLoader(val_dataset,
                        batch_size = 16,
                        shuffle = True,
                        num_workers = 4,
                        drop_last = True
                        )

    test_loader = DataLoader(test_dataset,
                        batch_size = 16,
                        shuffle = True,
                        num_workers = 4,
                        drop_last = True
                        )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def combined_dataloader(train_dataset1, val_dataset1, test_dataset1, train_dataset2, val_dataset2, test_dataset2):
    train_combined = ConcatDataset([train_dataset1, train_dataset2])
    val_combined = ConcatDataset([val_dataset1, val_dataset2])
    test_combined = ConcatDataset([test_dataset1, test_dataset2])
    
    train_loader = DataLoader(train_combined,
                        batch_size = 64,
                        shuffle = True,
                        num_workers = 4,
                        drop_last = True
                        )

    val_loader = DataLoader(val_combined,
                        batch_size = 64,
                        shuffle = True,
                        num_workers = 4,
                        drop_last = True
                        )

    test_loader = DataLoader(test_combined,
                        batch_size = 64,
                        shuffle = True,
                        num_workers = 4,
                        drop_last = True
                        )


    return train_combined, val_combined, test_combined, train_loader, val_loader, test_loader





'''
print(real_train_dataset[0])
print(gaussian_train_dataset[0])
print(ctgan_train_dataset[0])
print(copula_train_dataset[0])
print(tvae_train_dataset[0])
'''

