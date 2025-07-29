import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split, Subset
from sklearn.preprocessing import StandardScaler
torch.manual_seed(1)


class CompleteDataset(Dataset):
    def __init__(self, scaler):
        self.data = pd.read_csv('datasets/real_wine_data.csv')

        self.inputs = torch.tensor(self.data.iloc[:, :-1].values)
        self.targets = torch.tensor(self.data.iloc[:, -1].values)

        self.inputs = torch.from_numpy(scaler.transform(self.inputs))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.inputs[idx].float()
        label = self.targets[idx].float() 

        return features, label




class WineDataset(Dataset):
    def __init__(self, datatype, percentage, scaler = None):
        self.datatype = datatype

        if self.datatype == 'real':
            self.filepath = 'datasets/real_wine_data.csv'

        elif self.datatype == 'gaussian' or self.datatype == 'ctgan' or self.datatype == 'copula' or self.datatype == 'tvae':
            self.filepath = f'datasets/{percentage}_train/{datatype}_{percentage}_wine_data.csv'

        else: 
            print("Wrong datatype")
            exit()

        self.data = pd.read_csv(self.filepath)

        self.inputs = torch.tensor(self.data.iloc[:, :-1].values)
        self.targets = torch.tensor(self.data.iloc[:, -1].values)

        if scaler is not None:
            self.inputs = torch.from_numpy(scaler.transform(self.inputs))


    def fit_transform_scaling(self, training_data):
        inputs, _ = training_data[:]
        scaler = StandardScaler()
        scaled_training_data = scaler.fit(inputs)

        self.inputs = torch.from_numpy(scaler.transform(self.inputs))
        return scaler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.inputs[idx].float()
        label = self.targets[idx].float() 

        return features, label


def complete_dataset_loader():
    dataset = CompleteDataset()

    train_loader = DataLoader(dataset,
                        batch_size = 2,
                        shuffle = True,
                        num_workers = 4,
                        drop_last = True
                        )
        
    return dataset, train_loader



def dataset_and_loader(datatype, percentage):

    dataset = WineDataset(datatype, percentage)
    total_size = len(dataset)

    # Split sizes
    if datatype == 'real':
        train_size = int(percentage/100 * total_size)

        train_dataset = Subset(dataset, list(range(train_size)))
        test_dataset = Subset(dataset, list(range(train_size, len(dataset))))


    else: 
        train_size = int(0.7 * total_size)
        test_size = total_size - train_size # ensures full coverage
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    scaler = dataset.fit_transform_scaling(train_dataset)

    complete_dataset = CompleteDataset(scaler)  #Scale complete dataset according to same scaler

    real_dataset = WineDataset('real', percentage, scaler)  #Scale real dataset according to same scaler
    real_dataset_size = len(real_dataset)
    real_train_size = int((percentage/100) * real_dataset_size)
    real_test_dataset = Subset(real_dataset, list(range(real_train_size, real_dataset_size)))

    # print('train dataset size: ', len(train_dataset[:][0]))
    # print('real_train_size', real_train_size, len(real_dataset))
    # print('test dataset size: ', len(real_test_dataset[:][0]))
    # print("complete dataset size: ", len(complete_dataset))
    # exit()

    train_loader = DataLoader(train_dataset,
                        batch_size = 2,
                        shuffle = True,
                        num_workers = 4,
                        drop_last = True
                        )

    test_loader = DataLoader(test_dataset,
                        batch_size = 2,
                        shuffle = True,
                        num_workers = 4,
                        drop_last = True
                        )

    real_test_loader = DataLoader(real_test_dataset,
                        batch_size = 2,
                        shuffle = True,
                        num_workers = 4,
                        drop_last = True
                        )
    
    complete_loader = DataLoader(complete_dataset,
                        batch_size = 2,
                        shuffle = True,
                        num_workers = 4,
                        drop_last = True
                        )

    # print("len train_loader", len(train_loader))
    # print("len test_loader", len(test_loader))

    return train_loader, test_loader, real_test_loader, complete_loader


def combined_dataloader(train_dataset1, test_dataset1, train_dataset2, test_dataset2):
    train_combined = ConcatDataset([train_dataset1, train_dataset2])
    test_combined = ConcatDataset([test_dataset1, test_dataset2])
    
    train_loader = DataLoader(train_combined,
                        batch_size = 8,
                        shuffle = True,
                        num_workers = 4,
                        drop_last = True
                        )

    test_loader = DataLoader(test_combined,
                        batch_size = 8,
                        shuffle = True,
                        num_workers = 4,
                        drop_last = True
                        )


    return train_combined, test_combined, train_loader, test_loader





'''
print(real_train_dataset[0])
print(gaussian_train_dataset[0])
print(ctgan_train_dataset[0])
print(copula_train_dataset[0])
print(tvae_train_dataset[0])
'''

'''
elif datatype == 'real_complete':

    train_size = int(percentage/100 * total_size)
    test_size = total_size - train_size

    train_dataset = Subset(dataset, list(range(train_size)))
    test_dataset = Subset(dataset, list(range(train_size, len(dataset))))

    complete_set = ConcatDataset([train_dataset, test_dataset])

    train_loader = DataLoader(complete_set,
                    batch_size = 8,
                    shuffle = True,
                    num_workers = 4,
                    drop_last = True
                    )

    return complete_set, train_loader

    train_dataset = dataset
    dataset.fit_transform_scaling(train_dataset)

    train_loader = DataLoader(train_dataset,
                    batch_size = 8,
                    shuffle = True,
                    num_workers = 4,
                    drop_last = True
                    )
    
    return train_dataset, train_loader
    '''