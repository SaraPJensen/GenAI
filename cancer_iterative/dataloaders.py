import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split, Subset
from sklearn.preprocessing import StandardScaler
torch.manual_seed(1)


class CompleteDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv('datasets/real_cancer_data.csv')

        self.inputs = torch.tensor(self.data.iloc[:, :-1].values)
        self.targets = torch.tensor(self.data.iloc[:, -1].values)

        scaler = StandardScaler()
        self.inputs = torch.from_numpy(scaler.fit_transform(self.inputs))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.inputs[idx].float()
        label = self.targets[idx].float() 

        return features, label




class Cancer_Dataset(Dataset):
    def __init__(self, datatype, percentage):
        self.datatype = datatype

        if self.datatype == 'real' or self.datatype == 'real_complete':
            self.filepath = 'datasets/real_cancer_data.csv'

        elif self.datatype == 'gaussian' or self.datatype == 'ctgan' or self.datatype == 'copula' or self.datatype == 'tvae':
            self.filepath = f'datasets/{percentage}_train/{datatype}_{percentage}_cancer_data.csv'

        else: 
            print("Wrong datatype")
            exit()

        self.data = pd.read_csv(self.filepath)

        self.inputs = torch.tensor(self.data.iloc[:, :-1].values)
        self.targets = torch.tensor(self.data.iloc[:, -1].values)


    def fit_transform_scaling(self, training_data, scaler = None):
        inputs, _ = training_data[:]
        scaler = StandardScaler()

        if scaler is None: 
            scaled_training_data = scaler.fit(inputs)

        self.inputs = torch.from_numpy(scaler.transform(self.inputs))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.inputs[idx].float()
        label = self.targets[idx].float() 

        return features, label


def complete_dataset_loader():
    dataset = CompleteDataset()

    train_loader = DataLoader(dataset,
                        batch_size = 8,
                        shuffle = True,
                        num_workers = 4,
                        drop_last = True
                        )
        
    return dataset, train_loader



def dataset_and_loader(datatype, percentage):

    dataset = Cancer_Dataset(datatype, percentage)
    total_size = len(dataset)

    # Split sizes
    if datatype == 'real':
        train_size = int(percentage/100 * total_size)
        test_size = total_size - train_size

        train_dataset = Subset(dataset, list(range(train_size)))
        test_dataset = Subset(dataset, list(range(train_size, len(dataset))))

        
    else: 
        train_size = int(0.7 * total_size)
        test_size = total_size - train_size # ensures full coverage
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    dataset.fit_transform_scaling(train_dataset)

    # print('train dataset size: ', len(train_dataset[:][0]))
    # print('test dataset size: ', len(test_dataset[:][0]))

    train_loader = DataLoader(train_dataset,
                        batch_size = 8,
                        shuffle = True,
                        num_workers = 4,
                        drop_last = True
                        )

    test_loader = DataLoader(test_dataset,
                        batch_size = 8,
                        shuffle = True,
                        num_workers = 4,
                        drop_last = True
                        )

    # print("len train_loader", len(train_loader))
    # print("len test_loader", len(test_loader))

    return train_dataset, test_dataset, train_loader, test_loader


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