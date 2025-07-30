import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split, Subset
from sklearn.preprocessing import StandardScaler
torch.manual_seed(1)


class CompleteDataset(Dataset):
    def __init__(self, scaler):
        self.data = pd.read_csv('datasets/real_cancer_data.csv')

        self.inputs = torch.tensor(self.data.iloc[:, :-1].values)
        self.targets = torch.tensor(self.data.iloc[:, -1].values)

        self.inputs = torch.from_numpy(scaler.transform(self.inputs))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.inputs[idx].float()
        label = self.targets[idx].float() 

        return features, label




class CancerDataset(Dataset):
    def __init__(self, datatype, percentage, scaler = None):
        self.datatype = datatype

        if self.datatype == 'real':
            self.filepath = 'datasets/real_cancer_data.csv'

        elif self.datatype == 'gaussian' or self.datatype == 'ctgan' or self.datatype == 'copula' or self.datatype == 'tvae':
            self.filepath = f'datasets/{percentage}_train/{datatype}_{percentage}_cancer_data.csv'

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


class CombinedDataset(Dataset):
    def __init__(self, datatype, percentage, scaler = None):
        self.datatype = datatype

        self.filepath = f'datasets/{percentage}_train/{datatype}_{percentage}_cancer_data.csv'

        syn_data = pd.read_csv(self.filepath)
        real_data = pd.read_csv('datasets/real_cancer_data.csv') 

        real_train_size = int(percentage/100 * len(real_data))
        real_train_dataset = real_data.head(real_train_size)

        self.combined_dataset = pd.concat([syn_data, real_train_dataset], axis=0)

        self.inputs = torch.tensor(self.combined_dataset.iloc[:, :-1].values)
        self.targets = torch.tensor(self.combined_dataset.iloc[:, -1].values)


        if scaler is not None:
            self.inputs = torch.from_numpy(scaler.transform(self.inputs))


    def fit_transform_scaling(self, training_data):
        inputs, _ = training_data[:]
        scaler = StandardScaler()
        scaled_training_data = scaler.fit(inputs)

        self.inputs = torch.from_numpy(scaler.transform(self.inputs))

        return scaler

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, idx):
        features = self.inputs[idx].float()
        label = self.targets[idx].float() 

        return features, label
    


def dataset_and_loader(datatype, percentage):

    dataset = CancerDataset(datatype, percentage)
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

    real_dataset = CancerDataset('real', percentage, scaler)  #Scale real dataset according to same scaler
    real_dataset_size = len(real_dataset)
    real_train_size = int((percentage/100) * real_dataset_size)
    real_test_dataset = Subset(real_dataset, list(range(real_train_size, real_dataset_size)))

    # print('train dataset size: ', len(train_dataset[:][0]))
    # print('real_train_size', real_train_size)
    # print('test dataset size: ', len(real_test_dataset[:][0]))
    # print("complete dataset size: ", len(complete_dataset))
    # exit()

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

    real_test_loader = DataLoader(real_test_dataset,
                        batch_size = 8,
                        shuffle = True,
                        num_workers = 4,
                        drop_last = True
                        )
    
    complete_loader = DataLoader(complete_dataset,
                        batch_size = 8,
                        shuffle = True,
                        num_workers = 4,
                        drop_last = True
                        )

    # print("len train_loader", len(train_loader))
    # print("len test_loader", len(test_loader))

    return train_loader, test_loader, real_test_loader, complete_loader





def combined_dataloader(datatype, percentage):
    combined_dataset = CombinedDataset(datatype, percentage)

    total_size = len(combined_dataset)

    train_size = int(0.7 * total_size)
    test_size = total_size - train_size # ensures full coverage
    combined_train_dataset, combined_test_dataset = random_split(combined_dataset, [train_size, test_size])

    scaler = combined_dataset.fit_transform_scaling(combined_train_dataset)

    complete_dataset = CompleteDataset(scaler)  #Scale complete dataset according to same scaler

    real_dataset = CancerDataset('real', percentage, scaler)  #Scale real dataset according to same scaler
    real_dataset_size = len(real_dataset)
    real_train_size = int((percentage/100) * real_dataset_size)
    real_test_dataset = Subset(real_dataset, list(range(real_train_size, real_dataset_size)))

    # print('combined train dataset size: ', len(combined_train_dataset[:][0]))
    # print('combined test dataset size: ', len(combined_test_dataset[:][0]))
    # print('test dataset size: ', len(real_test_dataset[:][0]))
    # print("complete dataset size: ", len(complete_dataset))
    # exit()


    combined_train_loader = DataLoader(combined_train_dataset,
                        batch_size = 2,
                        shuffle = True,
                        num_workers = 4,
                        drop_last = True
                        )

    combined_test_loader = DataLoader(combined_test_dataset,
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

    return combined_train_loader, combined_test_loader, real_test_loader, complete_loader