import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class CodesDataset(Dataset):
    '''
    Dataset class to accommodate our nested bigram features and user number labels
    '''
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.from_numpy(self.features[idx]).type(torch.FloatTensor)
        labels = torch.tensor(int(self.labels[idx])).type(torch.LongTensor)
        return (features, labels)

class NN_Model(nn.Module):
    '''
    Our custom design of NN for this task.
    '''
    def __init__(self, num_classes, num_features):
        super(NN_Model, self).__init__()

        self.num_classes = num_classes
        self.num_features = num_features

        self.features = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.num_features, 500),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(500),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(500, self.num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
# Define training and testing functions
def test_model(model, dset_loader, criterion):
    '''
    Testing routine. Returns accuracy of "model" on a given dataset
    through it's loader. Also prints the loss according to the "criterion".
    '''
    model.train(False)
        
    running_loss = 0
    running_corrects = 0
    running_counter = 0
    for batch_idx, (inputs, labels) in enumerate(dset_loader):

        # forward
        outputs = model(inputs.cuda())
        _, preds = torch.max(outputs.data, 1)

        # loss
        loss = criterion(outputs, labels.cuda())

        # statistics
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data.cuda())

    epoch_loss = float(running_loss) / n_classes
    epoch_acc = float(running_corrects) / n_classes
    print('Test ~ Loss: {:.6f} Acc: {:.6f}'.format(epoch_loss, epoch_acc))
    return epoch_acc
    
    
def train_test(model, train_loader, test_loader, criterion, optimizer, num_epochs=25):
    '''
    Training routine (includes testing). Returns a list of accuracies (one per epoch)
    of the "model" on the test set through it's loader. Also prints the training loss
    and test loss according to the "criterion".
    '''
    optimizer.zero_grad()
    model.train()
    
    list_of_acc = []
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        
        running_loss = 0
        running_corrects = 0
        running_counter = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):

            # forward
            outputs = model(inputs.cuda())
            _, preds = torch.max(outputs.data, 1)

            # loss
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # statistics
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data.cuda())
            
        epoch_loss = float(running_loss) / n_samples
        epoch_acc = float(running_corrects) / n_samples
        list_of_acc.append(epoch_acc)
        print('Train ~ Loss: {:.6f} Acc: {:.6f}'.format(epoch_loss, epoch_acc))
        
        test_acc = test_model(model, test_loader, criterion)
        list_of_acc.append(test_acc)
        model.train(True)
       
    return list_of_acc
        
    
def full_train_test(train_features, train_labels, test_features, test_labels):
    '''
    This function creates the model, datasets, data loaders, criterion, and handles
    the hyperparamenter throught the whole training schedule. Returns the last accuracy 
    of the model in test set.
    '''
    model = NN_Model(num_classes=n_classes, num_features=n_features).cuda()
    train_dset = CodesDataset(train_features, train_labels)
    train_loader = DataLoader(train_dset, batch_size=20, shuffle=True, num_workers=10)
    test_dset = CodesDataset(test_features, test_labels)
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=1)
    criterion = nn.CrossEntropyLoss()
    
    list_of_acc = []
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.1)
    list_of_acc += train_test(model, train_loader, test_loader, criterion, optimizer, num_epochs=5)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.02)
    list_of_acc += train_test(model, train_loader, test_loader, criterion, optimizer, num_epochs=20)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    list_of_acc += train_test(model, train_loader, test_loader, criterion, optimizer, num_epochs=3)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    list_of_acc += train_test(model, train_loader, test_loader, criterion, optimizer, num_epochs=30)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    list_of_acc += train_test(model, train_loader, test_loader, criterion, optimizer, num_epochs=3)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00002)
    list_of_acc += train_test(model, train_loader, test_loader, criterion, optimizer, num_epochs=50)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
    list_of_acc += train_test(model, train_loader, test_loader, criterion, optimizer, num_epochs=3)
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0000001)
    list_of_acc += train_test(model, train_loader, test_loader, criterion, optimizer, num_epochs=25)
    
    return list_of_acc[-1]



if __name__ == '__main__':
    ###############################################################################
    Fsubdir = '/dataframe/df_2012_4_frequent1sameProb_70.csv'
    ###############################################################################
    # Load data
    df = pd.read_csv(Fsubdir).drop('Unnamed: 0',axis=1)
    n_features = df.shape[1] - 1
    n_samples = df.shape[0]
    print('Initial data:')
    print('-- Number of code samples: {}'.format(n_samples))
    print('-- Number of features: {}'.format(n_features))

    class_set = set(df['classes'])
    n_classes = len(class_set)
    n_problems = int(n_samples/n_classes)
    print('-- Number of users: {}'.format(n_classes))
    print('-- Number of problems per user: {}'.format(n_problems))

    # Data Pre-processing
    # Change user label
    user_2_index = {k:v for k,v in zip(class_set, range(n_classes))}
    # Substitudes username with numeric value
    df['classes'] = df['classes'].map(user_2_index)

    # Transform into np arrays
    df_features = df.drop('classes',axis=1)
    df_features.set_index([list(range(n_samples))], inplace=True)
    X = df_features.values
    y = df['classes'].values

    print('SRFNN on data starting...')

    # Split in n_problems subsets with n_problems-1 problems per user
    skf = StratifiedKFold(n_splits=n_problems)
    cv_final_accs = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        print('TRAINING AND TESTING FOR SPLIT {}'.format(i))
        round_acc = full_train_test(X_train, y_train, X_test, y_test)
        cv_final_accs.append(round_acc)
        print('\n')
        print('\n')
    
    # After collecting the final test accuracies for each cross-validation
    # round, print the mean and std of such collection.
    cv_final_accs = np.array(cv_final_accs)
    acc_mean = np.mean(cv_final_accs)
    acc_std = np.std(cv_final_accs)
    print('Mean Acc: {}, Sdt Acc: {}'.format(acc_mean, acc_std))
    