#!/usr/bin/env python3

"""
Aknowledgement

"""
#Import packages

import os, random
from IPython.display import Image

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, ImageFolder
from torchvision.utils import make_grid, save_image

import numpy as np
import pandas as pd

import matplotlib
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from sklearn.metrics import matthews_corrcoef

%matplotlib inline
matplotlib.rcParams['figure.facecolor'] = '#ffffff'

dirc = '/content/gdrive/MyDrive/AI in Microbiome/Filtered normalized data/noBE_removal_clr_and_other_normalizations/MAD/'

#all
file1 = 'phyla_all_3949x1177_pmi_0_clr_75p_MAD.csv' 
file2 = 'phyla_all_565x1177_pmi_0_clr_10p_MAD.csv'
file3 = 'phyla_all_753x1177_pmi_0_clr_15p_MAD.csv'

 #stool
file1 = 'phyla_stool_2840x1177_pmi_0_clr_75p_MAD.csv'
file2 = 'phyla_stool_401x1177_pmi_0_clr_10p_MAD.csv'
file3 = 'phyla_stool_540x1177_pmi_0_clr_15p_MAD.csv'

 #Biopsy
file1 = 'phyla_biopsy_1109x1177_pmi_0_clr_75p_MAD.csv'
file2 = 'phyla_biopsy_164x1177_pmi_0_clr_10p_MAD.csv'
file3 = 'phyla_biopsy_213x1177_pmi_0_clr_15p_MAD.csv'

train_data_file = dirc + file1
val_data_file = dirc + file2
test_data_file = dirc + file3

# sets device for model and PyTorch tensors
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set seeds
if True:
    seed = 2021
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
		
#Pandas data read
if train_data_file.endswith('tsv'):
  microbe_df_train = pd.read_csv(train_data_file, sep='\t')
  microbe_df_val = pd.read_csv(val_data_file, sep='\t')
elif train_data_file.endswith('csv'):
  microbe_df_train = pd.read_csv(train_data_file)
  microbe_df_val = pd.read_csv(val_data_file)
else:
  print('Please make sure file is either tab separated or comma separated')
print(microbe_df_train)

#We get the column index for metadata and count data
metadata_size = 7 #Number of columns that have metadata
col_length = microbe_df_train.shape[1] #Get total number of columns of the matrix
count_colms = col_length - metadata_size #Number of columns that have count data
metadata_start = count_colms + 1 #first column in the matrix that has metadata, we assume the metadata is located in the last columns

print([1, count_colms])
print([metadata_start, col_length])

#Now, we get the features/counts and metadata
#Train data
expr_df_train = microbe_df_train.iloc[:,1:count_colms]
phenotype_df_neat_train = microbe_df_train.iloc[:,metadata_start:col_length]
phenotype_df_train = phenotype_df_neat_train

#We can set 'sample_title' as index by:
phenotype_df_train.set_index('sample_title') #phenotype_df.set_index('sample_title')

#Let's covert UC and CD to IBD
phenotype_df_train.loc[(phenotype_df_train.uc_cd == 'CD'), 'uc_cd'] = 'IBD'
phenotype_df_train.loc[(phenotype_df_train.uc_cd == 'UC'), 'uc_cd'] = 'IBD'

#Validation data
expr_df_val = microbe_df_val.iloc[:,1:count_colms]
phenotype_df_neat_val = microbe_df_val.iloc[:,metadata_start:col_length]
phenotype_df_val = phenotype_df_neat_val

#We can set 'sample_title' as index by:
phenotype_df_val.set_index('sample_title')

#Let's covert UC and CD to IBD
phenotype_df_val.loc[(phenotype_df_val.uc_cd == 'CD'), 'uc_cd'] = 'IBD'
phenotype_df_val.loc[(phenotype_df_val.uc_cd == 'UC'), 'uc_cd'] = 'IBD'

class MicrobDataset_train(Dataset):
  """
  Dataset for binary classification Tumor/Normal
  """
  def __init__(self):
    
    # Select rows whose type is Tumor or Normal
    self.labels_train = phenotype_df_train[phenotype_df_train["uc_cd"].apply(lambda s: s == "IBD" or s == "Control")]

    # Compute categorical embedding, 0 is Normal, 1 is Tumor
    self.labels_train = self.labels_train["uc_cd"].apply(lambda s: s == "Control").astype(int)

    # Get corresponding gene expression profiles
    self.X_train = expr_df_train

  def __getitem__(self, index):
    sample_train = np.array(self.X_train.iloc[index], dtype=np.float32)
    label_train = np.array(self.labels_train.iloc[index], dtype=np.float32)

    return sample_train, label_train

  def __len__(self):
    return len(self.labels_train)

class MicrobDataset_val(Dataset):
  """
  Dataset for binary classification Tumor/Normal
  """
  def __init__(self):
    
    # Select rows whose type is Tumor or Normal
    self.labels_val = phenotype_df_val[phenotype_df_val["uc_cd"].apply(lambda s: s == "IBD" or s == "Control")]

    # Compute categorical embedding, 0 is Normal, 1 is Tumor
    self.labels_val = self.labels_val["uc_cd"].apply(lambda s: s == "Control").astype(int)

    # Get corresponding gene expression profiles
    self.X_val = expr_df_val

  def __getitem__(self, index):
    sample_val = np.array(self.X_val.iloc[index], dtype=np.float32)
    label_val = np.array(self.labels_val.iloc[index], dtype=np.float32)

    return sample_val, label_val

  def __len__(self):
    return len(self.labels_val)

dataset_train = MicrobDataset_train()
dataset_val = MicrobDataset_val()
num_samples_train, num_genes_train = dataset_train.X_train.shape
num_samples_val, num_genes_val = dataset_val.X_val.shape
if num_genes_train == num_genes_val:
  num_genes = num_genes_val

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=2, generator=torch.Generator().manual_seed(seed))
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=32, shuffle=True, num_workers=2, generator=torch.Generator().manual_seed(seed))

class LogisticRegression(nn.Module):
  def __init__(self, input_dim, out_dim):
    super(LogisticRegression, self).__init__()

    # Initialize linear layer ()
    self.linear = nn.Linear(input_dim, out_dim)

  def forward(self, x):
    x = self.linear(x)  # Compute beta . x
    return x

model_type = 'LR'

drop_out = 0.5
input_dim = num_genes
hidden_dim1 = 1024 #512
hidden_dim2 = 512 #256
hidden_dim3 = 256 #128
hidden_dim4 = 128
outputs = 2

if model_type == 'LR':
  model = LogisticRegression(input_dim,outputs)

lr = 0.0001 # 0.00005
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

criterion = nn.CrossEntropyLoss()

#From Aiken
def compute_accuracy_AK(loader, net, device):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    net.eval()
    net.to(device)
    accuracy_compute_history = []
    with torch.no_grad():      
        for data in loader:
            samples, labels = data
            samples = samples.to(device)
            outputs = net(samples)
            outputs = torch.max(F.softmax(outputs, dim=1), 1)[1]         
            for i in range(labels.shape[0]):
                sample_val = labels[i]
                predict_val= outputs[i]
                if sample_val == 1:
                    if predict_val>0.5:
                        TP = TP + 1
                    else:
                        FN = FN + 1
                elif sample_val == 0:
                    if predict_val <= 0.5:
                        TN = TN + 1
                    else:
                        FP = FP + 1
    recall = 0
    specificity = 0
    precision = 0
    accuracy = 0
    f1 = 0
    mcc = 0    
    if (TP+FN) != 0:
        recall = TP/(TP+FN) # sensitivity
    else:
        recall = 0
    if (TN+FP) != 0:
        specificity = TN/(TN+FP)
    else:
        specificity = 0
    if (TP+FP) != 0:
        precision = TP/(TP+FP)
    else:
        precision = 0
    if (TP+TN+FP+FN) != 0:    
        accuracy = 100*(TP+TN)/(TP+TN+FP+FN)
    else:
        accuracy = 0
    if (precision+recall) != 0:
        f1 = (2*precision*recall)/(precision+recall)
    else:
        f1 = 0
    # Matthews correlation coefficient
    if ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) == 0):
        mcc = 0
    else:
        mcc = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) 
    
    accuracy_compute_history.append(
        {"Accuracy":accuracy,
         "Precision":precision,
         "Recall":recall,
         "F1-score":f1,
         "MCC":mcc,
         "Specificity":specificity,# def compute_accuracy(loader, net):
         "TP": TP, "TN": TN, "FP":FP, "FN": FN
         }
    )
    return accuracy_compute_history

def fit():
  train_history = []
  val_history = []
  loss_history = []
  for epoch in range(30):  # loop over the dataset multiple times
    running_loss = 0.0; running_loss_g = 0.0
    for i, data in enumerate(train_loader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
      if model_type == 'CNN':
        inputs = inputs.view(inputs.size(0), 1, inputs.size(1))
      labels = labels.type(torch.LongTensor)

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs)

      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()  # Update the parameters of the model

      # print statistics
      running_loss += loss.item(); running_loss_g += loss.item()
      if i % 100 == 99: # print every 100 mini-batches
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
        running_loss = 0.0

    train_loss = running_loss_g/len(train_loader)
    loss_history.append( {"loss": train_loss, "set": "train", "epochs": epoch} )
    train_result = compute_accuracy_f1(train_loader, model)
    train_history.append({'epoch':train_result})
    #on validation data
    val_result = compute_accuracy_f1(val_loader, model)
    val_history.append({'epoch':val_result})
    loss_history.append( {"loss": val_result[4], "set": "val", "epochs": epoch} )
  return [train_history, val_history, loss_history]
  print('Finished Training')

train_results, val_results, loss_history = fit()

trained_model_file = F'/content/gdrive/MyDrive/PhD/Coursework/AI in Genomics/Projects/Phyla/phyla_dataset_d3/trained_model_LR_Stool.pth'
torch.save(model.state_dict(), trained_model_file)
model.load_state_dict(torch.load(trained_model_file))

train_accuracies = [result['epoch'][0] for result in train_results]
plt.plot(train_accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Train Accuracy vs. No. of epochs');

val_accuracies = [result['epoch'][0] for result in val_results]
plt.plot(val_accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Val Accuracy vs. No. of epochs');

#Plotting training and validation losses
loss_history = pd.DataFrame(loss_history)
print(loss_history.head())
ax = sns.lineplot(x="epochs", y="loss", hue= "set", data=loss_history)

#Model evaluation using Test data
#All
file3 = 'phyla_all_753x1177_pmi_0_clr_15p_MAD.csv'

 #stool
#file3 = 'phyla_stool_540x1177_pmi_0_clr_15p_MAD.csv'

 #Biopsy
#file3 = 'phyla_biopsy_213x1177_pmi_0_clr_15p_MAD.csv'

test_data_file = dirc + file3

class MicrobDataset(Dataset):
  """
  Dataset for binary classification Tumor/Normal
  """
  def __init__(self):
    
    # Select rows whose type is Tumor or Normal
    self.labels = phenotype_df[phenotype_df["uc_cd"].apply(lambda s: s == "IBD" or s == "Control")]

    # Compute categorical embedding, 0 is Normal, 1 is Tumor
    self.labels = self.labels["uc_cd"].apply(lambda s: s == "Control").astype(int)

    # Get corresponding gene expression profiles
    self.X = expr_df

  def __getitem__(self, index):
    sample = np.array(self.X.iloc[index], dtype=np.float32)
    label = np.array(self.labels.iloc[index], dtype=np.float32)

    return sample, label

  def __len__(self):
    return len(self.labels)

#Reading test data to evaluate the model on test data
#Pandas data read
if test_data_file.endswith('tsv'):
  microbe_df = pd.read_csv(test_data_file, sep='\t')
elif test_data_file.endswith('csv'):
  microbe_df = pd.read_csv(test_data_file)
else:
  print('Please make sure file is either tab separated or comma separated')

#We get indeces of metadata and expression 
metadata_size = 7
col_length = microbe_df.shape[1]
count_colms = col_length - metadata_size
metadata_start = count_colms + 1

#3 Now, we get the features and metadata
expr_df = microbe_df.iloc[:,1:count_colms]
phenotype_df_neat = microbe_df.iloc[:,metadata_start:col_length]
phenotype_df = phenotype_df_neat

#We can set 'sample_title' as index by:
phenotype_df.set_index('sample_title')

#Let's covert UC and CD to IBD
phenotype_df.loc[(phenotype_df.uc_cd == 'CD'), 'uc_cd'] = 'IBD'
phenotype_df.loc[(phenotype_df.uc_cd == 'UC'), 'uc_cd'] = 'IBD'

dataset = MicrobDataset()
num_examples, num_genes = dataset.X.shape

#Setting loaders for test data
test_dataset_size = int(len(dataset) * 1)
test_dataset, yx = torch.utils.data.random_split(dataset, lengths=[test_dataset_size, (0)], generator=torch.Generator().manual_seed(seed))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
len(dataset)

scores = compute_accuracy_f1(test_loader, model)
print('Accuracy = %s, F1-score = %s, Precision = %s, Recall = %s, MCC = %s' %(str(scores['Accuracy']),str(scores[1]),str(scores['F1-score']),str(scores['Precision']),str(scores['Recall'])))
















