#!/usr/bin/env python3

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim
import numpy as np
import pandas as pd
#progess bar
from tqdm import tqdm
import time
import os as os
import easydict
import MLP
# import EarlyStopping
from pytorchtools import EarlyStopping

#================================== Setting ==================================
base_path = './'
usingGoogleCloud = False # if using the local machine, please set 'usingGoogleCloud' to False

if usingGoogleCloud :
    base_path = '/content/gdrive/My Drive/Colab Notebooks/'

train_base_file = base_path+'phyla_stool_2840x1177_pmi_0_clr_75p_MAD.csv'
validate_base_file = base_path+'phyla_stool_401x1177_pmi_0_clr_10p_MAD.csv'
#train_data_prefix = 'phyla_stool_noCS'
train_data_prefix = 'phyla_stool'
train_data_surfix_BE_method = 'no_BE'

testing_file_1 = base_path+'phyla_all_753x1177_pmi_0_clr_15p_MAD.csv'
testing_file_2 = base_path+'phyla_biopsy_213x1177_pmi_0_clr_15p_MAD.csv'
testing_file_3 = base_path+'phyla_stool_540x1177_pmi_0_clr_15p_MAD.csv'
#testing_file_3 = base_path+'phyla_stool_541x1177_PMI_threshold_0_clr_15p.csv'

args = easydict.EasyDict({
        "feature_Num": 1177,        # Number of features (columns) in the input data
        "epochs": 5000,             # Number of iterations to train Model for
        "hidden_dim": 256,          # Size of each hidden layer in Discriminator
        "mlp_hidden_layers_num": 1, # How many (middle or hidden) layers in Discriminator
        "pre_output_layer_dim": 128,# Size of each hidden layer in Discriminator
        "output_dim": 1,            # Size of output layer
        "hidden_dropout": 0.5,      # dropout rate of hidden layer  
        "batch_size": 32,           # Batch size
        "learning_rate": 0.0001,    # Learning rate for the optimizer
        "beta1": 0.5,               # 'beta1' for the optimizer
        "adapt_lr_iters": 5,        # how often decrease the learning rate
})


fileNameToSave_base = ('MLP_'+ str(args.feature_Num) +'_'+ 
                               str(args.hidden_dim) + 'x' +
                               str(args.mlp_hidden_layers_num) + '_' +
                               str(args.pre_output_layer_dim) + '_' +
                               str(args.output_dim) + '_Adam_lr_'+
                               str(args.learning_rate).replace('.','p') + 
                               '_BCEWithLogitsLoss_bSize'+
                               str(args.batch_size) + '_epoch'+
                               str(args.epochs) + '_dropout_'+
                               str(args.hidden_dropout).replace('.','p')+'_'+
                               train_data_prefix+'_'+
                               train_data_surfix_BE_method)

if not os.path.exists(base_path+'data'):
    os.makedirs(base_path+'data')

if not os.path.exists(base_path+'data/MLP_trainedModels'):
    os.makedirs(base_path+'data/MLP_trainedModels')

if not os.path.exists(base_path+'data/MLP_trainedResults'):
    os.makedirs(base_path+'data/MLP_trainedResults')
    
modelFilePath = base_path+'data/MLP_trainedModels/'
resultFilePath = base_path+'data/MLP_trainedResults/'

#============================== End of Setting ================================


class PhylaDataset(Dataset):
    """ Phyla dataset"""
    """
    Dataset for binary classification IBD/Healthy
    """
    # Initialize your data, download, etc.
    def __init__(self, inputFile):
        ori_data = pd.read_csv(inputFile)
        phyla_input = ori_data[ori_data.columns[1:args.feature_Num+1]]
        phyla_input = phyla_input.assign(diagnosis=ori_data[ori_data.columns[args.feature_Num+2]])
        phyla_input = phyla_input.to_numpy(dtype=np.float32)
        self.len = phyla_input.shape[0]
        self.count_data = from_numpy(phyla_input[:, 0:-1])
        self.diagnosis_data = from_numpy(phyla_input[:, [-1]]) # 0: Control, 1: IBD

    def __getitem__(self, index):
        samples = self.count_data[index]
        labels = self.diagnosis_data[index]
        return samples, labels

    def __len__(self):
        return self.len
    

def training(model, loader, optimizer, criterion, device):
    running_loss = 0.0
    for i, data in enumerate(loader,0):
      # get the inputs; data is a list of [samples, labels]
      samples, labels = data
      
      # into the defined NN model in that device.
      samples = samples.to(device)
      labels = labels.to(device)
      
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(samples)
      loss = criterion(outputs, labels)
      loss.to(device)
      loss.backward()
      # Update the parameters of the model
      optimizer.step()  
          
      running_loss += loss.item()         
      return running_loss/len(loader)
       
    print('Finished Training')


def validating(model, data_loader, criterion, device):
    model.eval()
    with torch.no_grad():
      pass_loss = 0.0
      for samples, labels in data_loader:
          samples = samples.to(device)
          labels = labels.to(device)
          outputs = model(samples)
          loss = criterion(outputs, labels)
          pass_loss += loss.item()
    
    return pass_loss/len(data_loader)


def run_training_process(model, nb_epochs, train_loader, validate_loader, optimizer, scheduler, criterion, fileNameForModel, device, patience):
  #Subjecting the define NN model to the device which can be CPU or GPU
  model = model.to(device)
  progress_bar = tqdm(range(nb_epochs), position=0, leave=True)
  loss_history = []
  # initialize the early_stopping object
  early_stopping = EarlyStopping(patience=patience, verbose=True)
  for epoch in progress_bar:
      train_loss = training(model, train_loader, optimizer, criterion, device)
      test_loss = validating(model, validate_loader, criterion, device)
      validation_dataset_metric = compute_accuracy(validate_loader, model)
      loss_history.append(
          {"loss": train_loss, "set": "train", "epochs": epoch}
      )
      loss_history.append(
          {"loss": test_loss, "set": "validate", "epochs": epoch}
      )
      loss_history.append(
          {"loss": np.round(validation_dataset_metric[0]["Accuracy"]/100, decimals=2) 
           , "set": "Accuracy", "epochs": epoch}
      )
      loss_history.append(
          {"loss": np.round(validation_dataset_metric[0]["F1-score"], decimals=2) 
           , "set": "F1-score", "epochs": epoch}
      )
      # save models during each training iteration
      checkpoint = {'model' : model.state_dict(), 'optimizer': optimizer.state_dict()}
      fileNameToSave = fileNameForModel + ".pt"
      if epoch % args.adapt_lr_iters == 0 :
          save_model(checkpoint,fileNameToSave)
          # Using scheduler to update the learning rate every 100 iterations.
          scheduler.step()
      # early_stopping needs the validation loss to check if it has decresed, 
      # and if it has, it will make a checkpoint of the current model
      early_stopping(test_loss, model)
      if early_stopping.early_stop:
            print('')
            print("Early stopping")
            save_model(checkpoint,fileNameToSave)
            break
      
  return pd.DataFrame(loss_history)


def save_model(model_state, fileName):
    #print("=> Saving model")
    torch.save(model_state,fileName)

"""### Define the evaluation metric

We will use several evaluation metrics.

Accuracy = (TP+TN) / (TP+TN+FP+FN)
Specificity = TN / (TN+FP)
Recall (Sensitivity) = (TP) / (TP+FN)
Precision = TP / (TP+FP)
F1-score = (2*Precision*Recall) / (Precision+Recall)
MCC = (TP*TN - FP*FN) / sqrt((TP+FN)*(TP+FP)*(TN+FN)*(TN+FP))
"""
def compute_accuracy(loader, net):
    accuracy_compute_history = []
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    net.eval()
    with torch.no_grad():      
        for data in loader:
            samples, labels = data
            samples = samples.to(device)
            outputs = net(samples)           
            for i in range(labels.shape[0]):
                sample_val = labels[i,0]
                predict_val= outputs[i,0]
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
         "Specificity":specificity,
         "TP": TP, "TN": TN, "FP":FP, "FN": FN
         }
    )
    return accuracy_compute_history

def write_result(fileName, dataObj, modelName, testingFileName):
    with open(fileName, 'w') as f:
        theFirstLine = 'Model file: '+modelName+'\n'
        f.write(theFirstLine)
        theSecondLine = 'Test file: '+testingFileName+'\n'
        f.write(theSecondLine)
        for item in dataObj[0]:
            strToWrite = "{0}: {1}\n".format(item, np.round(dataObj[0][item], decimals=2))
            f.write(strToWrite)

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()
    

# sets device for model and PyTorch tensors
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Preparing data for training and validaing
"""
train_dataset = PhylaDataset(train_base_file)
validate_dataset = PhylaDataset(validate_base_file)


"""
We now need to split our dataset into two parts.
The **train set** will be used to train our model, and the **validate set** will be used for validation.
First, let us compute the number of samples to put in each split. Here we choose to keep 70\% of the samples for training and 30\% for testing
"""
# starting time
start = time.time()

# Initilize model, criterion, optimizer. Then train the model for multiclass classification
classifierMLP = MLP(args.feature_Num, 
                             args.hidden_dim, 
                             args.mlp_hidden_layers_num, 
                             args.pre_output_layer_dim, 
                             args.output_dim,
                             args.hidden_dropout)

# early stopping patience; how long to wait after last time validation loss improved.
patience = 50

# Define data loaders for training and validating data
train_loader = DataLoader(train_dataset, 
                          batch_size=args.batch_size, 
                          shuffle=True)
validate_loader = DataLoader(validate_dataset,
                             batch_size=args.batch_size, 
                             shuffle=True)    

"""
Start to train the MLP model
"""
# cost function (for predicting labels)
criterion = nn.BCEWithLogitsLoss()

# setup optimizer
optimizer_mlp = optim.Adam(list(classifierMLP.parameters()), 
                           lr=args.learning_rate, betas=(args.beta1, 0.999))
# use an exponentially decaying learning rate
scheduler_mlp= optim.lr_scheduler.ExponentialLR(optimizer_mlp, gamma=0.99)

fileNameToSave_base = fileNameToSave_base
modelNameToSave = modelFilePath + fileNameToSave_base
training_history = run_training_process(classifierMLP, args.epochs, train_loader, validate_loader, optimizer_mlp, scheduler_mlp, criterion, modelNameToSave, device=device , patience=patience)

train_dataset_metric = compute_accuracy(train_loader, classifierMLP)
train_dataset_metric_nameToSave = resultFilePath + fileNameToSave_base + "_train_result_metric.txt"
validation_dataset_metric = compute_accuracy(validate_loader, classifierMLP)
validation_dataset_metric_nameToSave = resultFilePath + fileNameToSave_base + "_validation_result_metric.txt"

write_result(train_dataset_metric_nameToSave, train_dataset_metric, fileNameToSave_base, train_base_file)
validationFileName = train_base_file[0:len(train_base_file)-4]+"_validation"
write_result(validation_dataset_metric_nameToSave, validation_dataset_metric, fileNameToSave_base, validationFileName)
        
print('Accuracy of the MLP on the ' + str(len(train_loader)) + ' train samples: %d %%' % (train_dataset_metric[0]["Accuracy"]))

print('Accuracy of the MLP on the ' + str(len(validate_loader)) + ' validation samples: %d %%' % (validation_dataset_metric[0]["Accuracy"]))

training_history.head()

plt.figure()
ax = sns.lineplot(x="epochs", y="loss", hue= "set", data=training_history)
fig_trainHistory = ax.get_figure()
training_history_plotName = resultFilePath + fileNameToSave_base +'_training_history.png'
fig_trainHistory.savefig(training_history_plotName)


"""
Run the testing procedure.
"""
# Initiate a dataloader of the testing file
fileToTestModel = testing_file_1
test_dataset = PhylaDataset(fileToTestModel)
test_loader = DataLoader(test_dataset, 
                         batch_size = args.batch_size, 
                         shuffle=True)
# Test the loaded model
test_dataset_metric = compute_accuracy(test_loader, classifierMLP.to(device))
# Save the testing metrics to a text file
modelFileName_toSave = fileNameToSave_base
test_dataset_metric_nameToSave = resultFilePath + fileNameToSave_base + "_test_result_metric.txt"
write_result(test_dataset_metric_nameToSave, test_dataset_metric, 
             modelFileName_toSave, fileToTestModel)
print('')
print(str(args.feature_Num) +'_'+ 
        str(args.hidden_dim) + 'x' +
        str(args.mlp_hidden_layers_num) + '_' +
        str(args.pre_output_layer_dim) + '_' +
        str(args.output_dim))
print(train_base_file)
print(fileToTestModel)
print('Accuracy:', np.round(test_dataset_metric[0]['Accuracy'], decimals=2), '%')
print('Precision:', np.round(test_dataset_metric[0]['Precision'], decimals=2))
print('Recall:', np.round(test_dataset_metric[0]['Recall'], decimals=2))
print('F1-score:', np.round(test_dataset_metric[0]['F1-score'], decimals=2))
print('MCC:', np.round(test_dataset_metric[0]['MCC'], decimals=2),'\n')

# Initiate a dataloader of the testing file
fileToTestModel = testing_file_2
test_dataset = PhylaDataset(fileToTestModel)
test_loader = DataLoader(test_dataset, 
                         batch_size = args.batch_size, 
                         shuffle=True)
# Test the loaded model
test_dataset_metric = compute_accuracy(test_loader, classifierMLP.to(device))
# Save the testing metrics to a text file
modelFileName_toSave = fileNameToSave_base
test_dataset_metric_nameToSave = resultFilePath + fileNameToSave_base + "_test_result_metric.txt"
write_result(test_dataset_metric_nameToSave, test_dataset_metric, 
             modelFileName_toSave, fileToTestModel)
print('')
print(train_base_file)
print(fileToTestModel)
print('Accuracy:', np.round(test_dataset_metric[0]['Accuracy'], decimals=2), '%')
print('Precision:', np.round(test_dataset_metric[0]['Precision'], decimals=2))
print('Recall:', np.round(test_dataset_metric[0]['Recall'], decimals=2))
print('F1-score:', np.round(test_dataset_metric[0]['F1-score'], decimals=2))
print('MCC:', np.round(test_dataset_metric[0]['MCC'], decimals=2),'\n')

# Initiate a dataloader of the testing file
fileToTestModel = testing_file_3
test_dataset = PhylaDataset(fileToTestModel)
test_loader = DataLoader(test_dataset, 
                         batch_size = args.batch_size, 
                         shuffle=True)
# Test the loaded model
test_dataset_metric = compute_accuracy(test_loader, classifierMLP.to(device))
# Save the testing metrics to a text file
modelFileName_toSave = fileNameToSave_base
test_dataset_metric_nameToSave = resultFilePath + fileNameToSave_base + "_test_result_metric.txt"
write_result(test_dataset_metric_nameToSave, test_dataset_metric, 
             modelFileName_toSave, fileToTestModel)
print('')
print(train_base_file)
print(fileToTestModel)
print('Accuracy:', np.round(test_dataset_metric[0]['Accuracy'], decimals=2), '%')
print('Precision:', np.round(test_dataset_metric[0]['Precision'], decimals=2))
print('Recall:', np.round(test_dataset_metric[0]['Recall'], decimals=2))
print('F1-score:', np.round(test_dataset_metric[0]['F1-score'], decimals=2))
print('MCC:', np.round(test_dataset_metric[0]['MCC'], decimals=2),'\n')


