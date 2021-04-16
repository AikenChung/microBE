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
import DANN
from pytorchtools import EarlyStopping
from NN_functions import write_result, save_model, compute_accuracy_DANN, test_model_DANN

#================================== Setting ==================================
base_path = './'
usingGoogleCloud = False # if using the local machine, please set 'usingGoogleCloud' to False

if usingGoogleCloud :
    base_path = '/content/gdrive/My Drive/Colab Notebooks/'

# Define the input and validate file
source_data_file = base_path+'phyla_stool_2840x1177_pmi_0_clr_75p_MAD.csv'
target_data_file = base_path+'phyla_biopsy_1109x1177_pmi_0_clr_75p_MAD.csv'
source_validate_file = base_path+'phyla_stool_401x1177_pmi_0_clr_10p_MAD.csv'
target_validate_file = base_path+'phyla_biopsy_164x1177_pmi_0_clr_10p_MAD.csv'


# Specify the filename tag
source_data_prefix = 'phyla_stool'
source_data_surfix_BE_method = 'no_BE'
target_data_prefix = 'phyla_biopsy'
target_data_surfix_BE_method = 'no_BE'


# Define the filenames for tesing. There are 3 testing files specified here.
testing_file_1 = base_path+'phyla_all_753x1177_pmi_0_clr_15p_MAD.csv'
testing_file_2 = base_path+'phyla_biopsy_213x1177_pmi_0_clr_15p_MAD.csv'
testing_file_3 = base_path+'phyla_stool_540x1177_pmi_0_clr_15p_MAD.csv'

# Specifiy the parameters of the DANN model
args = easydict.EasyDict({
        "feature_Num": 1177,             # Number of features (columns) in the input data
        "epochs": 5000,                  # Number of iterations to train Model for
        "hidden_dim": 512,               # Size of each hidden layer in DANN
        "dann_hidden_layers_num": 1,     # How many (middle or hidden) layers in DANN
        "feature_layer_size": 256,       # Size of feature_layer in the end of feature_extractor
        "hidden_dim_2nd": 256,           # Size of each hidden layer in DANN
        "dann_2nd_hidden_layers_num": 2, # How many (middle or hidden) layers in DANN
        "pre_output_layer_dim": 128,     # Size of pre-output layer in DANN
        "output_dim": 1,                 # Size of output layer
        "hidden_dropout": 0.5,           # dropout rate of hidden layer  
        "batch_size": 32,                # Batch size
        "learning_rate": 0.0001,         # Learning rate for the optimizer
        "beta1": 0.5,                    # 'beta1' for the optimizer
        "adapt_lr_iters": 10,            # how often decrease the learning rate
})

# Construct the filename for the model
fileNameToSave_base = ('DANN_'+ str(args.feature_Num) +'_'+ 
                               str(args.hidden_dim) + 'x' + 
                               str(args.dann_hidden_layers_num) + '_' +
                               str(args.feature_layer_size) + '_' +
                               str(args.pre_output_layer_dim) + '_' +
                               str(args.output_dim) + '_epoch'+
                               str(args.epochs) + '_dropout_'+
                               str(args.hidden_dropout).replace('.','p')+'_'+
                               source_data_prefix + '_' +
                               source_data_surfix_BE_method + '_' +
                               target_data_prefix + '_' +
                               target_data_surfix_BE_method )

# Create folders for results
if not os.path.exists(base_path+'data'):
    os.makedirs(base_path+'data')

if not os.path.exists(base_path+'data/DANN_trainedModels'):
    os.makedirs(base_path+'data/DANN_trainedModels')

if not os.path.exists(base_path+'data/DANN_trainedResults'):
    os.makedirs(base_path+'data/DANN_trainedResults')
    
modelFilePath = base_path+'data/DANN_trainedModels/'
resultFilePath = base_path+'data/DANN_trainedResults/'

#============================== End of Setting ================================


class PhylaDataset(Dataset):
    """ 
    Phyla dataset
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
    

def training(model, loader_S, loader_T, optimizer, criterion, device, epoch, nb_epoch):
    running_loss = 0.0
    len_dataloader = min(len(loader_S), len(loader_T))
    epoch_num = epoch
    nb_epoch_num = nb_epoch
    for i in range(len_dataloader):
        # Calculateing lambda_ (meta-paramet for gradient reversal layer)
        p = float(i + epoch_num * len_dataloader) / nb_epoch_num / len_dataloader
        lambda_ = 2. / (1. + np.exp(-10 * p)) - 1
        
        # train the model using source data
        train_data_S = loader_S.next()
        s_data, s_label = train_data_S
       
        # zero the parameter gradients
        optimizer.zero_grad()

        # Define zero for source domain data
        domain_label = torch.zeros_like(s_label)
        
        # put data in the device of 'cuda' or 'cpu'.
        s_data = s_data.to(device)
        #s_label = s_label.squeeze_().long()
        s_label = s_label.to(device)
        
        # Input source data into DANN model
        class_predict, domain_predict = model(s_data, lambda_=lambda_)
        
        #print('class_predict.view(-1).shape: ', class_predict.view(-1).shape)
        err_s_label = criterion(class_predict.to(device), s_label.to(device))
        err_s_domain = criterion(domain_predict.to(device), domain_label.to(device))
        
        # train the model using target data
        train_data_T = loader_T.next()
        t_data, t_label = train_data_T

        # Define one for source domain data
        domain_label = torch.ones_like(t_label.to(device))
        # Input target data into DANN model
        class_predict_t, domain_predict = model(t_data.to(device), lambda_=lambda_)
        err_t_domain = criterion(domain_predict.to(device), domain_label.to(device))

        err_t_label = criterion(class_predict_t.to(device), t_label.to(device))        
        
        # backward + optimize
        loss = err_t_label + err_s_label + err_s_domain + err_t_domain # Include the target_label_loss
        #loss = err_s_label + err_s_domain + err_t_domain
        loss.to(device)
        loss.backward()
        optimizer.step()  # Update the parameters of the model
            
        running_loss += loss.item()         
        return running_loss/len_dataloader
       
    print('Finished Training')


def validating(model, data_loader, criterion, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
      pass_loss = 0.0
      for samples, labels in data_loader:
          samples = samples.to(device)
          labels = labels.to(device)
          class_output, _ = model(samples, 1)
          loss = criterion(class_output, labels)
          pass_loss += loss.item()
      
      return pass_loss/len(data_loader)


def run_training_process(model, nb_epochs, dataloader_source, 
                         dataloader_target, validate_S_loader, validate_T_loader, 
                         optimizer, scheduler, criterion, fileNameForModel, 
                         device, patience ):
    #Subjecting the define NN model to the device which can be CPU or GPU
    model = model.to(device)
    progress_bar = tqdm(range(nb_epochs))
    loss_history = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    # This extra variable 'ind' is the same as the 'epoch' but with the 'int' datatype.
    # Passing this variable into 'training' for calculating parameter 'lambda'
    ind = 0
    for epoch in progress_bar:
        source_data_iter = iter(dataloader_source)
        target_data_iter = iter(dataloader_target)
        train_loss = training(model, source_data_iter, target_data_iter, 
                            optimizer, criterion, device, ind, nb_epochs)
        ind = ind + 1
        validate_S_loss = validating(model, validate_S_loader, criterion, device)
        validate_T_loss = validating(model, validate_T_loader, criterion, device)
        test_loss = validate_S_loss + validate_T_loss
        loss_history.append(
            {"loss": train_loss, "set": "train-S", "epochs": epoch}
        )
        loss_history.append(
            {"loss": validate_S_loss, "set": "validate-S", "epochs": epoch}
        )
        loss_history.append(
            {"loss": validate_T_loss, "set": "validate-T", "epochs": epoch}
        )
        loss_history.append(
            {"loss": test_loss, "set": "validate-S+T", "epochs": epoch}
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


# starting time
start = time.time()

# sets device for model and PyTorch tensors
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initilize phyla dataset
train_S_dataset = PhylaDataset(source_data_file)
train_T_dataset = PhylaDataset(target_data_file)

validate_S_dataset = PhylaDataset(source_validate_file)
validate_T_dataset = PhylaDataset(target_validate_file)

"""
Start to train the DANN model
"""
# Initilize model, criterion, optimizer. Then train the model
classifier_DANN = DANN.DANN(args.feature_Num, args.hidden_dim, 
                         args.dann_hidden_layers_num, args.feature_layer_size,
                         args.hidden_dim_2nd, args.dann_2nd_hidden_layers_num,
                         args.pre_output_layer_dim, args.output_dim, args.hidden_dropout)


# early stopping patience; how long to wait after last time validation loss improved.
patience = 20

"""
Initialize dataloader objects. 
These dataloaders will provide data one batch at a time, which is convenient to train our machine learning model.
"""

train_S_loader = DataLoader(train_S_dataset, batch_size = args.batch_size, shuffle=True)
validate_S_loader = DataLoader(validate_S_dataset, batch_size = args.batch_size, shuffle=True)

train_T_loader = DataLoader(train_T_dataset, batch_size = args.batch_size, shuffle=True)
validate_T_loader = DataLoader(validate_T_dataset, batch_size = args.batch_size, shuffle=True)
# cost function (for predicting labels)
criterion = nn.BCEWithLogitsLoss()

# setup optimizer
optimizer_dann = optim.Adam(list(classifier_DANN.parameters()), 
                            lr=args.learning_rate, betas=(args.beta1, 0.999))
# use an exponentially decaying learning rate
scheduler_dann= optim.lr_scheduler.ExponentialLR(optimizer_dann, gamma=0.99)

#Specify a filename for saving the model
modelNameToSave = modelFilePath + fileNameToSave_base

# Start the training procedure
training_history = run_training_process(classifier_DANN, args.epochs, 
                                        train_S_loader, train_T_loader, 
                                        validate_S_loader, validate_T_loader, 
                                        optimizer_dann, scheduler_dann, 
                                        criterion, modelNameToSave,
                                        device=device, patience=patience )

# Compute the training and validating related performance metrics
train_metric_result_nameToSave = resultFilePath + fileNameToSave_base + "_train_result_metric.txt"
train_S_metric = compute_accuracy_DANN(train_S_loader, classifier_DANN, device)
validation_S_metric = compute_accuracy_DANN(validate_S_loader, classifier_DANN, device)
train_T_metric = compute_accuracy_DANN(train_T_loader, classifier_DANN, device)
validation_T_metric = compute_accuracy_DANN(validate_T_loader, classifier_DANN, device)

# Save the training and validating related results
write_result(train_metric_result_nameToSave, train_S_metric, fileNameToSave_base, source_data_file, args)
write_result(train_metric_result_nameToSave, validation_S_metric, fileNameToSave_base, source_data_file, args)
write_result(train_metric_result_nameToSave, train_T_metric, fileNameToSave_base, target_data_file, args)
write_result(train_metric_result_nameToSave, validation_T_metric, fileNameToSave_base, target_data_file, args)
        
print('Accuracy of the DANN on the ' + str(len(train_S_dataset)) + ' train samples: %d %%' % train_S_metric[0]["Accuracy"])
print('Accuracy of the DANN on the ' + str(len(validate_S_dataset)) + ' validation samples: %d %%' % validation_S_metric[0]["Accuracy"])
print('Accuracy of the DANN on the ' + str(len(train_T_dataset)) + ' target samples: %d %%' % train_T_metric[0]["Accuracy"])
print('Accuracy of the DANN on the ' + str(len(validate_T_dataset)) + ' target validation samples: %d %%' % validation_T_metric[0]["Accuracy"])
training_history.head()

# Plot the training and validaing loss
plt.figure()
ax = sns.lineplot(x="epochs", y="loss", hue= "set", data=training_history)
fig_trainHistory = ax.get_figure()
training_history_plotName = resultFilePath + fileNameToSave_base +'_training_history.png'
fig_trainHistory.savefig(training_history_plotName)


"""
Run the testing procedure.
"""
# Testing for mixed data source (stool+biopsy) file
input_testing_filename = testing_file_1
test_result_filename_toSave = resultFilePath + fileNameToSave_base + "_test_result_metric.txt"
test_dataset = PhylaDataset(input_testing_filename)
test_loader = DataLoader(test_dataset, 
                          batch_size = args.batch_size, 
                          shuffle=True)               
test_model_DANN(input_testing_filename, test_loader, test_result_filename_toSave,
           fileNameToSave_base, classifier_DANN, args, device)

# Testing for biopsy data source file
input_testing_filename = testing_file_2
test_result_filename_toSave = resultFilePath + fileNameToSave_base + "_test_result_metric.txt"
test_dataset = PhylaDataset(input_testing_filename)
test_loader = DataLoader(test_dataset, 
                          batch_size = args.batch_size, 
                          shuffle=True)               
test_model_DANN(input_testing_filename, test_loader, test_result_filename_toSave,
           fileNameToSave_base, classifier_DANN, args, device)

# Testing for stool data source file
input_testing_filename = testing_file_3
test_result_filename_toSave = resultFilePath + fileNameToSave_base + "_test_result_metric.txt"
test_dataset = PhylaDataset(input_testing_filename)
test_loader = DataLoader(test_dataset, 
                          batch_size = args.batch_size, 
                          shuffle=True)               
test_model_DANN(input_testing_filename, test_loader, test_result_filename_toSave,
           fileNameToSave_base, classifier_DANN, args, device)

# end time
end = time.time()
totalSeconds = round(end - start)
print(f"Runtime of the program is {totalSeconds} seconds")