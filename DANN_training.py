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
import phylaDANN
from sklearn.model_selection import KFold
# import EarlyStopping
from pytorchtools import EarlyStopping

#================================== Setting ==================================
base_path = './'
usingGoogleCloud = True # if using the local machine, please set 'usingGoogleCloud' to False

if usingGoogleCloud :
    base_path = '/content/gdrive/My Drive/Colab Notebooks/'

source_data_file = base_path+'phyla_stool_3240x1177_PMI_threshold_0_clr_85p.csv'
source_data_prefix = 'phyla_stool'
source_data_surfix_BE_method = 'no_BE'

target_data_file = base_path+'phyla_biopsy_1273x1177_PMI_threshold_0_clr_85p.csv'
target_data_prefix = 'phyla_biopsy'
target_data_surfix_BE_method = 'no_BE'

testing_file_s = base_path+'phyla_stool_541x1177_PMI_threshold_0_clr_15p.csv'
testing_file_t = base_path+'phyla_biopsy_213x1177_PMI_threshold_0_clr_15p.csv'
testing_file_all = base_path+'phyla_all_753x1177_PMI_threshold_0_clr_15p.csv'

args = easydict.EasyDict({
        "feature_Num": 1177,        # Number of features (columns) in the input data
        "epochs": 3000,              # Number of iterations to train Model for
        "hidden_dim": 512,          # Size of each hidden layer in DANN
        "dann_hidden_layers_num": 1,# How many (middle or hidden) layers in DANN
        "feature_layer_size": 512,   # Size of feature_layer in the end of feature_extractor
        "hidden_dim_2nd": 256,          # Size of each hidden layer in DANN
        "dann_2nd_hidden_layers_num": 1,# How many (middle or hidden) layers in DANN
        "pre_output_layer_dim": 128, # Size of pre-output layer in DANN
        "output_dim": 1,            # Size of output layer      
        "batch_size": 32,           # Batch size
        "learning_rate": 0.0001,     # Learning rate for the optimizer
        "beta1": 0.5,               # 'beta1' for the optimizer
        "adapt_lr_iters": 10,        # how often decrease the learning rate
        "normalization_method":'Median' # Median, Stand, or minMax. Normalization method applied in the initailization of phyla dataset
})

fileNameToSave_base = ('DANN_'+ str(args.feature_Num) +'_'+ 
                               str(args.hidden_dim) + 'x' + 
                               str(args.dann_hidden_layers_num) + '_' +
                               str(args.feature_layer_size) + '_' +
                               str(args.pre_output_layer_dim) + '_' +
                               str(args.output_dim) + '_epoch'+
                               str(args.epochs) + '_' +
                               source_data_prefix + '_' +
                               source_data_surfix_BE_method + '_' +
                               target_data_prefix + '_' +
                               target_data_surfix_BE_method )


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
    def __init__(self, inputFile, norm_method):
        ori_data = pd.read_csv(inputFile)
        phyla_input = ori_data[ori_data.columns[1:args.feature_Num+1]]
        phyla_input = phyla_input.assign(diagnosis=ori_data[ori_data.columns[args.feature_Num+2]])
        phyla_input = phyla_input.to_numpy(dtype=np.float32)
        self.len = phyla_input.shape[0]
        self.count_data = from_numpy(phyla_input[:, 0:-1])
        self.diagnosis_data = from_numpy(phyla_input[:, [-1]]) # 0: Control, 1: IBD
        # feature-wise normalization
        self.count_data = self.normalization(self.count_data, norm_method)

    def normalization(self, inputTensor, method):
        # feature-wise normalization
        if method == 'Stand':
            # Standardization
            colMean = inputTensor.mean(0, keepdim=True)[0]
            colStd = inputTensor.std(0, keepdim=True)[0]
            outputTensor = (inputTensor - colMean) / colStd
        elif method == 'minMax':
            # Min-Max
            colMin = inputTensor.min(0, keepdim=True)[0]
            colMax = inputTensor.max(0, keepdim=True)[0]    
            outputTensor = (inputTensor - colMin) / (colMax - colMin)
        else:
            # Median Normalization
            colMedian = inputTensor.median(0, keepdim=True)[0]
            colMAD = torch.abs(inputTensor-colMedian)
            colMAD = colMAD.median(0, keepdim=True)[0]
            outputTensor = (inputTensor - colMedian) / colMAD
        return outputTensor

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

  with torch.no_grad():
    pass_loss = 0.0
    for samples, labels in data_loader:
        samples = samples.to(device)
        labels = labels.to(device)
        class_output, _ = model(samples, 1)
        loss = criterion(labels, class_output)
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

    with torch.no_grad():      
        for data in loader:
            samples, labels = data
            samples = samples.to(device)
            outputs, _ = net(samples,1)
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
    with open(fileName, 'a') as f:
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

# Initilize phyla dataset
soure_dataset = PhylaDataset(source_data_file, args.normalization_method)
num_cohorts, num_genus = soure_dataset.count_data.shape

target_dataset = PhylaDataset(target_data_file, args.normalization_method)

# starting time
start = time.time()

"""
Start to train the DANN model
"""
# Initilize model, criterion, optimizer. Then train the model
classifier_DANN = phylaDANN.DANN(args.feature_Num, args.hidden_dim, 
                         args.dann_hidden_layers_num, args.feature_layer_size,
                         args.hidden_dim_2nd, args.dann_2nd_hidden_layers_num,
                         args.pre_output_layer_dim, args.output_dim)

# Configuration options
k_folds = 5
  
# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)

# early stopping patience; how long to wait after last time validation loss improved.
patience = 5000

best_validate_accuracy = 0
best_fold = 0
import copy
best_model = copy.deepcopy(classifier_DANN)
# K-fold Cross Validation model evaluation
for i in range(k_folds):
    i = i + 1
    # Print
    print('')
    print('--------------------------------')
    print(f'FOLD {i}')
    print('--------------------------------')
    """
    Split the dataset into two parts.
    The **train set** will be used to train our model, and the **validate set** will be used for validation.
    First, let us compute the number of samples to put in each split. Here we choose to keep 70\% of the samples for training and 30\% for testing
    """
    kFold_ratio_num = 1 - (1/k_folds)
    train_set_size = int(len(soure_dataset) * kFold_ratio_num)
    validate_set_size = len(soure_dataset) - train_set_size
    
    target_set_size = int(len(target_dataset) * kFold_ratio_num)
    validate_T_set_size = len(target_dataset) - target_set_size
    
    """
    Randomly split the dataset into two parts for preparing data for training and validaing
    """
    
    train_S_dataset, validate_S_dataset = torch.utils.data.random_split(soure_dataset, 
                                                                lengths=[train_set_size, validate_set_size], 
                                                                generator=torch.Generator().manual_seed(0))
    train_T_dataset, validate_T_dataset = torch.utils.data.random_split(target_dataset, 
                                                                lengths=[target_set_size, validate_T_set_size], 
                                                                generator=torch.Generator().manual_seed(0))
    
    """
    Initialize dataloader objects. 
    These dataloaders will provide data one batch at a time, which is convenient to train our machine learning model.
    """
    
    train_S_loader = DataLoader(train_S_dataset, batch_size = args.batch_size, shuffle=True)
    validate_S_loader = DataLoader(validate_S_dataset, batch_size = args.batch_size, shuffle=True)
    
    train_T_loader = DataLoader(train_T_dataset, batch_size = args.batch_size, shuffle=True)
    validate_T_loader = DataLoader(validate_T_dataset, batch_size = args.batch_size, shuffle=True)
    classifier_DANN.apply(reset_weights)
    # cost function (for predicting labels)
    criterion = nn.BCEWithLogitsLoss()
    
    # setup optimizer
    optimizer_dann = optim.Adam(list(classifier_DANN.parameters()), 
                                lr=args.learning_rate, betas=(args.beta1, 0.999))
    # use an exponentially decaying learning rate
    scheduler_dann= optim.lr_scheduler.ExponentialLR(optimizer_dann, gamma=0.99)
    
    modelNameToSave = modelFilePath + fileNameToSave_base
    training_history = run_training_process(classifier_DANN, args.epochs, 
                                            train_S_loader, train_T_loader, 
                                            validate_S_loader, validate_T_loader, 
                                            optimizer_dann, scheduler_dann, 
                                            criterion, modelNameToSave,
                                            device=device, patience=patience )
    
    train_metric_result_nameToSave = resultFilePath + fileNameToSave_base + "_train_result_metric.txt"
    train_S_metric = compute_accuracy(train_S_loader, classifier_DANN)
    validation_S_metric = compute_accuracy(validate_S_loader, classifier_DANN)
    train_T_metric = compute_accuracy(train_T_loader, classifier_DANN)
    validation_T_metric = compute_accuracy(validate_T_loader, classifier_DANN)
    
    
    write_result(train_metric_result_nameToSave, train_S_metric, fileNameToSave_base, source_data_file)
    write_result(train_metric_result_nameToSave, validation_S_metric, fileNameToSave_base, source_data_file)
    write_result(train_metric_result_nameToSave, train_T_metric, fileNameToSave_base, target_data_file)
    write_result(train_metric_result_nameToSave, validation_T_metric, fileNameToSave_base, target_data_file)
            
    print('Accuracy of the DANN on the ' + str(len(train_S_dataset)) + ' train samples: %d %%' % train_S_metric[0]["Accuracy"])
    print('Accuracy of the DANN on the ' + str(len(validate_S_dataset)) + ' validation samples: %d %%' % validation_S_metric[0]["Accuracy"])
    print('Accuracy of the DANN on the ' + str(len(train_T_dataset)) + ' target samples: %d %%' % train_T_metric[0]["Accuracy"])
    print('Accuracy of the DANN on the ' + str(len(validate_T_dataset)) + ' target validation samples: %d %%' % validation_T_metric[0]["Accuracy"])
    training_history.head()
    
    if best_validate_accuracy < validation_T_metric[0]["Accuracy"]:
        best_validate_accuracy = validation_T_metric[0]["Accuracy"]
        best_fold = i
        best_model = copy.deepcopy(classifier_DANN)
    
    
    plt.figure()
    ax = sns.lineplot(x="epochs", y="loss", hue= "set", data=training_history)
    fig_trainHistory = ax.get_figure()
    training_history_plotName = resultFilePath + fileNameToSave_base +'_training_history.png'
    fig_trainHistory.savefig(training_history_plotName)


print('')
print('The best validation accuracy: ' + str(np.round(best_validate_accuracy, decimals=2)) + '% at fold-'+str(best_fold))

"""
Run the testing procedure.
"""
# Initiate a dataloader of the testing file
fileToTestModel = testing_file_s
test_dataset_s = PhylaDataset(fileToTestModel, 'Median')
test_loader = DataLoader(test_dataset_s, 
                         batch_size = args.batch_size, 
                         shuffle=True)
# Test the loaded model
test_dataset_metric = compute_accuracy(test_loader, best_model.to(device))
# Save the testing metrics to a text file
modelFileName_toSave = fileNameToSave_base + '_fold'+str(best_fold)
test_dataset_metric_nameToSave = resultFilePath + fileNameToSave_base + "_test_result_metric.txt"
write_result(test_dataset_metric_nameToSave, test_dataset_metric, 
             modelFileName_toSave, fileToTestModel)
print('')
print('Testing source domain data =>')
print(fileToTestModel)
print('Accuracy:', np.round(test_dataset_metric[0]['Accuracy'], decimals=2), '%')
print('Precision:', np.round(test_dataset_metric[0]['Precision'], decimals=2))
print('Recall:', np.round(test_dataset_metric[0]['Recall'], decimals=2))
print('F1-score:', np.round(test_dataset_metric[0]['F1-score'], decimals=2))
print('MCC:', np.round(test_dataset_metric[0]['MCC'], decimals=2),'\n')

# Initiate a dataloader of the testing file
fileToTestModel = testing_file_t
test_dataset_t = PhylaDataset(fileToTestModel, 'Median')
test_loader = DataLoader(test_dataset_t, 
                         batch_size = args.batch_size, 
                         shuffle=True)
# Test the loaded model
test_dataset_metric = compute_accuracy(test_loader, best_model.to(device))
# Save the testing metrics to a text file
modelFileName_toSave = fileNameToSave_base + '_fold'+str(best_fold)
test_dataset_metric_nameToSave = resultFilePath + fileNameToSave_base + "_test_result_metric.txt"
write_result(test_dataset_metric_nameToSave, test_dataset_metric, 
             modelFileName_toSave, fileToTestModel)
print('')
print('Testing target domain data =>')
print(fileToTestModel)
print('Accuracy:', np.round(test_dataset_metric[0]['Accuracy'], decimals=2), '%')
print('Precision:', np.round(test_dataset_metric[0]['Precision'], decimals=2))
print('Recall:', np.round(test_dataset_metric[0]['Recall'], decimals=2))
print('F1-score:', np.round(test_dataset_metric[0]['F1-score'], decimals=2))
print('MCC:', np.round(test_dataset_metric[0]['MCC'], decimals=2),'\n')

# Initiate a dataloader of the testing file
fileToTestModel = testing_file_all
test_dataset = PhylaDataset(fileToTestModel, 'Median')
test_loader = DataLoader(test_dataset, 
                         batch_size = args.batch_size, 
                         shuffle=True)
# Test the loaded model
test_dataset_metric = compute_accuracy(test_loader, best_model.to(device))
# Save the testing metrics to a text file
modelFileName_toSave = fileNameToSave_base + '_fold'+str(best_fold)
test_dataset_metric_nameToSave = resultFilePath + fileNameToSave_base + "_test_result_metric.txt"
write_result(test_dataset_metric_nameToSave, test_dataset_metric, 
             modelFileName_toSave, fileToTestModel)
print('')
print('Testing mix domain data =>')
print(fileToTestModel)
print('Accuracy:', np.round(test_dataset_metric[0]['Accuracy'], decimals=2), '%')
print('Precision:', np.round(test_dataset_metric[0]['Precision'], decimals=2))
print('Recall:', np.round(test_dataset_metric[0]['Recall'], decimals=2))
print('F1-score:', np.round(test_dataset_metric[0]['F1-score'], decimals=2))
print('MCC:', np.round(test_dataset_metric[0]['MCC'], decimals=2),'\n')




# Initiate a dataloader of the testing file
test_dataset_mix = torch.utils.data.ConcatDataset([test_dataset_s, test_dataset_t])
test_loader = DataLoader(test_dataset_mix, 
                         batch_size = args.batch_size, 
                         shuffle=True)
# Test the loaded model
test_dataset_metric = compute_accuracy(test_loader, best_model.to(device))
# Save the testing metrics to a text file
modelFileName_toSave = fileNameToSave_base + '_fold'+str(best_fold)
test_dataset_metric_nameToSave = resultFilePath + fileNameToSave_base + "_test_result_metric.txt"
write_result(test_dataset_metric_nameToSave, test_dataset_metric, 
             modelFileName_toSave, fileToTestModel)
print('')
print('Testing mix of source and target domain data =>')
print(fileToTestModel)
print('Accuracy:', np.round(test_dataset_metric[0]['Accuracy'], decimals=2), '%')
print('Precision:', np.round(test_dataset_metric[0]['Precision'], decimals=2))
print('Recall:', np.round(test_dataset_metric[0]['Recall'], decimals=2))
print('F1-score:', np.round(test_dataset_metric[0]['F1-score'], decimals=2))
print('MCC:', np.round(test_dataset_metric[0]['MCC'], decimals=2),'\n')

# end time
end = time.time()
totalSeconds = round(end - start)
print(f"Runtime of the program is {totalSeconds} seconds")

