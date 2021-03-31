#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy
import numpy as np
import pandas as pd
import os as os
import easydict
import phylaMLP


#================================== Setting ==================================
if not os.path.exists('./data/MLP_testResults'):
    os.makedirs('./data/MLP_testResults')
    
resultFilePath = './data/MLP_testResults/'    

# Load the saved MLP model
modelFilePath = './data/MLP_trainedModels/'
modelFileName = 'MLP_1177_128_32_1_Adam_lr_0.001_MSELoss_bSize32_epoch5000_phyla_biopsy_noCS_plsda_BE.pt'

# Define the parameters according to the loaded model
args = easydict.EasyDict({
        "feature_Num": 1177,          # Number of features (columns) in the input data
        "mlp_hidden_layers_num": 1,   # How many (middle or hidden) layers in the NN model
        "hidden_dim": 128,            # Size of each hidden layer in the NN model
        "pre_output_layer_dim": 32,   # Size of the layer right before the output layer in the NN model
        "output_dim": 1,              # Size of the output layer
        "batch_size": 32,             # Batch size
})

# Define the file to test
testing_file = './phyla_biopsy_noCS_209x1177_PMI_threshold_0_clr_plsda_15p.csv'
test_data_prefix = 'phyla_biopsy_noCS'
test_data_surfix_BE_method = 'plsda_BE'

fileNameToSave_base = (modelFileName[0:len(modelFileName)-3]+
                                 '_vs_'+test_data_prefix+'_'+
                                   test_data_surfix_BE_method)

# Load the file of the trained model
loadedModel = torch.load(modelFilePath+modelFileName)
# Initiate a model object with the same architecture of the loaded model 
Model_for_Test = phylaMLP.MLP(args.feature_Num, args.hidden_dim, 
                         args.mlp_hidden_layers_num, 
                         args.pre_output_layer_dim, args.output_dim)
# Put the loaded model into the initiated model object
Model_for_Test.load_state_dict(loadedModel[ 'model' ])

#============================== End of Setting ================================

# sets device for model and PyTorch tensors
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PhylaDataset(Dataset):
    """ 
    Dataset for binary classification of IBD/Healthy
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
        # feature-wise normalization
        self.count_data = self.normalization(self.count_data)

    def normalization(self, inputTensor):
        # feature-wise normalization
        colMin = inputTensor.min(0, keepdim=True)[0]
        colMax = inputTensor.max(0, keepdim=True)[0]    
        outputTensor = (inputTensor - colMin) / (colMax - colMin)
        return outputTensor

    def __getitem__(self, index):
        samples = self.count_data[index]
        labels = self.diagnosis_data[index]
        return samples, labels

    def __len__(self):
        return self.len
    

"""
### Define the evaluation metric

The evaluation metrics include Accuracy, Specificity, Precision, Recall, 
F1-score, and MCC.

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
            outputs = net(samples)           
            for i in range(labels.shape[0]):
                sample_val = labels[i,0]
                predict_val= outputs[i,0]
                if sample_val == 1:
                    if predict_val>0.5:
                        TP = TP + 1
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

# Function of writing the testing results to a text file
def write_result(fileName, dataObj, modelName, testingFileName):
    with open(fileName, 'w') as f:
        theFirstLine = 'Model file: '+modelName+'\n'
        f.write(theFirstLine)
        theSecondLine = 'Test file: '+testingFileName+'\n'
        f.write(theSecondLine)
        for item in dataObj[0]:
            strToWrite = "{0}: {1}\n".format(item, np.round(dataObj[0][item], decimals=2))
            f.write(strToWrite)

    

"""
Run the testing procedure
"""

# Initiate a dataloader of the testing file
test_dataset = PhylaDataset(testing_file)
test_loader = DataLoader(test_dataset, 
                         batch_size = args.batch_size, 
                         shuffle=True)


# Test the loaded model
test_dataset_metric = compute_accuracy(test_loader, Model_for_Test)
# Save the testing metrics to a text file
test_dataset_metric_nameToSave = resultFilePath + fileNameToSave_base + "_test_result_metric.txt"
write_result(test_dataset_metric_nameToSave, test_dataset_metric, modelFileName, testing_file)



