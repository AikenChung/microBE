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
        ori_BE_data = pd.read_csv(inputFile)
        phyla_BE_input = ori_BE_data[ori_BE_data.columns[1:args.feature_Num+1]]
        phyla_BE_input = phyla_BE_input.assign(diagnosis=ori_BE_data[ori_BE_data.columns[args.feature_Num+2]])
        phyla_BE_input = phyla_BE_input.to_numpy(dtype=np.float32)
        self.len = phyla_BE_input.shape[0]
        self.count_data_raw = from_numpy(phyla_BE_input[:, 0:-1])
        self.diagnosis_data_raw = from_numpy(phyla_BE_input[:, [-1]]) # 0: Control, 1: IBD
        self.count_data_BE = from_numpy(phyla_BE_input[:, 0:-1])
        self.diagnosis_data_BE = from_numpy(phyla_BE_input[:, [-1]]) # 0: Control, 1: IBD

    def __getitem__(self, index):
        #return self.count_data_raw[index], self.count_data_BE[index]
        samples = self.count_data_BE[index]
        labels = self.diagnosis_data_BE[index]
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
    if (TP+FN) != 0:
        recall = TP/(TP+FN) # sensitivity
    if (TN+FP) != 0:
        specificity = TN/(TN+FP)
    if (TP+FP) != 0:
        precision = TP/(TP+FP)
    if (TP+TN+FP+FN) != 0:    
        accuracy = 100*(TP+TN)/(TP+TN+FP+FN)
    if (precision+recall) != 0:
        f1 = (2*precision*recall)/(precision+recall)
    mcc = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))  # Matthews correlation coefficient
    accuracy_compute_history.append(
        {"TP": TP, "TN": TN, "FP":FP, "FN": FN,
         "Recall":recall, "Specificity":specificity,
         "Precision":precision, "Accuracy":accuracy,
         "F1-score":f1, "MCC":mcc}
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



