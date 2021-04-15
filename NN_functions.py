#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 10:31:38 2021

@author: aikenchung
"""
import numpy as np
import json
import torch


def write_result(fileName, dataObj, modelName, testingFileName, args):
    with open(fileName, 'a') as f:
        theFirstLine = 'Model file: '+modelName+'\n'
        f.write(theFirstLine)
        theSecondLine = 'Test file: '+testingFileName+'\n'
        f.write(theSecondLine)
        for item in dataObj[0]:
            strToWrite = "{0}: {1}\n".format(item, np.round(dataObj[0][item], decimals=2))
            f.write(strToWrite)
        f.write(json.dumps(args))
        f.write('\n')

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()
    
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
def compute_accuracy(loader, net, device):
    accuracy_compute_history = []
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    net.eval()
    net.to(device)
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
         "Specificity":specificity,# def compute_accuracy(loader, net):
         "TP": TP, "TN": TN, "FP":FP, "FN": FN
         }
    )
    return accuracy_compute_history


def test_model(fileToTest, test_loader, test_result_filename, model_name_to_save, 
               classifier_model, args, device):

    # compute the performance metric for testing the loaded model
    test_dataset_metric = compute_accuracy(test_loader, classifier_model, device)
    # Save the testing metrics and related information to a text file
    write_result(test_result_filename, test_dataset_metric, 
                 model_name_to_save, fileToTest, args)
    print('')
    print('phyla classifier model testing')
    print(fileToTest)
    print('Accuracy:', np.round(test_dataset_metric[0]['Accuracy'], decimals=2), '%')
    print('Precision:', np.round(test_dataset_metric[0]['Precision'], decimals=2))
    print('Recall:', np.round(test_dataset_metric[0]['Recall'], decimals=2))
    print('F1-score:', np.round(test_dataset_metric[0]['F1-score'], decimals=2))
    print('MCC:', np.round(test_dataset_metric[0]['MCC'], decimals=2),'\n')




'''
This metric computing function is specific for DANN model
'''
def compute_accuracy_DANN(loader, net, device):
    accuracy_compute_history = []
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    net.to(device)
    net.eval()
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

'''
This model testing function is specific for DANN model
'''
def test_model_DANN(fileToTest, test_loader, test_result_filename, model_name_to_save, 
               classifier_model, args, device):

    # compute the performance metric for testing the loaded model
    test_dataset_metric = compute_accuracy_DANN(test_loader, classifier_model, device)
    # Save the testing metrics and related information to a text file
    write_result(test_result_filename, test_dataset_metric, 
                 model_name_to_save, fileToTest, args)
    print('')
    print('phylaDANN model testing')
    print(fileToTest)
    print('Accuracy:', np.round(test_dataset_metric[0]['Accuracy'], decimals=2), '%')
    print('Precision:', np.round(test_dataset_metric[0]['Precision'], decimals=2))
    print('Recall:', np.round(test_dataset_metric[0]['Recall'], decimals=2))
    print('F1-score:', np.round(test_dataset_metric[0]['F1-score'], decimals=2))
    print('MCC:', np.round(test_dataset_metric[0]['MCC'], decimals=2),'\n')
