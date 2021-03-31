#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This VAE is modeling parameters of Gaussian distribution in its latent space.
The input data is PMI (threshold:0) filtered and CLR normalized phyla data
with 1177 features. (Both original and BE corrected data are needed)
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy
from torch.distributions import NegativeBinomial
import numpy as np
import pandas as pd
#progess bar
from tqdm import tqdm
import time
import os as os
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import easydict
from torch.autograd import Variable

import os as os
if not os.path.exists('/content/gdrive/My Drive/Colab Notebooks/data'):
    os.makedirs('/content/gdrive/My Drive/Colab Notebooks/data')

if not os.path.exists('/content/gdrive/My Drive/Colab Notebooks/data'):
    os.makedirs('/content/gdrive/My Drive/Colab Notebooks/data')

if not os.path.exists('/content/gdrive/My Drive/Colab Notebooks/data/VAE_trainedModels'):
    os.makedirs('/content/gdrive/My Drive/Colab Notebooks/data/VAE_trainedModels')

if not os.path.exists('/content/gdrive/My Drive/Colab Notebooks/data/VAE_trainedResults'):
    os.makedirs('/content/gdrive/My Drive/Colab Notebooks/data/VAE_trainedResults')


if not os.path.exists('./data/VAE_testResults'):
    os.makedirs('./data/VAE_testResults')
    
vae_testResultFilePath = '/content/gdrive/My Drive/Colab Notebooks/data/VAE_testResults/'    
modelFilePath = '/content/gdrive/My Drive/Colab Notebooks/data/VAE_trainedModels/'
resultFilePath = '/content/gdrive/My Drive/Colab Notebooks/data/VAE_trainedResults/'

#================================== Setting ==================================
train_raw_file = '/content/gdrive/My Drive/Colab Notebooks/phyla_stool_noNC_2798x1177_PMI_threshold_0_clr_85p.csv'
train_BE_file = '/content/gdrive/My Drive/Colab Notebooks/phyla_stool_noNC_2798x1177_PMI_threshold_0_clr_85p.csv'

args_vae = easydict.EasyDict({
        "feature_Num": 1177,        # Number of features (columns) in the input data
        "epochs": 500,              # Number of iterations to train Model for
        "hidden_dim": 512,          # Size of each hidden layer in Discriminator
        "latent_dim": 64,           # Size of each hidden layer in Discriminator
        "vae_hidden_layer_num": 2,  # How many (middle or hidden) layers in Discriminator (ie. 'mlp':  w/o 1st & last; 'resnet's: num. resudual blocks)
        "batch_size": 32,           # Batch size
        "learning_rate": 0.001,     # Learning rate for the optimizer
        "vae_type": 'BCELogits',          # 'MSE' for Gaussian VAE, 'nbELBO' for Negative Binomial VAE
        "adapt_lr_iters": 10,        # how often decrease the learning rate
        "lmbda": 10,                  # 'lmbda' for Lipschitz gradient penalty hyperparameter
})

args_mlp = easydict.EasyDict({
        "feature_Num": 1177,        # Number of features (columns) in the input data
        "epochs": 5000,             # Number of iterations to train Model for
        "hidden_dim": 128,          # Size of each hidden layer in Discriminator
        "pre_output_layer_dim": 32, # Size of each hidden layer in Discriminator
        "output_dim": 1,            # Size of output layer
        "mlp_hidden_layers_num": 1, # How many (middle or hidden) layers in Discriminator (ie. 'mlp':  w/o 1st & last; 'resnet's: num. resudual blocks)
        "batch_size": 32,           # Batch size
        "learning_rate": 0.0003,     # Learning rate for the optimizer
        "beta1": 0.5,               # 'beta1' for the optimizer
        "adapt_lr_iters": 10,        # how often decrease the learning rate
})

input_data_prefix = 'phyla_stool_noNC'
input_data_surfix_BE_method = 'no_BE'
fileNameToSave_base_vae = ('VAE_'+ str(args_vae.feature_Num) +'_'+ 
                               str(args_vae.hidden_dim) + 'x' + 
                               str(args_vae.vae_hidden_layer_num) + '_' +
                               str(args_vae.latent_dim) + '_' +
                               str(args_vae.hidden_dim) + 'x' + 
                               str(args_vae.vae_hidden_layer_num) + '_Adam_lr_' +
                               str(args_vae.learning_rate) + '_'+
                               str(args_vae.vae_type) + 'Loss_bSize'+
                               str(args_vae.batch_size) + '_epoch'+
                               str(args_vae.epochs) + '_phyla_stool_noNC_no_BE')

#####################################  MLP ####################################
testing_file = '/content/gdrive/My Drive/Colab Notebooks/phyla_stool_noNC_467x1177_PMI_threshold_0_clr_15p.csv'

# Load the saved MLP model
#modelFilePath_loaded_mlp = './data/MLP_trainedModels/'
modelFileName_loaded_mlp = '/content/gdrive/My Drive/Colab Notebooks/MLP_1177_128_32_1_Adam_lr_0.001_MSELoss_bSize32_epoch5000_phyla_stool_noNC_no_BE.pt'

args_loaded_mlp = easydict.EasyDict({
        "feature_Num": 1177,          # Number of features (columns) in the input data
        "hidden_dim": 128,            # Size of each hidden layer in the NN modelReference file testing result
        "mlp_hidden_layers_num": 1,   # How many (middle or hidden) layers in the NN model       
        "pre_output_layer_dim": 32,   # Size of the layer right before the output layer in the NN model
        "output_dim": 1,              # Size of the output layer
        "batch_size": 32,             # Batch size
})

norm_data_df = pd.read_csv(testing_file, low_memory=False, lineterminator='\n')
the_first_colum_df = norm_data_df.iloc[:, 0:1].copy()
the_first_colum_df.columns = ['run_accession']
data_value_df = norm_data_df.iloc[:, 1:1178].copy()
metaData_df = norm_data_df.iloc[:, 1178:1185].copy()

class MLP(nn.Module):
    """ Multi-Layer Perceptron for classifying IBD and Healthy microbiome data"""
    def __init__(self, input_dim=1177, hidden_dim=256, 
                 hidden_layer_num=1, 
                 pre_output_dim = 64, 
                 output_dim=1):        
        super(MLP, self).__init__()        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for layer in range(hidden_layer_num):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, pre_output_dim))
        self.layers.append(nn.Linear(pre_output_dim, output_dim))
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        out = self.layers[-1](x)
        out = torch.sigmoid(out)
        return out

# Load the file of the trained model
loadedModel_mlp = torch.load(modelFileName_loaded_mlp)
# Initiate a model object with the same architecture of the loaded model 
Model_mlp = MLP(args_loaded_mlp.feature_Num, args_loaded_mlp.hidden_dim, 
                         args_loaded_mlp.mlp_hidden_layers_num, 
                         args_loaded_mlp.pre_output_layer_dim, args_loaded_mlp.output_dim)
# Put the loaded model into the initiated model object
Model_mlp.load_state_dict(loadedModel_mlp[ 'model' ])


# sets device for model and PyTorch tensors
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#============================== End of Setting ================================


class Phyla_VAE_Dataset(Dataset):
    """ Phyla dataset"""
    """
    Dataset for binary classification IBD/Healthy
    """
    # Initialize your data, download, etc.
    def __init__(self, inputFile, inputFile_BE):
        ori_data = pd.read_csv(inputFile, low_memory=False, lineterminator='\n')
        phyla_input = ori_data[ori_data.columns[1:args_vae.feature_Num+1]]
        phyla_input = phyla_input.to_numpy(dtype=np.float32)
        BE_data = pd.read_csv(inputFile_BE)
        phyla_input_BE = BE_data[BE_data.columns[1:args_vae.feature_Num+1]]
        phyla_input_BE = phyla_input_BE.to_numpy(dtype=np.float32)
        self.len = phyla_input.shape[0]
        self.count_data_raw = from_numpy(phyla_input)
        self.count_data_BE = from_numpy(phyla_input_BE)
        # feature-wise normalization
        self.count_data_raw = self.normalization(self.count_data_raw)
        self.count_data_BE = self.normalization(self.count_data_BE)
        
    def normalization(self, inputTensor):
        # feature-wise normalization
        colMin = inputTensor.min(0, keepdim=True)[0]
        colMax = inputTensor.max(0, keepdim=True)[0]    
        outputTensor = (inputTensor - colMin) / (colMax - colMin)
        return outputTensor
    
    def __getitem__(self, index):
        return self.count_data_raw[index], self.count_data_BE[index]

    def __len__(self):
        return self.len
 
class Phyla_MLP_Dataset(Dataset):
    """ Phyla dataset"""
    """
    Dataset for binary classification IBD/Healthy
    """
    # Initialize your data, download, etc.
    def __init__(self, inputFile):
        ori_data = pd.read_csv(inputFile)
        phyla_input = ori_data[ori_data.columns[1:args_mlp.feature_Num+1]]
        phyla_input = phyla_input.assign(diagnosis=ori_data[ori_data.columns[args_mlp.feature_Num+2]])
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
        samples =self.count_data[index]
        labels = self.diagnosis_data[index]
        return samples, labels

    def __len__(self):
        return self.len

dataset = Phyla_VAE_Dataset(train_raw_file, train_BE_file)
 
"""
We now need to split our dataset into two parts.
The **train set** will be used to train our model, 
and the **validation set** will be used for evaluation.
First, let us compute the number of samples to put in each split. 
Here we choose to keep 80\% of the samples for training and 20\% for validating.
"""

# starting time
start = time.time()

train_set_size = int(len(dataset) * 0.8)
validation_set_size = len(dataset) - train_set_size

"""Split randomly our dataset into two parts"""

train_dataset, validation_dataset = torch.utils.data.random_split(dataset, 
                                                            lengths=[train_set_size, validation_set_size], 
                                                            generator=torch.Generator().manual_seed(0))

"""We initialize dataloader objects. These dataloaders will provide data one batch at a time, which is convenient to train our machine learning model."""

train_loader = DataLoader(train_dataset, batch_size=args_vae.batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=args_vae.batch_size, shuffle=True)


# VAE model
class VarAutoEncoder(torch.nn.Module):
    """
    This VAE model has 1 encoder with 1 input layer, 2 latent output layer and 
    1 decoder with 1 latent input layer and 1 final output layer.
    """
    def __init__(self, input_size=args_vae.feature_Num, hidden_size=args_vae.hidden_dim, 
                 hidden_layer_num=args_vae.vae_hidden_layer_num, bottleneck_size=args_vae.latent_dim):
        super(VarAutoEncoder, self).__init__()
        self.hidden_layer_num = hidden_layer_num
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.encoderLayer = nn.ModuleList()
        for layer in range(hidden_layer_num-1):
            self.encoderLayer.append(nn.Linear(hidden_size, hidden_size))
        self.fc2 = torch.nn.Linear(hidden_size, bottleneck_size)
        self.fc3 = torch.nn.Linear(hidden_size, bottleneck_size)
        self.fc4 = torch.nn.Linear(bottleneck_size, hidden_size)
        self.decoderLayer = nn.ModuleList()
        for layer in range(hidden_layer_num-1):
            self.decoderLayer.append(nn.Linear(hidden_size, hidden_size))
        self.fc5 = torch.nn.Linear(hidden_size, input_size)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        if (self.hidden_layer_num>1):
            for layer in self.encoderLayer:
                h = F.relu(layer(h))
        return self.fc2(h), self.fc3(h)
    
    def reparameterize(self, mu, log_var):
        # Ref: Reparameterization Trick
        # 1. https://gokererdogan.github.io/2016/07/01/reparameterization-trick/
        # 2. https://towardsdatascience.com/generating-images-with-autoencoders-77fd3a8dd368
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        if (self.hidden_layer_num>1):
            for layer in self.decoderLayer:
                h = F.relu(layer(h))
        #return torch.sigmoid(self.fc5(h))
        return self.fc5(h)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var



def vae_train_pass(model_D, model_G, data_loader, optimizer_D, optimizer_G, criterion, device):
    model_G.train()
    model_D.train()
    pass_lossD = 0.0
    pass_lossG = 0.0
    for count_data_raw, count_data_BE in data_loader:
        ############################
        # (1) Update model_ D network: maximize log(D(x)) + log(1 - D(G(z))) + GP(disc_grad_penalty) wanting D(x)=1 & D(G(z))=0
        # Ref: Ahmad's source code for training GAN
        ############################
        count_data_raw = count_data_raw.float().to(device)
        count_data_BE = count_data_BE.float().to(device)
        fake_count, mu, log_var = model_G(count_data_raw)
        
        disc_real = model_D(count_data_BE).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = model_D(fake_count).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        # regularization_penalty term: GP(disc_grad_penalty) wanting D(x)=1 & D(G(z))=0
        #regularization_penalty = disc_grad_penalty(model_D, count_data_BE, penalty_amount=args_vae.lmbda)
        # cost function that discriminator network tries to minimize by maximizing the eq:
        # log(D(x)) + log(1 - D(G(z))) + GP(disc_grad_penalty) wanting D(x)=1 & D(G(z))=0
        #lossD = lossD_real + lossD_fake + regularization_penalty
        lossD = (lossD_real + lossD_fake) / 2
        model_D.zero_grad()
        lossD.to(device)
        lossD.backward(retain_graph=True)
        optimizer_D.step()
        pass_lossD += lossD.item()
        
        ############################
        # (2) Update model_G network: maximize log(model_D(G(z))) wanting model_D(G(z))=1
        # Ref: Ahmad's source code for training GAN
        ############################
        outputD_fake = model_D(fake_count).view(-1)
        # Try to produce the output that mimics the input and fools the model_D at this step
        lossG = criterion(outputD_fake, torch.ones_like(outputD_fake))
        model_G.zero_grad()
        lossG.backward()
        optimizer_G.step()
        pass_lossG += lossG.item()
        
    return pass_lossD/len(data_loader), pass_lossG/len(data_loader)

def vae_validate_pass(model_D, model_G, data_loader, criterion, device):
    with torch.no_grad():
      pass_loss_real = 0.0
      pass_loss_fake = 0.0
      for count_data_raw, count_data_BE in data_loader:         
          # Loss for real count data
          real_count = count_data_BE.float().to(device)
          real_count_result = model_D(real_count)         
          loss_real_count = criterion(real_count_result, torch.ones_like(real_count_result))
          loss_real_count.to(device)
          pass_loss_real += loss_real_count.item()          
          count_data_raw = count_data_raw.float().to(device)
          # Loss for fake (generated) count data
          fake_count, mu, log_var = model_G(count_data_raw)
          fake_count_result = model_D(fake_count)
          loss_fake_count = criterion(fake_count_result, torch.ones_like(fake_count_result))
          loss_fake_count.to(device)
          pass_loss_fake += loss_fake_count.item()
    return pass_loss_real/len(data_loader), pass_loss_fake/len(data_loader)


def vae_training(model_D, model_G, nb_epochs, train_loader, test_loader, 
                 optimizerD, optimizerG, schedulerD, schedulerG, criterion, device, fileNameForModel):
    model_D = model_D.to(device)
    model_G = model_G.to(device)
    progress_bar = tqdm(range(nb_epochs), position=0, leave=True)
    loss_history = [] 
    for epoch in progress_bar:
        validate_loss = vae_validate_pass(model_D, model_G, test_loader, criterion, device=device)
        train_loss = vae_train_pass(model_D, model_G, train_loader, optimizerD, optimizerG, criterion, device=device)
        loss_history.append(
            {"loss": train_loss[0], "set": "train_D", "epochs": epoch}
        )
        loss_history.append(
            {"loss": train_loss[1], "set": "train_G", "epochs": epoch}
        )
        loss_history.append(
            {"loss": validate_loss[0], "set": "validate_real", "epochs": epoch}
        )
        loss_history.append(
            {"loss": validate_loss[1], "set": "validate_fake", "epochs": epoch}
        )
        checkpointG = {'model' : model_G.state_dict(), 'optimizer': optimizerG.state_dict()}
        fileNameToSaveG = fileNameForModel + ".pt"
        checkpointD = {'model' : model_D.state_dict(), 'optimizer': optimizerD.state_dict()}
        fileNameToSaveD = fileNameForModel + "_disc.pt"
        epochToSave = modelFilePath + 'training_epoch.txt'
        if epoch % args_vae.adapt_lr_iters == 0 :
            save_model(checkpointG,fileNameToSaveG)
            save_model(checkpointD,fileNameToSaveD)
            f = open(epochToSave, "w")
            f.write("Epoch-"+str(epoch))
            f.close()
            # Using scheduler to update the learning rate every 100 iterations.
            schedulerG.step()
            schedulerD.step()
    print('')
    print('Finished Training')
    return pd.DataFrame(loss_history)


def save_model(model_state, fileName):
    #print("=> Saving model")
    torch.save(model_state,fileName)

def vae_gaussian_Loss(x_reconst, mu, log_var, x_ori):
    # For gaussian distribution
    kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    mse_loss = nn.MSELoss(reduction='sum')
    MSE = mse_loss(x_reconst,x_ori)
    loss = MSE + kl_div
    return loss

def vae_nb_Loss(x_reconst, mu, log_var, x_ori):
    #### NB distribution    
    kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    theta = torch.exp(torch.nn.Parameter(x_ori))
    nb_logits = (x_reconst + 1e-4).log() - (theta + 1e-4).log() # Each entry represents logits for the probability of success for independent Negative Binomial distributions
    log_lik = NegativeBinomial(total_count=theta, logits=nb_logits).log_prob(x_reconst).sum(dim=-1)
    # ELBO: Evidence Lower Bound
    # Reference:
    # He Zhao et. al. Variational Autoencoders for Sparse and Overdispersed Discrete Data AISTATS (2020)
    elbo = log_lik - kl_div
    loss = torch.mean(-elbo)
    #return neg_elbo
    return loss


"""#### Gradient Penalty Function (or some other helpful functions)
Ref: Ahmad's source code for training GAN
"""
def disc_grad_penalty(disc, real_samples, penalty_amount=10, retain=True):
    """Function to compute gradient (norm) penalty loss in order to hinder undesirable large changes in gradient computation 
    Parameters:
        disc_network (torch.nn.module):Discriminator object.
        real_samples (tensor):'Real' data samples.
        penalty_amount (in):'lmbda' or weight of the penalty.
        retain (bool):If False, the graph used to compute the grad will be freed (i.e., cannot do another loss.backward)      
    Returns:
        GP_loss (int):Gradient (norm) penalty loss (ie. of the gradient regularization term in the total loss funtion).
    """
    def _get_gradient(inp, output):
        gradient = torch.autograd.grad(outputs=output, inputs=inp,
                                 grad_outputs=torch.ones_like(output),
                                 create_graph=True, retain_graph=True,
                                 only_inputs=True, allow_unused=True)[0]
        return gradient
    
    real_samples = Variable(real_samples.clone().detach().requires_grad_(True).float().to(device))
    if not isinstance(real_samples, (list, tuple)):
        real_samples = [real_samples]
        
    real_samples = [inp.detach() for inp in real_samples]
    real_samples = [inp.requires_grad_() for inp in real_samples]
    with torch.set_grad_enabled(True):
        output = disc(real_samples[0])
    # compute gradient
    gradient = _get_gradient(real_samples, output)
    
    # get norm square: ||grad||^2
    gradient = gradient.view(gradient.size()[0], -1)
    penalty = (gradient ** 2).sum(1).mean()
    
    gp_loss = penalty_amount * penalty
    
    gp_loss.backward(retain_graph=retain)
    
    return gp_loss



"""
Define the evaluation metric
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
def write_result(fileName, dataObj, vae_modelName, mlp_modelName, testingFileName, message):
    with open(fileName, 'a') as f:
        theZeroLine = message+'\n'
        f.write(theZeroLine)
        theFirstLine = 'VAE Model file: '+vae_modelName+'\n'
        f.write(theFirstLine)
        theSecondLine = 'MLP Model file: '+mlp_modelName+'\n'
        f.write(theSecondLine)
        theThirdLine = 'Test file: '+testingFileName+'\n'
        f.write(theThirdLine)
        for item in dataObj[0]:
            strToWrite = "{0}: {1}\n".format(item, np.round(dataObj[0][item], decimals=2))
            f.write(strToWrite)


#criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()

# Initialization of the VAE obj
varAutoEncoder = VarAutoEncoder(input_size=args_vae.feature_Num, 
                                hidden_size=args_vae.hidden_dim, 
                                hidden_layer_num=args_vae.vae_hidden_layer_num, 
                                bottleneck_size=args_vae.latent_dim)
# Initilize model, criterion, optimizer. Then train the model
discriminatorMLP = MLP(args_mlp.feature_Num, args_mlp.hidden_dim, 
                         args_mlp.mlp_hidden_layers_num, 
                         args_mlp.pre_output_layer_dim, args_mlp.output_dim)


# For printing out the structure of VAE
varAutoEncoder.to(device)
discriminatorMLP.to(device)

# setup optimizer
optimizer_vae = torch.optim.Adam(varAutoEncoder.parameters(), lr=args_vae.learning_rate)
optimizer_mlp = torch.optim.Adam(list(discriminatorMLP.parameters()), lr=args_mlp.learning_rate, betas=(args_mlp.beta1, 0.999))
# use an exponentially decaying learning rate
scheduler_mlp= torch.optim.lr_scheduler.ExponentialLR(optimizer_mlp, gamma=0.99)
scheduler_vae= torch.optim.lr_scheduler.ExponentialLR(optimizer_vae, gamma=0.99)

"""
Start to run the VAE model
"""

training_history = vae_training(discriminatorMLP, varAutoEncoder, args_vae.epochs, train_loader, validation_loader, 
                                   optimizer_mlp, optimizer_vae, scheduler_mlp, scheduler_vae, 
                                   criterion, device=device, fileNameForModel=modelFilePath+fileNameToSave_base_vae)

plt.figure()
ax = sns.lineplot(x="epochs", y="loss", hue= "set", data=training_history)
fig_trainHistory = ax.get_figure()
training_history_plotName = resultFilePath + fileNameToSave_base_vae +'_training_history.png'
fig_trainHistory.savefig(training_history_plotName)

# end time
end = time.time()
totalSeconds = round(end - start)


print(f"Runtime of the program is {totalSeconds} seconds")

"""
Run the testing procedure.
"""
# Initiate a dataloader of the testing file
test_dataset = Phyla_MLP_Dataset(testing_file)
test_loader = DataLoader(test_dataset, 
                         batch_size = args_mlp.batch_size, 
                         shuffle=True)
# Test the loaded model
test_dataset_metric = compute_accuracy(test_loader, Model_mlp.to(device))
# Save the testing metrics to a text file
test_dataset_metric_nameToSave = resultFilePath + fileNameToSave_base_vae + "_test_result_metric.txt"
write_result(test_dataset_metric_nameToSave, test_dataset_metric, 
             fileNameToSave_base_vae, modelFileName_loaded_mlp, 
             testing_file, 'Ref. file results:')

print(testing_file)
print('Accuracy:', np.round(test_dataset_metric[0]['Accuracy'], decimals=2), '%')
print('Precision:', np.round(test_dataset_metric[0]['Precision'], decimals=2))
print('Recall:', np.round(test_dataset_metric[0]['Recall'], decimals=2))
print('F1-score:', np.round(test_dataset_metric[0]['F1-score'], decimals=2))
print('MCC:', np.round(test_dataset_metric[0]['MCC'], decimals=2),'\n')
# Load the trained model
generated_data, generated_mu, generated_log_var = varAutoEncoder(test_dataset.count_data.to(device))

#test_dataset.count_data = reconst_data_tensor
test_dataset.count_data = generated_data
test_loader_vae = DataLoader(test_dataset, 
                         batch_size = args_mlp.batch_size, 
                         shuffle=True)
# Test the loaded MLP model again with the VAE reconstructed data
test_dataset_metric_vae = compute_accuracy(test_loader_vae, Model_mlp.to(device))
# Save the testing metrics to a text file
write_result(test_dataset_metric_nameToSave, test_dataset_metric_vae, 
             fileNameToSave_base_vae, modelFileName_loaded_mlp, 
             testing_file, 'Ref. file results:')

print('VAE generated data:')
print('Accuracy:', np.round(test_dataset_metric_vae[0]['Accuracy'], decimals=2), '%')
print('Precision:', np.round(test_dataset_metric_vae[0]['Precision'], decimals=2))
print('Recall:', np.round(test_dataset_metric_vae[0]['Recall'], decimals=2))
print('F1-score:', np.round(test_dataset_metric_vae[0]['F1-score'], decimals=2))
print('MCC:', np.round(test_dataset_metric_vae[0]['MCC'], decimals=2))

