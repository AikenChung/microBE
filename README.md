# microBE
A deep learning model to classify between microbiota of healthy and IBD patients.

## Preprocessing the data
The scripts used to preprocess data with pointwise mutual information and centered log-ratio (CLR) normalization can be found in `/preprocessing/Data_preprocessing_functions.py`. Three scaling methods were explored: standardization (using mean and standard deviation), min-max, and median absolute deviation (MAD), which can be found in `/preprocessing/normalization.py` (converted from .ipynb to .py).
## Classifers

Several traditional classifiers as well as VAEs were implemented to handle the healthy/diseased classification as well as batch effects. They can be found under `/classifiers`.

- Logistic Regression
- SVM (Linear and RBF)
- RF
- MLP
- DANN (Domain Adaptation Neural Net)
- DeepMicro (modified DM.py file for predefined training and test sets, for more information on how to implement: https://github.com/minoh0201/DeepMicro)


## System Diagram
<p align="center">
  <img src="/assets/overall_data_pipeline.png" width="600" title="microBE System Diagram">
</p>

### System Diagram - DANN
<p align="center">
  <img src="/assets/DANN_diagram.png" width="600" title="microBE System Classifier">
</p>

