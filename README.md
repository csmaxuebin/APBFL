# This code is the source code implementation for the paper "AdaDP-CFL: Cluster Federated Learning with Adaptive Clipping Threshold Differential Privacy."



## Abstract
![输入图片说明](https://github.com/csmaxuebin/PACFL/blob/main/PACFL/pic/1.png)
Federated learning is a distributed machine learning method that allows multiple clients to collaborate to train models. Since the data is still stored locally, this method effectively protects the privacy of client information. Nevertheless, in today's context, federated learning still faces the risk of privacy leakage. Differential privacy techniques are widely used in federated learning to protect client privacy, where the size of the privacy budget directly affects the utility of the model. This study does not fully consider different privacy requirements, because client parameters in different communication rounds will bring different privacy leakage risks. To address these challenges, we propose a differential privacy federated learning algorithm based on privacy budget allocation and sparsification, called ADP-PFL. This method introduces an adaptive privacy budget allocation mechanism, which dynamically allocates privacy budgets to clients in different communication rounds according to the quantified privacy leakage risk, providing adaptive privacy protection. At the same time, model pruning reduces the dimension of parameters to reduce sensitivity, effectively reducing the amount of noise added after applying differential privacy. Experimental results show that our algorithm outperforms the existing state-of-the-art algorithms in terms of accuracy and effectively balances privacy protection and model practicality.


# Experimental Environment

```
- breaching==0.1.2
- calmsize==0.1.3
- h5py==3.8.0
— opacus==1.4.0
- Pillow==9.2.0
- scikit-learn==1.2.2
- sklearn==0.0.post1
- torch==2.0.0
- torchvision~=0.15.1+cu117
- ujson==5.7.0
- numpy==1.23.2
- scipy==1.8.1
- matplotlib==3.5.2
```

## Datasets

`CIFAR10, FMNIST, SVHN, and CIFAR100`


## Experimental Setup

### Model Configurations    
A CNN network comprising two convolutional layers and three fully connected layers is employed. The convolutional layers output 6 and 16 channels respectively, with a kernel size of 5x5, while the fully connected layers output 120, 84, and 10 channels respectively. In terms of data distribution, the experiment initially randomly assigns 20% of the total labels in the dataset to each client and then distributes samples of each label randomly to the clients that hold that label.

### Hyperparameter Settings

Hyperparameter settings include 100 global rounds, 5 local iterations per round, a learning rate of 0.01, 100 clients, a sampling rate of 0.1, and momentum set at 0.5. These settings are designed to evaluate performance across different datasets and model architectures while examining the impact of client data heterogeneity on learning outcomes.
## Python Files
	
-   **Client_ClusterFL.py**:    
    The code defines a class `Client_ClusterFL` for a federated learning client with differential privacy features. It includes methods for training with gradient clipping and noise addition for privacy, adjusting learning parameters, and evaluating model performance. This setup allows clients to improve their local models in a privacy-preserving manner during federated learning rounds.
-   **cluster_fl.py**:    
    The code facilitates the evaluation and clustering of client models in a federated learning setting based on their predictive performance on shared data, aiming to improve model collaboration and privacy through effective grouping and performance analysis.
-   **fedavg.py**:    
Used to aggregate parameters.
-   **Distance.py**:    
The provided Python function, `Distance`, calculates the relationship or similarity between different clusters of clients in a federated learning environment based on their model parameters.
-   **prox.py**:
This code defines a function called L2 that calculates and returns the loss value in a particular optimization environment. This function is mainly used in federated learning or model aggregation scenarios, where the L2 norm difference of the model parameters and the weighted difference between related parameters are considered.
-   **utils.py**:    
   Used to load data from different datasets, set model weights, and initialize model parameters.

##  Experimental Results
The images showcase model performance with varying privacy budgets across four datasets (CIFAR10, FMNIST, SVHN, and MNIST):

1. **Figure 3**: Displays model accuracy for different privacy levels. It compares a non-DP baseline with both standard and improved DP models. Higher privacy budgets correlate with increased accuracy.

2. **Figure 4**: A table comparing algorithm accuracies for different privacy budgets, confirming that higher ε values improve accuracy.

3. **Figure 5**: Compares different DP sparsification strategies. The proposed method ("Ours") typically outperforms or matches other methods, illustrating its effectiveness.

4. **Figure 6**: Illustrates accuracy progression across training rounds under different sparsification levels. While sparsification initially delays accuracy gains, all methods eventually converge to similar levels, showing that sparsification can reduce data transmission costs without greatly affecting final accuracy.
![输入图片说明](https://github.com/csmaxuebin/PACFL/blob/main/PACFL/pic/2.png)
![输入图片说明](https://github.com/csmaxuebin/PACFL/blob/main/PACFL/pic/3.png)
![输入图片说明](https://github.com/csmaxuebin/PACFL/blob/main/PACFL/pic/4.png)
![输入图片说明](https://github.com/csmaxuebin/PACFL/blob/main/PACFL/pic/5.png)




## Update log

```
- {24.06.13} Uploaded overall framework code and readme file
```

