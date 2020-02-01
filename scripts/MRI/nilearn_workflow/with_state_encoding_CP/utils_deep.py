#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:07:08 2020

@author: nmei
"""

from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score
from scipy.spatial import distance
from joblib import Parallel,delayed
import gc
gc.collect()


import torch
from torch import nn,no_grad
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import TensorDataset,DataLoader
torch.manual_seed(12345)


class VAE(nn.Module):
    def __init__(self,
                 input_dim = 1000,
                 output_dim = 1000,
                 encode_dims = [1280,300,],
                 decode_dims = [300,1280,],
                 vae_dim = 100,
                 dropout_rate = .5,
                 ):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encode_dims = encode_dims
        self.decode_dims = decode_dims
        self.vae_dim = vae_dim
        self.dropout_rate = dropout_rate
        
        self.encoding = nn.ModuleList()
        current_dim = self.input_dim
        for hidden_dim in self.encode_dims:
            self.encoding.append(nn.Linear(current_dim,hidden_dim))
            current_dim = hidden_dim
        self.decoding = nn.ModuleList()
        current_dim = self.vae_dim
        for hidden_dim in self.decode_dims:
            self.decoding.append(nn.Linear(current_dim,hidden_dim))
            current_dim = hidden_dim
        
        if len(self.encode_dims) > 0:
            self.hidden_mu = nn.Linear(self.encode_dims[-1],self.vae_dim)
            self.hidden_var = nn.Linear(self.encode_dims[-1],self.vae_dim)
        else:
            self.hidden_mu = nn.Linear(self.input_dim,self.vae_dim)
            self.hidden_var = nn.Linear(self.input_dim,self.vae_dim)
        if len(self.decode_dims) > 0:
            self.output_layer = nn.Linear(self.decode_dims[-1],self.output_dim)
        
        self.dropout_layer = nn.AlphaDropout(p = self.dropout_rate)
        
    def encode(self, x):
        for layer in self.encoding:
                x = F.selu(layer(x))
                x = self.dropout_layer(x)
        return torch.sigmoid(self.hidden_mu(x)), torch.sigmoid(self.hidden_var(x))
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        if len(self.decode_dims) > 0:
            for layer in self.decoding:
                z = F.selu(layer(z))
            return torch.sigmoid(self.output_layer(z))
        else:
            return torch.sigmoid(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def VEA_loss_function(recon_x, x, mu, logvar,train = False,BATCH_SIZE = 1,):
    if train:
        BCE = 0
        for recon_x_one,x_one in zip(recon_x,x):
            BCE += F.binary_cross_entropy(recon_x_one.view(1,-1),x_one.view(1,-1),)
        BCE /= len(recon_x)
    else:
        BCE = F.binary_cross_entropy(recon_x, x,)
    
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= x.shape[1] * BATCH_SIZE
    
    # 
    convolution = nn.Conv1d(1,1,kernel_size = 2,bias = False)
    list(convolution.parameters())[0].requires_grad = False
    convolution.weight.copy_(torch.tensor([1.,-1.]))
    dists = convolution(x.view(BATCH_SIZE,1,-1)).view(BATCH_SIZE,-1)
    dist_loss =  torch.sum(torch.pow(dists,2)) 
    dist_loss /= dists.shape[1] * BATCH_SIZE
    return BCE + KLD + dist_loss

def train(epoch,model,dataloader_train,device,optimizer,loss_function,batch_size,print_train = False):
    model.train()
    train_loss = 0
    if print_train:
        iterator = tqdm(enumerate(dataloader_train))
    else:
        iterator = enumerate(dataloader_train)
    for batch_idx, (features_, BOLD_) in iterator:
        features_ = features_.to(device)
        optimizer.zero_grad()
        prediction_batch, mu, logvar = model(features_)
        loss = loss_function(prediction_batch, BOLD_, mu, logvar,train = True,BATCH_SIZE = batch_size)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    if print_train:
        print(f'Epoch: {epoch} Average loss: {train_loss / len(dataloader_train.dataset):.4f}')
    return model,train_loss / len(dataloader_train.dataset)

def validation(epoch,model,dataloader_valid,device,loss_function,batch_size,print_train = False):
    model.eval()
    validation_loss = 0
    if print_train:
        iterator = tqdm(enumerate(dataloader_valid))
    else:
        iterator = enumerate(dataloader_valid)
    with torch.no_grad():
        for i, (features_, BOLD_) in iterator:
            features_ = features_.to(device)
            prediction_batch, mu, logvar = model(features_)
            validation_loss += loss_function(prediction_batch, BOLD_, mu, logvar,train = False,BATCH_SIZE = batch_size).item()

    validation_loss /= len(dataloader_valid.dataset)
    if print_train:
        print(f'Test set loss: {validation_loss:.4f}')
    return validation_loss

def fit(model,device,optimizer,dataloader_train,dataloader_valid,loss_function,
        batch_size = 1,n_epochs = int(3e3),patience = 5,print_train = False,tol = 1e-4,):
    train_losses, validation_losses = [],[]
    current_best = np.inf
    counter = 0
    for epoch in range(n_epochs):
        model,train_loss = train(epoch,
                                 model,
                                 dataloader_train,
                                 device,
                                 optimizer,
                                 loss_function,
                                 batch_size,
                                 print_train = print_train)
        validation_loss = validation(epoch,
                                     model,
                                     dataloader_valid,
                                     device,
                                     loss_function,
                                     batch_size,
                                     print_train = print_train)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        
        if (np.abs(current_best - validation_loss) >= tol) and (current_best > validation_loss):
            current_best = validation_loss
            counter = 0
        else:
            counter += 1
        
        if counter > patience:
            break
    return model,validation_losses

def train_valid_data_loader(X_train,X_valid,y_train,y_valid,batch_size = 1,shuffle = True,drop_last = True):
    X_train,X_valid = torch.Tensor(X_train),torch.Tensor(X_valid)
    y_train,y_valid = torch.Tensor(y_train),torch.Tensor(y_valid)
    dataset_train = TensorDataset(X_train,y_train)
    dataset_valid = TensorDataset(X_valid,y_valid)
    dataloader_train = DataLoader(dataset_train,batch_size = batch_size,shuffle = shuffle,drop_last = drop_last,)
    dataloader_valid = DataLoader(dataset_valid,batch_size = batch_size,shuffle = shuffle,drop_last = drop_last,)
    return dataloader_train,dataloader_valid

###############################################################################################################################
###############################################################################################################################
############################# this block is directly from: ####################################################################
############################ https://github.com/thuijskens/bayesian-optimization ##############################################
###############################################################################################################################
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize

def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement


def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False,
                               bounds=(0, 10), n_restarts=25):
    """ sample_next_hyperparameter
    Proposes the next hyperparameter to sample the loss function for.
    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.
    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x


def bayesian_optimisation(n_iters, sample_loss, bounds, x0=None, n_pre_samples=5,
                          gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7,
                          print_train = False,
                          patience = 5,
                          best_y = 0,
                          tol = 1e-4,):
    """ bayesian_optimisation
    Uses Gaussian Processes to optimise the loss function `sample_loss`.
    Arguments:
    ----------
        n_iters: integer.
            Number of iterations to run the search algorithm.
        sample_loss: function.
            Function to be optimised.
        bounds: array-like, shape = [n_params, 2].
            Lower and upper bounds on the parameters of the function `sample_loss`.
        x0: array-like, shape = [n_pre_samples, n_params].
            Array of initial points to sample the loss function for. If None, randomly
            samples from the loss function.
        n_pre_samples: integer.
            If x0 is None, samples `n_pre_samples` initial points from the loss function.
        gp_params: dictionary.
            Dictionary of parameters to pass on to the underlying Gaussian Process.
        random_search: integer.
            Flag that indicates whether to perform random search or L-BFGS-B optimisation
            over the acquisition function.
        alpha: double.
            Variance of the error term of the GP.
        epsilon: double.
            Precision tolerance for floats.
    """
    
    x_list = []
    y_list = []
    
    n_params = bounds.shape[0]
    
    if x0 is None:
        for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
            x_list.append(params)
            y_list.append(sample_loss(params))
    else:
        for params in x0:
            x_list.append(params)
            y_list.append(sample_loss(params))
            best_y = np.max(y_list)
            
    xp = np.array(x_list)
    yp = np.array(y_list)
    
    # Create the GP
    if gp_params is not None:
        model = gp.GaussianProcessRegressor(**gp_params)
    else:
        kernel = gp.kernels.Matern()
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            n_restarts_optimizer=10,
                                            normalize_y=False,
                                            random_state = 12345,)
    if print_train:
        iterator = tqdm(range(n_iters),desc='Bayesian optimization')
    else:
        iterator = range(n_iters)
    
    counter = 0
    for n in iterator:
        
        model.fit(xp, yp)
        
        # Sample next hyperparameter
        if random_search:
            x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
            ei = -1 * expected_improvement(x_random, model, yp, greater_is_better=True, n_params=n_params)
            next_sample = x_random[np.argmax(ei), :]
        else:
            next_sample = sample_next_hyperparameter(expected_improvement, 
                                                     model, 
                                                     yp, 
                                                     greater_is_better=True, # <--
                                                     bounds=bounds, 
                                                     n_restarts=100)
            
        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(next_sample - xp) <= epsilon):
            next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])
        
        # Sample loss for new set of parameters
        cv_score = sample_loss(next_sample)
        
        # Update lists
        x_list.append(next_sample)
        y_list.append(cv_score)
        
        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)
        
        if (cv_score > best_y) and (np.abs(cv_score - best_y) >= tol):
#            print(cv_score,np.abs(best_y - cv_score))
            best_y = cv_score
            counter = 0
        else:
            counter += 1
        
        if counter >= patience:
            return xp, yp
            break
        
    return xp, yp
####################################################################################################################
####################################################################################################################
def black_box_process(X_train,y_train,
                      X_valid,y_valid,
                      model,loss_function,device,
                      patience = 5,
                      batch_size = 1,
                      n_epochs = int(1e3),
                      learning_rate = 1e-4,
                      Bayesian_optimization_params = None,
                      print_train = False,
                      ):
    
    dataloader_train,dataloader_valid = train_valid_data_loader(X_train,
                                                                X_valid,
                                                                y_train,
                                                                y_valid,
                                                                batch_size = batch_size,
                                                                shuffle = True,
                                                                drop_last = True,
                                                                )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model,test_losses = fit(model,
                            device,
                            optimizer,
                            dataloader_train,
                            dataloader_valid,
                            loss_function = loss_function,
                            n_epochs = n_epochs,
                            patience = patience,
                            print_train = print_train,)
    return model,test_losses

def search_for_n_repeats(X_in,y_in,
                         X_test,y_test,
                         model,
                         Bayesian_optimization_params = None,
                         print_train = False,
                         n_jobs = -1,
                         verbose = 0,):
    
    # cross validate one of the hyperparameters
    def sample_loss(n_repeat):
        predictions = []
        def small_func(X):
            with torch.no_grad():
                model.eval()
                return model(torch.autograd.Variable(X))[0].detach().cpu().numpy()
        gc.collect()
        predictions = Parallel(n_jobs = n_jobs,verbose = verbose)(delayed(small_func)(**{
                'X':X_in}) for i in range(int(n_repeat)))
        gc.collect()
        predictions = np.array(predictions)
        pred = predictions.mean(0)
        score = r2_score(y_in,pred,multioutput = 'raw_values')
        n_voxels = np.sum(score >= 0 )
        if n_voxels < 1:
            return 0
        else:
            return score[score>=0].sum()
    if Bayesian_optimization_params != None:
        print('customized parameters')
        xp, yp = bayesian_optimisation(sample_loss = sample_loss,
                                       print_train = print_train,
                                       **Bayesian_optimization_params)
    else:
        xp, yp = bayesian_optimisation(n_iters = 20, 
                                       sample_loss = sample_loss, 
                                       bounds = np.array([[1,int(1e3)]]),
                                       n_pre_samples = 5,
                                       gp_params = {'random_state':12345},
                                       print_train = print_train,)
    best_n_repeat = xp[yp.argmin()]
    print(f'best n = {int(best_n_repeat)}')
    # testing phase
    X_test = torch.Tensor(X_test)
    def small_func(X):
        with torch.no_grad():
            model.eval()
            return model(torch.autograd.Variable(X))[0].detach().cpu().numpy()
    gc.collect()
    predictions = Parallel(n_jobs = n_jobs,verbose = verbose)(delayed(small_func)(**{
            'X':X_test}) for i in range(int(best_n_repeat)))
    gc.collect()
    predictions = np.array(predictions)
    pred = predictions.mean(0)
    score = r2_score(y_test,pred,multioutput = 'raw_values')
    corr = np.array([distance.cdist(a.reshape(1,-1),b.reshape(1,-1),'cosine',
                                   ).flatten()[0] for a,b in zip(y_test,pred)])
    return pred,score,corr