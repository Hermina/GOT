import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import torch
torch.set_default_tensor_type('torch.DoubleTensor')
import random

import numpy.linalg as lg
import scipy.linalg as slg
import sklearn
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

import copy

# ---------------------------------------------------------------------------------------------------------------

def create_permutation(n, l1, seed_nb):
    np.random.seed(seed_nb)
    idx = np.random.permutation(n)
    P_true = np.eye(n);
    P_true = P_true[idx, :]
    l2 = np.array(P_true @ l1 @ P_true.T)
    
    return np.double(l2), P_true

def create_test_graphs(n,  block_sizes = [], block_prob = [], graph_type = 'inv_cov', seed_nb = 123):
    if graph_type == 'inv_cov':
        l1 = sklearn.datasets.make_spd_matrix(n)
    elif graph_type == 'cov':
        l1 = sklearn.datasets.make_spd_matrix(n)
        l1 = lg.inv(l1)
    else:
        if graph_type == 'geo':
            g1 = nx.random_geometric_graph(n, 0.55)
        if graph_type == 'er':
            g1 = nx.erdos_renyi_graph(n, 0.45)
        if graph_type == 'sbm':
            g1 = nx.stochastic_block_model(block_sizes, block_prob, seed = seed_nb)
        g1.remove_nodes_from(list(nx.isolates(g1)))
        n = len(g1)
        l1 = nx.laplacian_matrix(g1,range(n))
        l1 = np.array(l1.todense())
        
    # Permutation and second graph
    l2, P_true = create_permutation(n, l1, seed_nb)
    x = np.double(l1)
    y = l2
    return [x, y, P_true]

# ---------------------------------------------------------------------------------------------------------------

def graph_from_laplacian(L):
    A = -L.copy()
    np.fill_diagonal(A, 0)
    G = nx.from_numpy_array(A)
    return G

def change_graph(L, nb_edges, seed):
    rng = np.random.RandomState(seed)
    
    G = graph_from_laplacian(L)
    edges = np.triu(L, k=1).nonzero()
    
    removed = 0
    for idx in rng.permutation(edges[0].size):
        u, v = edges[0][idx], edges[1][idx]

        G.remove_edge(u,v)
        if nx.is_connected(G):
            removed += 1
        else:
            G.add_edge(u,v)

        if removed == nb_edges:
            break
     
    return nx.laplacian_matrix(G).todense()

# ---------------------------------------------------------------------------------------------------------------

def doubly_stochastic(P, tau, it):
    """Uses logsumexp for numerical stability."""
    
    A = P / tau
    for i in range(it):
        A = A - A.logsumexp(dim=1, keepdim=True)
        A = A - A.logsumexp(dim=0, keepdim=True)
    return torch.exp(A)

def wasserstein_initialisation(A, B):
    #Wasserstein directly on covariance
    Root_1= slg.sqrtm(A)
    Root_2= slg.sqrtm(B)
    C1_tilde = torch.from_numpy(Root_1.astype(np.double))
    C2_tilde = torch.from_numpy(Root_2.astype(np.double)) 
    return [C1_tilde, C2_tilde]

def loss(DS, x, y, params, loss_type):
    if loss_type == 'w':
        [C1_tilde, C2_tilde] = params
        loss_c = torch.trace(x) + torch.trace(torch.transpose(DS,0,1) @ y @ DS)
        # svd version
        u, sigma, v = torch.svd(C2_tilde @ DS @ C1_tilde)
        cost = loss_c - 2* torch.sum(sigma) #torch.abs(sigma))

    elif loss_type == 'kl':  
        yy = torch.transpose(DS,0,1) @ y @ DS
        term1 = torch.trace(torch.inverse(x) @ yy)
        K     = x.shape[0]
        term2 = torch.logdet(x) - torch.logdet(yy)
        cost = 0.5*(term1 - K + term2)
    else: # l2
        cost = torch.sum((y @ DS - DS @ x)**2, dim=1).sum()
    return cost

def show_matrix(P, name=None, **args):
    plt.rcParams.update({'font.size': 24})
    fig = plt.figure(figsize=(8,8))
    plt.pcolor(P[::-1], edgecolors='lightblue', linewidth=1, cmap='Blues', **args)
    plt.axis('square')
    plt.xticks(np.arange(len(P))+0.5, np.arange(len(P)))
    plt.yticks(np.arange(len(P))+0.5, np.arange(len(P))[::-1])
    plt.gca().xaxis.tick_top()
    plt.colorbar()
    plt.show()
    if name is not None:
        fig.savefig(name+'.png', dpi=fig.dpi, pad_inches=0, bbox_inches='tight')


def regularise_and_invert(x, y, alpha, ones):
    x_reg = regularise_invert_one(x, alpha, ones)
    y_reg = regularise_invert_one(y, alpha, ones)
    return [x_reg, y_reg]

def regularise_invert_one(x, alpha, ones):
    if ones:
        x_reg = lg.inv(x   + alpha * np.eye(len(x)) + np.ones([len(x),len(x)])/len(x)) 
    else:
        x_reg = lg.pinv(x) + alpha * np.eye(len(x))
    return x_reg

def find_permutation(x, y, it, tau, n_samples, epochs, lr, loss_type = 'l2', alpha = 0, ones = True, graphs = True):
    if graphs:
        [x_reg, y_reg] = regularise_and_invert(x, y, alpha, ones)
    else:
        x_reg = x
        y_reg = y
    return x_reg, y_reg, permutation_stochastic(x_reg, y_reg, it, tau, n_samples, epochs, lr, loss_type)        

# ---------------------------------------------------------------------------------------------------------------

def permutation_stochastic(A, B, it, tau, n_samples, epochs, lr, loss_type = 'w', seed=42, verbose=True):
    
    # NumPy -> PyTorch
    x = torch.from_numpy(A.astype(np.double))
    y = torch.from_numpy(B.astype(np.double))
    
    # Initialization
    torch.manual_seed(seed)
    n = x.shape[0]
    mean = torch.rand(n, n, requires_grad=True)
    std  = 10 * torch.ones(n, n)
    std  = std.requires_grad_()
    params = []
    
    if loss_type == 'w':
        params = wasserstein_initialisation(A, B)

    # Optimization
    optimizer = torch.optim.Adam([mean,std], lr=lr, amsgrad=True)
    history = []
    for epoch in range(epochs):
        cost = 0;
        cost_vec = np.zeros((1,n_samples))
        for sample in range(n_samples):
            # Sampling
            eps = torch.randn(n, n)
            P_noisy = mean + std * eps #torch.log(1+torch.exp(std)) * eps

            # Cost function
            DS = doubly_stochastic(P_noisy, tau, it)
            cost = cost + loss(DS, x, y, params, loss_type)
            cost_vec[0,sample] = loss(DS, x, y, params, loss_type)
        cost = cost/n_samples
        #print(cost_vec)
        
        # Gradient step
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Tracking
        history.append(cost.item())
        if verbose and (epoch==0 or (epoch+1) % 100 == 0):
            print('[Epoch %4d/%d] loss: %f - std: %f' % (epoch+1, epochs, cost.item(), std.detach().mean()))
            
    # PyTorch -> NumPy
    P = doubly_stochastic(mean, tau, it)
    #print('a la fin' + str(loss(P, x, y, params, loss_type)))
    P = P.squeeze()
    P = P.detach().numpy()
    
    # Keep the max along the rows
    idx = P.argmax(1)
    P = np.zeros_like(P)
    P[range(n),idx] = 1.
    
    # Convergence plot
    plt.plot(history)
    plt.show()
    
    return P

# ---------------------------------------------------------------------------------------------------------------

def show_network(G, y=None, labels=None, pos=None, ax=None, figsize=(5,5)):

    if ax is None:
        plt.figure(figsize=figsize)  # image is 8 x 8 inches
        plt.axis('off')
        ax = plt.gca()

    if pos is None:
        pos = nx.kamada_kawai_layout(G)
        
    nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.8, cmap=plt.cm.RdYlGn, node_color=y, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax)
            
    if labels is None:
        nx.draw_networkx_labels(G, pos, font_color='w', font_weight='bold', font_size=15, ax=ax)
    else:
        labeldict = {}
        for i, v in enumerate(G.nodes):
            labeldict[v] = labels[i]
        nx.draw_networkx_labels(G, pos, font_color='w', font_weight='bold', font_size=15, labels=labeldict, ax=ax)

# ---------------------------------------------------------------------------------------------------------------        

def show_permutation(P, P_true):
    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.imshow(P)
    plt.title("Estimated P")

    plt.subplot(1,3,2)
    plt.imshow(P_true)
    plt.title("True P")

    plt.subplot(1,3,3)
    plt.imshow(P_true-P)
    plt.title("Differences in P")

    plt.show()

def plot_cov(x, y, P):
    plt.figure(figsize=(15,10))

    plt.subplot(1,3,1)
    plt.imshow(x)
    plt.title("x")

    plt.subplot(1,3,2)
    plt.imshow( P.T @ y @ P)
    plt.title("P.T @ y @ P")

    plt.subplot(1,3,3)
    plt.imshow(np.abs(x -  P.T @ y @ P))
    plt.title("Absolute errors in L1")

    plt.show()

def plot_graphs(x, y, x_inv, y_inv, P):

    plt.figure(figsize=(15,10))

    plt.subplot(2,3,1)
    plt.imshow(x)
    plt.title("x")

    plt.subplot(2,3,2)
    plt.imshow( P.T @ y @ P)
    plt.title("P.T @ y @ P")

    plt.subplot(2,3,3)
    plt.imshow(np.abs(x -  P.T @ y @ P))
    plt.title("Absolute errors in L1")

    plt.subplot(2,3,4)
    plt.imshow(x_inv)
    plt.title("lg.inv(x)")

    plt.subplot(2,3,5)
    plt.imshow( P.T@ y_inv @P)
    plt.title("P.T @ lg.inv(y) @ P")

    plt.subplot(2,3,6)
    plt.imshow(np.abs( x_inv - P.T@ y_inv @P))
    plt.colorbar()
    plt.title("Absolute errors in inverted L1")

    plt.show()

def wass_dist(A, B):
    Root_1= slg.sqrtm(A)
    Root_2= slg.sqrtm(B)
    return np.trace(A) + np.trace(B) - 2*np.trace(slg.sqrtm(Root_1 @ B @ Root_1)) 

def plot_everything(x, y, x_inv, y_inv, P, P_true, cov = False):

    params = wasserstein_initialisation(x_inv, y_inv)
    
    #print("------------------------------------------")
    #print("Wasserstein -- result")
    #print("------------------------------------------")
    
    #print('Wasserstein distance', wass_dist(x_inv, P.T @ y_inv @ P))
    #print('L2 distance', (lg.norm(y @ P - P @ x))**2)
    
    show_permutation(P, P_true)
    if cov:
        plot_cov(x, y, P)
    else:
        plot_graphs(x, y, x_inv, y_inv, P)

# ---------------------------------------------------------------------------------------------------------------        

def plot_graphs_ch(x_ch, y_ch, P_true, P_est=None, pos = None):
    
    if P_est is None:
        P_est = np.eye(*P_true.shape)
    
    N_nodes = x_ch.shape[0]
    A1 = -x_ch.copy()
    np.fill_diagonal(A1, 0)
    G1 = nx.from_numpy_array(A1)

    A2 = -y_ch.copy()
    np.fill_diagonal(A2, 0)
    G2 = nx.from_numpy_array(P_true.T @ A2 @ P_true)
    
    n1 = np.arange(N_nodes)
    n2 = (P_est.T@P_true@n1).astype(int)

    
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].axis('off')
    ax[1].axis('off')
    
    pos = nx.kamada_kawai_layout(G1)
    
    show_network(G1, pos=pos, ax=ax[0])
    show_network(G2, pos=pos, labels=n2, y=np.ones(N_nodes), ax=ax[1])

    #fig.savefig('graph2.png', dpi=fig.dpi, pad_inches=0, bbox_inches='tight')
    
    plt.show()
