import os
import time
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset, TensorDataset
from collections import defaultdict


# for VAE
from models_gcn import VAE
from config import *

# for stochastic programming
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib notebook
import stochoptim.stochprob.facility_location.facility_location_problem as flp
import stochoptim.stochprob.facility_location.facility_location_solution as fls
import stochoptim.stochprob.facility_location.facility_location_uncertainty as flu
import stochoptim.scengen.scenario_tree as st
# for clustering
from sklearn_extra.cluster import KMedoids
np.set_printoptions(linewidth=120)


def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # first-stage dataset
    n_facility_locations = 10
    n_client_locations = 20
    n_zones = 1
    #------------
    client_node = []
    facility_node = [] 
    dist = []
    
    n_scenarios = 200
    p = 0.8
    #------------------
    scenarios = [] 
    
    objs = []
    for i in range(args.num_total_ins):
        param = flp.generate_random_parameters(n_facility_locations, n_client_locations, n_zones)
        pos_c = param['pos_client']
        pos_f = param['pos_facility']
        open_cost = param['opening_cost']/81.
        fac_cap = param['facility_capacity']/60.  
        facility_node.append(np.concatenate((pos_f,open_cost[:,None],fac_cap[:,None]),axis=1)) ### n_facility_locations*4
        client_node.append(pos_c)   ###  n_client_locations*2      
        #construct problem
        facility_problem = flp.FacilityLocationProblem(param)            

        client_uncertainty = flu.ClientsPresence(n_scenarios, n_client_locations, lb=0.4, ub=0.6)      
        scenario_tree = client_uncertainty.get_scenario_tree()
        scenarios.append(scenario_tree.to_numpy())
#         if i<args.num_labeled_ins:
#             for j in range(n_scenarios): 
#                 single_sce=st.twostage_from_scenarios(scenarios=scenario_tree.to_numpy()[j:j+1], 
#                                                      n_rvar={'h': 20})  
#                 actual_network_solution = facility_problem.solve(single_sce, verbose=0)       
#                 objs.append(actual_network_solution.objective_value)        
#             print("labeled instance_"+str(i))
        
        all_nodes = np.concatenate((pos_c,pos_f),axis=0)
        dist.append(np.linalg.norm(all_nodes[:, np.newaxis, :] - all_nodes[np.newaxis, :, :], axis=2)) ### (n_client+n_facility)^2
        
    dist_mat = np.array(dist)     
    tensor_dist = torch.Tensor(dist_mat)         
        
#     objs_mat = np.array(objs).reshape((-1, n_scenarios))/np.array(objs).max()    ### 128000*n_scenarios
#     tensor_obj = torch.Tensor(objs_mat)      
    
    tensor_c_fn = np.array(facility_node)   
    print(tensor_c_fn.shape)  
    tensor_c_fn = torch.Tensor(tensor_c_fn) 
    
    tensor_c_cn = np.array(client_node)      
    print(tensor_c_cn.shape)  
    tensor_c_cn = torch.Tensor(tensor_c_cn)     
    
    print(np.array(scenarios).shape)    
    tensor_x = torch.Tensor(np.array(scenarios))   
    
    
#     torch.save(tensor_dist, 'tensor_dist1020.pt')     
#     torch.save(tensor_obj, 'tensor_obj1020.pt')  
#     torch.save(tensor_c_fn, 'tensor_c_fn1020.pt')  
#     torch.save(tensor_c_cn, 'tensor_c_cn1020.pt')      
#     torch.save(tensor_x, 'tensor_x1020.pt')   

#     tensor_dist = torch.load('tensor_dist1020.pt')     
#     tensor_obj = torch.load('tensor_obj1020.pt')
#     tensor_c_fn = torch.load('tensor_c_fn1020.pt') 
#     tensor_c_cn = torch.load('tensor_c_cn1020.pt') 
#     tensor_x = torch.load('tensor_x1020.pt')  
    
    print(tensor_dist.size())  
#     print(tensor_obj.size())  
    print(tensor_c_fn.size())  
    print(tensor_c_cn.size())  
    print(tensor_x.size())      

    print('Data preparation is done.')  
    
    ### unlabeled data
    my_dataset = TensorDataset(tensor_x,tensor_c_cn, tensor_c_fn, tensor_dist)    
    data_loader = DataLoader(
        dataset=my_dataset, batch_size=args.batch_size, shuffle=True)     
    
    
    def loss_fn(recon_x, x, mean, log_var):
        
#        BCE = torch.nn.functional.mse_loss(
#            recon_x.view(-1, 20), x.view(-1, 20), reduction='sum')
        BCE = torch.nn.functional.binary_cross_entropy(recon_x.view(-1, 20), x.view(-1, 20), reduction='sum')        
        KLD = -0.005 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        
#         print(BCE)
#         print(KLD)

        return (BCE+KLD) / x.size(0)
    
    
    config_path = "configs/tsp50.json"
    config = get_config(config_path)
    print("Loaded {}:\n{}".format(config_path, config))
    vae = VAE(
        config=config,        
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=2 if args.conditional else 0).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):

        for iteration, (x, y_cn, y_fn, y_dist) in enumerate(data_loader):

            x, y_cn, y_fn, y_dist = x.to(device), y_cn.to(device), y_fn.to(device), y_dist.to(device)              
            
            if args.conditional:
                recon_x, mean, log_var, z, target, _, _ = vae(x, y_cn, y_fn, y_dist, epoch%n_scenarios)
            else:
                recon_x, mean, log_var, z, target, _, _ = vae(x)  

#             target = target*0.98+0.01                
            loss = loss_fn(recon_x, target, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))
                
        if epoch % args.save_net == 0 or epoch == args.epochs-1:
            # Specify a path
            PATH = "trained_nets/1saved_model_GCN_"+ str(epoch) + ".pt"
            # Save
            torch.save(vae.state_dict(), PATH)                


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[20, 256, 128])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[128, 256, 20])
    parser.add_argument("--latent_size", type=int, default=2) 
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--save_net", type=int, default=10)    
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')
    ### I add the followings
    parser.add_argument("--num_total_ins", type=int, default=128000)  
    parser.add_argument("--num_labeled_ins", type=int, default=128)      

    args = parser.parse_args()

    main(args)
