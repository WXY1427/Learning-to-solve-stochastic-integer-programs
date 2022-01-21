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
import stochoptim.stochprob.network_design.network_design_problem as ndp
import stochoptim.stochprob.network_design.network_design_solution as nds
import stochoptim.stochprob.network_design.network_design_uncertainty as ndu
from stochoptim.scenclust.cost_space_partition import CostSpaceScenarioPartitioning
import stochoptim.scengen.scenario_tree as st
np.set_printoptions(linewidth=120)


def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()

#    dataset = MNIST(
#        root='data', train=True, transform=transforms.ToTensor(),
#        download=True)
#    data_loader = DataLoader(
#        dataset=dataset, batch_size=args.batch_size, shuffle=True)


    # first-stage dataset
    n_origins = 2
    n_destinations = 2
    n_intermediates = 10
    #------------
    opening_cost = []
    shipping_cost = []
    capacity = []
    
    
    n_scenarios = 200
    distribution = 'uniform'
    lb = 5
    ub = 20
    corr = 0.4
    #------------------
#     scenarios = [] 
    
#     objs = []
#     for i in range(args.num_total_ins):
#         param = ndp.generate_random_parameters(n_origins, n_destinations, n_intermediates)
#         opening_cost.append(param['opening_cost'])
#         shipping_cost.append(param['shipping_cost'])
#         capacity.append(param['capacity'])  
#         #construct problem
#         network_problem = ndp.NetworkDesign(param)
#         network_problem      
#         #generate scenarios
#         network_uncertainty = ndu.Demands(n_scenarios=n_scenarios,
#                                   n_commodities=n_origins*n_destinations, 
#                                   distribution=distribution,
#                                   lb=lb, 
#                                   ub=ub, 
#                                   corr=corr)              
#         scenario_tree = network_uncertainty.get_scenario_tree()              
#         scenarios.append(scenario_tree.to_numpy()/20)
#         if i<args.num_labeled_ins:
#             for j in range(n_scenarios):
#                 single_sce=st.twostage_from_scenarios(scenarios=scenario_tree.to_numpy()[j:j+1], 
#                                                      n_rvar={'d': 4})            
#                 actual_network_solution = network_problem.solve(single_sce, verbose=0)
#                 objs.append(actual_network_solution.objective_value)        
#             print("labeled instance_"+str(i))
#    objs_mat = np.array(objs).reshape((-1, n_scenarios))/np.array(objs).max()    ### 12800*n_scenarios
#    tensor_obj = torch.Tensor(objs_mat)      
    
#     tensor_c = np.stack([np.array(opening_cost),np.array(shipping_cost),np.array(capacity)]).transpose((1,0,2,3))    
#     print(tensor_c.shape)  
#     tensor_c[tensor_c==1000] = 0
#     tensor_c[tensor_c==100000] = 0
#     tensor_c = torch.Tensor(tensor_c)    
    
#     print(np.array(scenarios).shape)    
#     tensor_x = torch.Tensor(np.array(scenarios))   
    
#    torch.save(tensor_obj, 'tensor_obj.pt')  
#     torch.save(tensor_c, 'tensor_c_10x.pt')  
#     torch.save(tensor_x, 'tensor_x_10x.pt')   

#     tensor_obj = torch.load('tensor_obj.pt')
    tensor_c = torch.load('tensor_c_10x.pt')  
    tensor_x = torch.load('tensor_x_10x.pt')    
    
    ### unlabeled data
    my_dataset = TensorDataset(tensor_x,tensor_c)    
    data_loader = DataLoader(
        dataset=my_dataset, batch_size=args.batch_size, shuffle=True)     
    
    
    def loss_fn(recon_x, x, mean, log_var):
#        print(recon_x)
#        print(x)
        
        BCE = torch.nn.functional.mse_loss(
            recon_x.view(-1, 4), x.view(-1, 4), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD) / x.size(0)
    
    
    config_path = "configs/tsp10.json"
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

        for iteration, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)            
            
            if args.conditional:
                recon_x, mean, log_var, z, target, _, _ = vae(x, y, epoch%n_scenarios)
            else:
                recon_x, mean, log_var, z, target, _, _ = vae(x)  

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
            PATH = "trained_nets/saved_model_GCN_4d_"+ str(epoch) + ".pt"
            # Save
            torch.save(vae.state_dict(), PATH)                

#                 if args.conditional:
#                     c = torch.arange(0, 10).long().unsqueeze(1).to(device)
#                     z = torch.randn([c.size(0), args.latent_size]).to(device)
#                     x = vae.inference(z, c=c)
#                 else:
#                     z = torch.randn([10, args.latent_size]).to(device)
#                     x = vae.inference(z)

#                 plt.figure()
#                 plt.figure(figsize=(5, 10))
#                 for p in range(10):
#                     plt.subplot(5, 2, p+1)
#                     if args.conditional:
#                         plt.text(
#                             0, 0, "c={:d}".format(c[p].item()), color='black',
#                             backgroundcolor='white', fontsize=8)
#                     plt.imshow(x[p].view(28, 28).cpu().data.numpy())
#                     plt.axis('off')

#                 if not os.path.exists(os.path.join(args.fig_root, str(ts))):
#                     if not(os.path.exists(os.path.join(args.fig_root))):
#                         os.mkdir(os.path.join(args.fig_root))
#                     os.mkdir(os.path.join(args.fig_root, str(ts)))

#                 plt.savefig(
#                     os.path.join(args.fig_root, str(ts),
#                                  "E{:d}I{:d}.png".format(epoch, iteration)),
#                     dpi=300)
#                 plt.clf()
#                 plt.close('all')

#         df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
#         g = sns.lmplot(
#             x='x', y='y', hue='label', data=df.groupby('label').head(100),
#             fit_reg=False, legend=True)
#         g.savefig(os.path.join(
#             args.fig_root, str(ts), "E{:d}-Dist.png".format(epoch)),
#             dpi=300)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[4, 128])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[128, 4])
    parser.add_argument("--latent_size", type=int, default=2) 
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--save_net", type=int, default=25)    
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')
    ### I add the followings
    parser.add_argument("--num_total_ins", type=int, default=128000)  
    parser.add_argument("--num_labeled_ins", type=int, default=128)      

    args = parser.parse_args()

    main(args)
