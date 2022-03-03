import torch
import torch.nn as nn

from utils import idx2onehot

import numpy as np

from gcn.gcn_model import ResidualGatedGCNModel

class CURL(nn.Module):

    def __init__(self, z_dim):
        super(CURL, self).__init__()

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.con = nn.Sigmoid() 
        self.m1 = nn.BatchNorm1d(128)
        self.m2 = nn.BatchNorm1d(128)
    
    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.t())  # (z_dim,B)
        Wz = self.m1(Wz.t())
        logits = torch.matmul(z_a, Wz.t())  # (B,B)
        #logits = logits - torch.max(logits, 1)[0][:, None]
        logits = self.con(logits)
#         print(logits)
        return logits

# class CURL(nn.Module):

#     def __init__(self, z_dim):
#         super(CURL, self).__init__()

#         self.W = nn.Parameter(torch.rand(z_dim, z_dim)) 
#         self.m1 = nn.BatchNorm1d(128)
#         self.m2 = nn.BatchNorm1d(128)
    
#     def compute_logits(self, z_a, z_pos):
#         """
#         Uses logits trick for CURL:
#         - compute (B,B) matrix z_a (W z_pos.T)
#         - positives are all diagonal elements
#         - negatives are all other elements
#         - to compute loss use multiclass cross entropy with identity matrix for labels
#         """
#         Wz = torch.matmul(self.W, z_pos.t())  # (z_dim,B)
#         Wz = self.m1(Wz.t())
#         logits = torch.matmul(z_a, Wz.t())  # (B,B)
#         #logits = logits - torch.max(logits, 1)[0][:, None]
# #         print(logits)
#         return logits


class VAE(nn.Module):

    def __init__(self, config, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels)
        
#        self.compress = nn.Sequential()

#         for i, (in_s, out_s) in enumerate(zip([14*14*3,128][:-1], [14*14*3,128][1:])):
#             self.compress.add_module(
#                 name="L{:d}".format(i), module=nn.Linear(in_s, out_s))
#             self.compress.add_module(name="A{:d}".format(i), module=nn.ReLU())       
            
#         self.linear_compress = nn.Linear(128, 2)  
#         self.m = nn.Sigmoid() 

        if torch.cuda.is_available():
            print("CUDA available, using GPU ID {}".format(config.gpu_id))
            dtypeFloat = torch.cuda.FloatTensor
            dtypeLong = torch.cuda.LongTensor
            torch.cuda.manual_seed(1)
        else:
            print("CUDA not available")
            dtypeFloat = torch.FloatTensor
            dtypeLong = torch.LongTensor
            torch.manual_seed(1)

        self.GCN = ResidualGatedGCNModel(config, dtypeFloat, dtypeLong)
        if torch.cuda.is_available():
            self.GCN.cuda()
#        print(self.GCN)        


    def forward(self, x, c=None, e=None):
        
        if e is not None:
            ind = e
            xt = x[:,ind]             
        else:
            ind = np.random.randint(x.size(1))            
            xt = x      
        
        
        d0,d1,d2,d3 = c.size()
        edges = torch.ones(d0,d3,d3)
        mask = torch.eye(d3, d3).byte()
        edges.masked_fill_(mask, 2)
        nodes = torch.ones(d0,d3)   

        c = c.permute(0,2,3,1)
        
        x = torch.cat((xt[:,0:2][None, :, :],xt[:,-2:][None, :, :],torch.zeros(d3-4,d0,2).cuda(),-xt[:,[0,2]][None, :, :],-xt[:,[1,3]][None, :, :])).transpose(0,1) 
        
        x0 = torch.zeros_like(x)

        x = self.GCN(edges.cuda().type(torch.cuda.LongTensor), c, nodes.cuda().type(torch.cuda.LongTensor), x)
        
        x0 = self.GCN(edges.cuda().type(torch.cuda.LongTensor), c, nodes.cuda().type(torch.cuda.LongTensor), x0)        
            
        means, log_var, obj, lat = self.encoder(x)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, x0)

        return recon_x, means, log_var, z, xt, obj, ind, lat

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        
        if self.conditional:
            layer_sizes[0] += num_labels

#         self.MLP = nn.Sequential()

#         for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
#             self.MLP.add_module(
#                 name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
#             self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)     
        
        self.obj = nn.Sequential()

        for i, (in_s, out_s) in enumerate(zip([layer_sizes[-1],128][:-1], [layer_sizes[-1],128][1:])):
            self.obj.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_s, out_s))
            self.obj.add_module(name="A{:d}".format(i), module=nn.ReLU())       
            
        self.linear_obj = nn.Linear(128, 1)  
        self.m_obj = nn.Sigmoid()         

    def forward(self, x):

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)        
        
        obj = self.linear_obj(self.obj(x))
        obj = self.m_obj(obj)             

        return means, log_vars, obj, x


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()    
        
        self.linear_dim_red = nn.Linear(layer_sizes[0], num_labels)          

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):

#         if self.conditional:
#             c = idx2onehot(c, n=10)
#             z = torch.cat((z, c), dim=-1)

        c = self.linear_dim_red(c)
        

        z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x
