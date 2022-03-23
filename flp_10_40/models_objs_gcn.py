import torch
import torch.nn as nn

from utils import idx2onehot

import numpy as np

from gcn.gcn_model import ResidualGatedGCNModel


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

        self.linear_cn = nn.Linear(3, 64)        
        self.linear_fn = nn.Linear(4, 64)                    
            
    def forward(self, x, y_cn=None, y_fn=None, y_dist=None, e=None):
        
        if e is not None:
            ind = e
        else:
            ind = np.random.randint(x.size(1))            
    
        xt = x[:,ind]    
        
        bs, num_cn, _, = y_cn.size()
        _, num_fn, _, = y_fn.size()        
        
        y_cn0 = torch.cat((y_cn,-torch.ones(bs, num_cn, 1).cuda()),2)
        y_cns = torch.cat((y_cn,xt[:,:,None]),2)   
        y_cn0 = self.linear_cn(y_cn0)
        y_cns = self.linear_cn(y_cns)  
        
        y_fn = self.linear_fn(y_fn)
        
        y_cn0 = torch.cat((y_cn0,y_fn),1)  
        y_cns = torch.cat((y_cns,y_fn),1)   
        
        d3 = num_cn+num_fn
        edges = torch.ones(bs,d3,d3)
        mask = torch.eye(d3, d3).byte()
        edges.masked_fill_(mask, 2)
        nodes = torch.ones(bs,d3)   

#         y_dist = y_dist[:,:,:,None]

        y_dist = torch.ones(bs,d3,d3,1).type(torch.cuda.FloatTensor)

        x = self.GCN(edges.cuda().type(torch.cuda.LongTensor), y_dist, nodes.cuda().type(torch.cuda.LongTensor), y_cns)
        
        x0 = self.GCN(edges.cuda().type(torch.cuda.LongTensor), y_dist, nodes.cuda().type(torch.cuda.LongTensor), y_cn0)               
            
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

        c = self.linear_dim_red(c)        

        z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x
