import torch
from torch import nn
import numpy as np

from cnf.components import BatchLinears, FeatureMapping, NLS_AND_INITS, FourierLayer, GaborLayer
from cnf.initialization import *
from einops.layers.torch import Rearrange

# auto encoder: full projection 
# siren based 
# auto decoder: full projection 
# in construction 
class SIRENAutodecoder_film(nn.Module):
    '''
    siren network with author decoding 
    '''
    def __init__(self, in_coord_features, in_latent_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', weight_init=None,bias_init=None,
                 premap_mode = None, **kwargs):
        super().__init__()
        self.premap_mode = premap_mode
        if not self.premap_mode ==None: 
            self.premap_layer = FeatureMapping(in_coord_features,mode = premap_mode, **kwargs)
            in_coord_features = self.premap_layer.dim # update the nf in features     

        self.first_layer_init = None
                        
        self.nl, nl_weight_init, first_layer_init = NLS_AND_INITS[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init # those are default init funcs 

        #self.net1 = nn.ModuleList([nn.Linear(in_coord_features,hidden_features)] + 
        #                          [nn.Linear(hidden_features,hidden_features) for i in range(num_hidden_layers)] 
        #                          )#+ 
        
        #self.net3 = nn.ModuleList([nn.Linear(hidden_features,hidden_features) for i in range(6)])
        n = 16
        self.encoder = nn.Sequential(nn.Linear(128*3,hidden_features*n),
                                     nn.LayerNorm(hidden_features*n),
                                     nn.GELU(),
                                     nn.Linear(hidden_features*n,hidden_features*n),
                                     nn.LayerNorm(hidden_features*n),
                                     nn.GELU(),
                                     #nn.Linear(hidden_features*n,hidden_features*n),
                                     #nn.LayerNorm(hidden_features*n),
                                     #nn.GELU(),
                                     nn.Linear(hidden_features*n, in_latent_features),
                                     #nn.BatchNorm1d(in_latent_features)
                                     nn.LayerNorm(in_latent_features)
                                     ) 
        self.net1 = nn.Sequential(nn.Linear(in_coord_features,hidden_features,bias=False),
                                  #nn.LayerNorm(hidden_features),
                                  nn.Tanh(),
                                  #nn.Linear(hidden_features, hidden_features, bias=False),
                                  #nn.LayerNorm(hidden_features),
                                  )
        self.net2 = nn.Sequential(
                                  #nn.LayerNorm(in_latent_features),
                                  nn.Linear(in_latent_features,hidden_features, bias=False),
                                  nn.Tanh(),
                                  #nn.Linear(hidden_features, hidden_features,bias=False),
                                  #nn.LayerNorm(hidden_features),
                                                  )
        #self.net3 = nn.Sequential(nn.Linear(hidden_features,hidden_features,bias=False),
        #                          nn.GELU(),
                                  #nn.LayerNorm(hidden_features),
        #                          )
        #self.net4 = nn.Sequential(
        #        nn.Linear(in_latent_features,hidden_features, bias=False),
                #nn.Tanh(),
                #nn.LayerNorm(hidden_features),
        #        )
        
        self.tanh = nn.Tanh()
        self.net3 = nn.ModuleList([nn.Linear(hidden_features,hidden_features, bias = False) for i in range(4)])
        self.out = nn.Sequential(
                                 #nn.LayerNorm(hidden_features),
                                 #nn.Tanh(),
                                 #nn.Linear(hidden_features, hidden_features, bias=False),
                                 #nn.LayerNorm(hidden_features),
                                 #nn.Tanh(),
                                 #nn.ELU(),
                                 #nn.Linear(hidden_features, hidden_features,bias=False),
                                 #nn.LayerNorm(hidden_features),
                                 #nn.ELU(),
                                 #nn.Tanh(),
                                 nn.Linear(hidden_features,out_features,bias=False))


    def forward(self, coords, latents):
        # coords: <t,h,w,c_coord>
        # latents: <t,h,w,c_latent>
        # premap 
        if not self.premap_mode ==None: 
            x = self.premap_layer(coords)
        else: 
            x = coords
        
        latents = self.encoder(latents.reshape(-1,1,128*3))
        # pass it through  the nf network 
        x = self.net1(x) + self.net2(latents)
        #x = self.net3(x) + self.net4(latents)
        for i in range(4):
            x = self.net3[i](x) + x
            x = self.tanh(x)
        #print(x.shape)
        #x = self.bn(x)
        x = self.out(x)
        return x 

    def gen_latent(self, coords, latents):
        latents = self.encoder(latents.reshape(-1,1,128*3))
        return latents
    def infer(self, coords, latents):
        x = coords
        x = self.net1(x) + self.net2(latents)
        for i in range(4):
            x = self.net3[i](x) + x
            x = self.tanh(x)
        #x = self.nl(x)
        x = self.out(x)
        return x

    def disable_gradient(self):
        for param in self.parameters():
            param.requires_grad = False


class SIRENAutodecoder_film_extra_in(SIRENAutodecoder_film):
    def forward(self, coord, latents):
        coord = torch.concat(
            [torch.ones_like(coord[0][...,:1])*coord[1], coord[0]], dim = -1
        )
        return super().forward(coord, latents)

# fn auto decoder: FILM
class FNAutodecoder_film(nn.Module):
    '''
    fourier network with author decoding 
    '''
    def __init__(self, in_coord_features, in_latent_features, out_features, num_hidden_layers, hidden_features,
                 bias=True, output_act=False,input_scale=256.0,weight_scale=1.0,alpha=6.0,beta=1.0
                 ,premap_mode = None, **kwargs):
        super().__init__()
        self.premap_mode = premap_mode
        if not self.premap_mode ==None: 
            self.premap_layer = FeatureMapping(in_coord_features,mode = premap_mode, **kwargs)
            in_coord_features = self.premap_layer.dim # update the nf in features     
            print('premap dim:',self.premap_layer.dim)
        # assemble the MFN
        self.net1 = nn.ModuleList(
            [nn.Linear(hidden_features, hidden_features, bias) for _ in range(num_hidden_layers)]+[nn.Linear(hidden_features, out_features)]
        )

        # assemble the FILM
        self.net2 = nn.ModuleList([nn.Linear(in_latent_features, hidden_features,bias = False ) for _ in range(num_hidden_layers+1)])

        # initialize MFN
        for lin in self.net1:
            in_dim = lin.weight.shape[1]
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / in_dim),
                np.sqrt(weight_scale / in_dim),
            )

        # initialize FILM
        for lin in self.net2:
            in_dim = lin.weight.shape[1]
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / in_dim),
                np.sqrt(weight_scale / in_dim),
            )

        # initialize filters 
        self.filters = nn.ModuleList(
            [
                FourierLayer(in_coord_features, hidden_features, input_scale / np.sqrt(num_hidden_layers + 1))
                for _ in range(num_hidden_layers + 1)
            ]
        )


    def forward(self, coords, latents):
        # coords: <t,h,w,c_coord>
        # latents: <t,h,w,c_latent>
        
        # premap 
        if not self.premap_mode ==None: 
            x0 = self.premap_layer(coords)
        else: 
            x0 = coords

        x = self.filters[0](x0)*(self.net2[0](latents))

        # pass it through  the nf network 
        for i in range(1,len(self.filters)):
            x = self.filters[i](x0)*(self.net1[i-1](x)+self.net2[i](latents))
        x = self.net1[-1](x)
        return x 


# auto decoder: FILM
class GNAutodecoder_film(nn.Module):
    '''
    siren network with author decoding 
    '''
    def __init__(self, in_coord_features, in_latent_features, out_features, num_hidden_layers, hidden_features,
                 bias=True, output_act=False,input_scale=256.0,weight_scale=1.0,alpha=6.0,beta=1.0
                 ,premap_mode = None, **kwargs):
        super().__init__()
        self.premap_mode = premap_mode
        if not self.premap_mode ==None: 
            self.premap_layer = FeatureMapping(in_coord_features,mode = premap_mode, **kwargs)
            in_coord_features = self.premap_layer.dim # update the nf in features     
            print('premap dim:',self.premap_layer.dim)
        # assemble the MFN
        self.net1 = nn.ModuleList(
            [nn.Linear(hidden_features, hidden_features, bias) for _ in range(num_hidden_layers)]+[nn.Linear(hidden_features, out_features)]
        )

        # assemble the FILM
        self.net2 = nn.ModuleList([nn.Linear(in_latent_features, hidden_features,bias = False ) for _ in range(num_hidden_layers+1)])

        # initialize MFN
        for lin in self.net1:
            in_dim = lin.weight.shape[1]
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / in_dim),
                np.sqrt(weight_scale / in_dim),
            )

        # initialize FILM
        for lin in self.net2:
            in_dim = lin.weight.shape[1]
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / in_dim),
                np.sqrt(weight_scale / in_dim),
            )

        # initialize filters 
        self.filters = nn.ModuleList(
            [
                GaborLayer(
                    in_coord_features,
                    hidden_features,
                    input_scale / np.sqrt(num_hidden_layers + 1),
                    alpha / (num_hidden_layers + 1),
                    beta,
                )
                for _ in range(num_hidden_layers + 1)
            ]
        )

    def forward(self, coords, latents):
        # coords: <t,h,w,c_coord>
        # latents: <t,h,w,c_latent>
        
        # premap 
        if not self.premap_mode ==None: 
            x0 = self.premap_layer(coords)
        else: 
            x0 = coords

        x = self.filters[0](x0)*(self.net2[0](latents))

        # pass it through  the nf network 
        for i in range(1,len(self.filters)):
            x = self.filters[i](x0)*(self.net1[i-1](x)+self.net2[i](latents))
        x = self.net1[-1](x)
        return x 


class SIRENAutodecoder_fp(nn.Module):
    '''
    siren network with author decoding 
    '''
    def __init__(self, hyper_latent_features,
                 nf_in_features, out_features , nf_num_hidden_layers, nf_hidden_features, hyper_nonlinearity = 'sine',nf_nonlinearity= 'sine',
                 omega_0_e = DEFAULT_W0, omega_0 = DEFAULT_W0,
                 premap_mode = None, **kwargs):
        super().__init__()
        self.premap_mode = premap_mode
        if not self.premap_mode ==None: 
            self.premap_layer = FeatureMapping(nf_in_features,mode = premap_mode, **kwargs)
            nf_in_features = self.premap_layer.dim # update the nf in features 

        self.nf_in_features = nf_in_features
        self.out_features = out_features
        self.nf_num_hidden_layers = nf_num_hidden_layers
        self.nf_hidden_features = nf_hidden_features

        self.hyper_latent_features = hyper_latent_features
        self.hyper_out_features = self.calculate_num_parameters(nf_in_features, out_features , nf_num_hidden_layers, nf_hidden_features)
        self.omega_0_e = omega_0_e 
        self.omega_0  = omega_0 
        self.first_layer_init = None
        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        # different layers has different initialization schemes: 
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init), # act name, init func, first layer init func 
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None),
                         'swish':(Swish(), init_weights_xavier, None),
                         }

        self.nf_nl, _,_ = nls_and_inits[nf_nonlinearity]
        # set omega:
        self.nf_nl.w0 = omega_0
        # print('init hyper dims: ', hyper_latent_features,self.hyper_out_features)
        self.hyper_net_last = BatchLinear(hyper_latent_features,self.hyper_out_features)
        # print('init hyper weight shape: ', self.hyper_net_last.weight.data.shape)
        self.init_hyper_layer()

    def forward(self, coords, latents):
        # coords: <t,h,w,c_coord> or <1,h,w,c_coord> 
        # latents: <t,h,w,c_priors> or <t,1,1,c_priors>
        params = self.hyper_net_last(latents) # <t,1,1,params>
        print('param shape:', params.shape)
        
        # fill x into nf net, full projection 
        cursors = [0]
        cursors.append(cursors[-1] + self.nf_in_features * self.nf_hidden_features)
        w1 = params[..., cursors[-2]:cursors[-1]].reshape(params.shape[:-1]+(self.nf_in_features,self.nf_hidden_features))
        hidden_ws = []
        for i in range(self.nf_num_hidden_layers):
            cursors.append(cursors[-1] + self.nf_hidden_features*self.nf_hidden_features)
            hidden_ws.append(params[..., cursors[-2]: cursors[-1]].reshape(params.shape[:-1]+(self.nf_hidden_features,self.nf_hidden_features)))
        
        cursors.append(cursors[-1] + self.nf_hidden_features*self.out_features)
        w2 = params[..., cursors[-2]:cursors[-1]].reshape(params.shape[:-1]+(self.nf_hidden_features,self.out_features))
        
        # fill bias
        cursors.append(cursors[-1] + self.nf_hidden_features)
        b1 = params[..., cursors[-2]: cursors[-1]]

        hidden_bs = []
        for i in range(self.nf_num_hidden_layers):
            cursors.append(cursors[-1] + self.nf_hidden_features)
            hidden_bs.append(params[..., cursors[-2]: cursors[-1]])
        
        b2 = params[..., cursors[-1]:]
        
        # propagate through the nf net, implementing SIREN
        if not self.premap_mode ==None: 
            out = self.premap_layer(coords)
        else: 
            out = coords

        # pass it through the nf network 
        # here out:<1,h,w,c_coord>; w1: <t,1,1,c_oords, hidden_dim>; b1: <t,1,1,hidden_dim>
        out = self.nf_nl(torch.einsum('thwi,thwij->thwj', out, w1) + b1)

        for i in range(self.nf_num_hidden_layers):
            # out = torch.sin(self.omega_0 * torch.einsum('thwi,thwij->thwj', out, hidden_ws[i]) + hidden_bs[i])
            # here out:<t,h,w,hidden_dim>; w1: <t,1,1,hidden_dim, hidden_dim>; b1: <t,1,1,hidden_dim>
            out = self.nf_nl(torch.einsum('thwi,thwij->thwj', out, hidden_ws[i]) + hidden_bs[i])
        
        print('ensum shapes: ' , out.shape, w2.shape, b2.shape)
        # here out:<t,h,w,hidden_dim>; w1: <t,1,1,hidden_dim, out_dim>; b2: <t,1,1,out_dim>
        out = torch.einsum('thwi,thwij->thwj', out, w2) + b2
        return out
    
    
    def init_hyper_layer(self):
        # init weights
        w_init = nn.init.uniform_(torch.empty(self.hyper_out_features, self.hyper_latent_features),a=-np.sqrt(6.0 / self.hyper_latent_features) * 1e-2, b=np.sqrt(6.0 / self.hyper_latent_features) * 1e-2)
        # init bias
        tmp = torch.ones((self.hyper_out_features))
        tmp[:self.nf_in_features * self.nf_hidden_features] = tmp[:self.nf_in_features * self.nf_hidden_features] * 1.0 / self.nf_in_features  # 1st layer weights
        temp_num_weights =self.nf_in_features * self.nf_hidden_features+ self.nf_num_hidden_layers*self.nf_hidden_features*self.nf_hidden_features + self.nf_hidden_features*self.out_features
        tmp[self.nf_in_features * self.nf_hidden_features:temp_num_weights] = tmp[self.nf_in_features * self.nf_hidden_features:temp_num_weights] * \
                                                            np.sqrt(6.0 /self.nf_hidden_features) / self.omega_0_e  # other layer weights
        tmp[temp_num_weights:] = 1.0 / self.nf_hidden_features  # all biases
        b_init = torch.distributions.uniform.Uniform(low= -tmp,high= tmp).sample().squeeze(-1)   # uniformly distributed in the range

        with torch.no_grad():
            self.hyper_net_last.weight.data = w_init 
            self.hyper_net_last.bias.data = b_init 

    @staticmethod
    def calculate_num_parameters(nf_in_features, out_features , nf_num_hidden_layers, nf_hidden_features):
        return (nf_in_features+1) * nf_hidden_features+ nf_num_hidden_layers*(nf_hidden_features+1)*nf_hidden_features + (nf_hidden_features+1)*out_features

    def disable_gradient(self):
        for param in self.parameters():
            param.requires_grad = False
            
# siren auto decoder: modified FILM
class SIRENAutodecoder_mdf_film(nn.Module):
    '''
    siren network with auto decoding 
    mdf mean the conditioning is full projection 
    '''
    def __init__(self, in_coord_features, in_latent_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', weight_init=None,bias_init=None,
                 premap_mode = None, **kwargs):
        super().__init__()
        self.premap_mode = premap_mode
        if not self.premap_mode ==None: 
            self.premap_layer = FeatureMapping(in_coord_features,mode = premap_mode, **kwargs)
            in_coord_features = self.premap_layer.dim # update the nf in features     

        self.first_layer_init = None
                        
        self.nl, nl_weight_init, first_layer_init = NLS_AND_INITS[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init # those are default init funcs 

        # create the net for the nf 
        self.nf_net = nn.ModuleList([BatchLinear(in_coord_features,hidden_features)] + 
                                  [BatchLinear(hidden_features,hidden_features) for i in range(num_hidden_layers)] + 
                                  [BatchLinear(hidden_features,out_features)])

        # create the net for the weights and bias, the hypernet it self has no bias. 
        self.hw_net = nn.ModuleList([BatchLinear(in_latent_features,in_coord_features*hidden_features,bias = False)]+
                                    [BatchLinear(in_latent_features,hidden_features*hidden_features,bias = False) for i in range(num_hidden_layers)])
                                    # [BatchLinear(in_latent_features,hidden_features*out_features,bias = False)])
        self.hb_net = nn.ModuleList([BatchLinear(in_latent_features,hidden_features,bias = False) for i in range(num_hidden_layers+1)])
                                    #  [BatchLinear(in_latent_features,out_features*out_features,bias = False)])

        if self.weight_init is not None:
            self.nf_net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.nf_net[0].apply(first_layer_init)
        self.init_hyper_layer()     

    def init_hyper_layer(self):
        # init weights
        self.hw_net.apply(init_weights_uniform_siren_scale)
        self.hb_net.apply(init_weights_uniform_siren_scale)

    def forward(self, coords, latents):
        # _, h_size, w_size, coord_size = coords.shape
        t_size = latents.shape[0]
        # coords: <t,h,w,c_coord> or <1, h,w,c_coords>
        # latents: <t,h,w,c_latent> or <t, 1,1, coords>
        
        # premap 
        if not self.premap_mode ==None: 
            x = self.premap_layer(coords)
        else: 
            x = coords

        # pass it through  the nf network 
        for i in range(len(self.nf_net) -1):

            x = (
                    self.nf_net[i](x) + 
                    torch.einsum(
                        'thwi,thwji->thwj', 
                        x, 
                        self.hw_net[i](latents).reshape((t_size, 1, 1)+self.nf_net[i].weight.shape)
                    )
                    + self.hb_net[i](latents)
                )
            
            x = self.nl(x)

        x = self.nf_net[-1](x)

        return x 
    

class SIRENAutodecoder_mdf_film_extra_in(SIRENAutodecoder_mdf_film):
    def forward(self, coord, latents):
        coord = torch.concat(
            [torch.ones_like(coord[0][...,:1])*coord[1], coord[0]], dim = -1
        )
        return super().forward(coord, latents)

        
