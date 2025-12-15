import torch
from torch import nn
import numpy as np
from collections import OrderedDict
import math
import cnf.initialization as init


########################
# define all activation functions that are not in pytorch library. 
########################
class Swish(nn.Module):
	def __init__(self):
		super().__init__()
		self.Sigmoid = nn.Sigmoid()
	def forward(self,x):
		return x*self.Sigmoid(x)

class Sine(nn.Module):
    def __init__(self, w0 = init.DEFAULT_W0):
        self.w0 = w0
        super().__init__()

    def forward(self, input):
        return torch.sin(self.w0 * input)

class Sine_tw(nn.Module):
    def __init__(self, w0 = init.DEFAULT_W0):
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor([w0], dtype = torch.float32))

    def forward(self, input):
        return torch.sin(self.w0 * input)


# Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
# special first-layer initialization scheme
# different layers has different initialization schemes:
NLS_AND_INITS = {
    # act name: (init func, first layer init func )
    'sine':(Sine(), init.sine_init, init.first_layer_sine_init), 
    'relu':(nn.ReLU(inplace=True), init.init_weights_normal, None),
    'sigmoid':(nn.Sigmoid(), init.init_weights_xavier, None),
    'tanh':(nn.Tanh(), init.init_weights_xavier, None),
    'selu':(nn.SELU(inplace=True), init.init_weights_selu, None),
    'softplus':(nn.Softplus(), init.init_weights_normal, None),
    'elu':(nn.ELU(inplace=True), init.init_weights_elu, None),
    'swish':(Swish(), init.init_weights_xavier, None),
}

########################
# denfine all the basic layers 
########################
from einops.layers.torch import Rearrange
class BatchLinear(nn.Linear):
    '''
    This is a linear transformation implemented manually. It also allows maually input parameters. 
    for initialization, (in_features, out_features) needs to be provided. 
    weight is of shape (out_features*in_features)
    bias is of shape (out_features)
    
    '''
    __doc__ = nn.Linear.__doc__
    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']
        #print('before multiply',input.shape)
        output = torch.matmul(input,weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2) )
        #print('after multiply',output.shape)
        if not bias == None: 
            output += bias.unsqueeze(-2)
        # print('after bias',output.shape)
        return output


class  BatchLinears(nn.Module):
       def __init__(self, in_features, hidden_features,):
            super().__init__()
            self.linear1 = BatchLinear(in_features, hidden_features,bias=False)
            self.bn = nn.Sequential(
                                    Rearrange('b n d -> b d n'),
                                    nn.BatchNorm1d(hidden_features),
                                    Rearrange('b d n -> b n d'),
                                    )

            #nn.BatchNorm1d(hidden_features)
            self.linear2 = BatchLinear(hidden_features, hidden_features,bias =False)
            self.gelu = nn.GELU()
       def forward(self, x):
           x1 = self.linear1(x)
           x2 = self.bn(x1)
           x2 = self.gelu(x2)
           x2 = self.linear2(x2) + x1

           return x2
 
class MLP_base(nn.Module):
    '''
    MLP with different activation functions 
    '''
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        num_hidden_layers: int, 
        hidden_features: int,
        *,
        outermost_linear: bool = True, 
        nonlinearity: str = 'relu', 
        weight_init = None, 
        output_mode: str = 'single',
        premap_mode: str = None, 
        **kwargs
    ):
        super().__init__()

        self.premap_mode = premap_mode

        if not self.premap_mode == None: 
            self.premap_layer = FeatureMapping(in_features,mode = premap_mode, **kwargs)
            in_features = self.premap_layer.dim # update the nf in features 

        self.first_layer_init = None
        self.output_mode = output_mode                       

        self.nl, nl_weight_init, self.first_layer_init = NLS_AND_INITS[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init # those are default init funcs 

        self.net = []

        self.__customized_init__(
            in_features, 
            out_features, 
            num_hidden_layers, 
            hidden_features,
            outermost_linear, 
            nonlinearity, 
            weight_init, 
            output_mode,
            premap_mode, 
            **kwargs
        )

    def __customized_init__(self):
        raise NotImplementedError
    
    def premap(self, x):
        # propagate through the nf net, implementing SIREN
        if not self.premap_mode ==None: 
            x = self.premap_layer(x)

        return x


class MLP(MLP_base):

    def __customized_init__(
        self, 
        in_features: int, 
        out_features: int, 
        num_hidden_layers: int, 
        hidden_features: int,
        outermost_linear: bool = True, 
        **kwargs
    ):
        # append the first layer
        self.net.append(nn.Sequential(
            BatchLinear(in_features, hidden_features), self.nl
        ))
        
        # append the hidden layers
        for _ in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, hidden_features), self.nl
            ))

        # append the last layer
        if outermost_linear:
            self.net.append(nn.Sequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, out_features), self.nl
            ))

        # put them as a meta sequence
        self.net = nn.Sequential(*self.net)

        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if self.first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(self.first_layer_init)


    def forward(self, x):
        x = super().premap(x)

        if self.output_mode == 'single':
            output = self.net(x)
            return output

        elif self.output_mode=='double':
            x = x.clone().detach().requires_grad_(True)
            output = self.net(x)
            return output, x


class MLP_rezblk(MLP_base):
    '''
    MLP with different activation functions 
    '''
    def __customized_init__(
        self, 
        num_hidden_layers, 
        hidden_features,        
        **kwargs
    ):
        
        
        # append the hidden layers
        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, hidden_features), self.nl
            ))

        # put them as a meta sequence
        self.net = nn.Sequential(*self.net)

        if self.weight_init is not None:
            self.net.apply(self.weight_init)

    def forward(self, x):
        # propagate through the nf net, implementing SIREN
        x = super().premap(x)

        output = 0.5*self.net(x) + 0.5*x
        return output


class MLP_reznet(MLP_base):
    '''
    MLP with different activation functions with rez
    '''
    def __customized_init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        num_hidden_layers_rez: int,
        num_hidden_blocks: int,
        **kwargs
    ):

        # append the first layer
        self.fc1 = BatchLinear(in_features, hidden_features)

        # append the hidden layers
        self.rezblks  = [] 
        for _ in range(num_hidden_blocks):
            self.rezblks.append(
                MLP_rezblk(num_hidden_layers_rez, hidden_features)
            )

        # append the last layer
        self.fc2 = BatchLinear(hidden_features, out_features)

        if self.weight_init is not None:
            self.fc2.apply(self.weight_init)

        if self.first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.fc1.apply(self.first_layer_init)

    def forward(self, x):
        x = super().premap(x)

        output = self.nl(self.fc1(x))
        for blk in self.rezblks:
            output = blk(output)
        output = self.fc2(output)
        return output


class FeatureMapping():
    '''
    This is feature mapping class for  fourier feature networks 
    '''
    def __init__(self, in_features, mode = 'basic', 
                 gaussian_mapping_size = 256, gaussian_rand_key = 0, gaussian_tau = 1.,
                 pe_num_freqs = 4, pe_scale = 2, pe_init_scale = 1,pe_use_nyquist=True, pe_lowest_dim = None, 
                 rbf_out_features = None, rbf_range = 1., rbf_std=0.5):
        '''
        inputs:
            in_freatures: number of input features
            mapping_size: output features for Gaussian mapping
            rand_key: random key for Gaussian mapping
            tau: standard deviation for Gaussian mapping
            num_freqs: number of frequencies for P.E.
            scale = 2: base scale of frequencies for P.E.
            init_scale: initial scale for P.E.
            use_nyquist: use nyquist to calculate num_freqs or not. 
        
        '''
        self.mode = mode
        if mode == 'basic':
            self.B = np.eye(in_features)
        elif mode == 'gaussian':
            rng = np.random.default_rng(gaussian_rand_key)
            self.B = rng.normal(loc = 0., scale = gaussian_tau, size = (gaussian_mapping_size, in_features))
        elif mode == 'positional':
            if pe_use_nyquist == 'True' and pe_lowest_dim:  
                pe_num_freqs = self.get_num_frequencies_nyquist(pe_lowest_dim)
            self.B = pe_init_scale * np.vstack([(pe_scale**i)* np.eye(in_features) for i in range(pe_num_freqs)])
            self.dim = self.B.shape[0]*2
        elif mode == 'rbf':
            self.centers = nn.Parameter(torch.empty((rbf_out_features, in_features), dtype = torch.float32))
            self.sigmas = nn.Parameter(torch.empty(rbf_out_features, dtype = torch.float32))
            nn.init.uniform_(self.centers, -1*rbf_range, rbf_range)
            nn.init.constant_(self.sigmas, rbf_std)

    def __call__(self, input):
        if self.mode in ['basic', 'gaussian', 'positional']: 
            return self.fourier_mapping(input,self.B)
        elif self.mode =='rbf':
            return self.rbf_mapping(input)
            
    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    # Fourier feature mapping
    @staticmethod
    def fourier_mapping(x, B):
        '''
        x is the input, B is the reference information 
        '''
        if B is None:
            return x
        else:
            B = torch.tensor(B, dtype = torch.float32, device = x.device)
            x_proj = (2.*np.pi*x) @ B.T
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        
    # rbf mapping
    def rbf_mapping(self, x):

        size = (x.shape[:-1])+ self.centers.shape
        x = x.unsqueeze(-2).expand(size)
        # c = self.centres.unsqueeze(0).expand(size)
        # distances = (x - self.centers).pow(2).sum(-1) * self.sigmas.unsqueeze(0)
        distances = (x - self.centers).pow(2).sum(-1) * self.sigmas
        return self.gaussian(distances)
    
    @staticmethod
    def gaussian(alpha):
        phi = torch.exp(-1 * alpha.pow(2))
        return phi


# multiplicative 
class MFNBase(nn.Module):
    """
    Multiplicative filter network base class.

    Expects the child class to define the 'filters' attribute, which should be 
    a nn.ModuleList of num_hidden_layers+1 filters with output equal to hidden_features.
    """

    def __init__(self, hidden_features, out_features, num_hidden_layers, weight_scale, bias=True, output_act=False):
        super().__init__()


        self.linear = nn.ModuleList(
            [nn.Linear(hidden_features, hidden_features, bias) for _ in range(num_hidden_layers)]
        )
        self.output_linear = nn.Linear(hidden_features, out_features)
        self.output_act = output_act

        # for lin in self.linear:
        #     lin.weight.data.uniform_(
        #         -np.sqrt(weight_scale / hidden_features),
        #         np.sqrt(weight_scale / hidden_features),
        #     )

        # kaimin init
        for lin in self.linear:
            nn.init.kaiming_uniform_(lin.weight, a = math.sqrt(5))

    def forward(self, x):
        if not self.premap_mode ==None: 
            x = self.premap_layer(x)

        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i - 1](out)
        out = self.output_linear(out)

        if self.output_act:
            out = torch.sin(out)

        return out


class FourierLayer(nn.Module):
    """
    Sine filter as used in FourierNet.
    """

    def __init__(self, in_features, out_features, weight_scale):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.data *= weight_scale  # gamma
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x):
        return torch.sin(self.linear(x))


class FourierNet(MFNBase):
    def __init__(
        self,
        in_features,
        out_features,
        num_hidden_layers,
        hidden_features,
        input_scale=256.0,
        weight_scale=1.0,
        bias=True,
        output_act=False,
        premap_mode = None, 
        **kwargs
    ):
        super().__init__(
            hidden_features, out_features, num_hidden_layers, weight_scale, bias, output_act
        )
        self.premap_mode = premap_mode
        if not self.premap_mode ==None: 
            self.premap_layer = FeatureMapping(in_features,mode = premap_mode, **kwargs)
        in_features = self.premap_layer.dim # update the nf in features 

        self.filters = nn.ModuleList(
            [
                FourierLayer(in_features, hidden_features, input_scale / np.sqrt(num_hidden_layers + 1))
                for _ in range(num_hidden_layers + 1)
            ]
        )

class GaborLayer(nn.Module):
    """
    Gabor-like filter as used in GaborNet.
    """

    def __init__(self, in_features, out_features, weight_scale, alpha=1.0, beta=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)


    def forward(self, x):
        D = (
            (x ** 2).sum(-1)[..., None]
            + (self.mu ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.linear(x)) * torch.exp(-0.5 * D * self.gamma[None, :])


class GaborNet(MFNBase):
    def __init__(
        self,
        in_features,
        out_features,
        num_hidden_layers,
        hidden_features,
        input_scale=256.0,
        weight_scale=1.0,
        alpha=6.0,
        beta=1.0,
        bias=True,
        output_act=False,
        premap_mode = None, 
        **kwargs
    ):
        super().__init__(
            hidden_features, out_features, num_hidden_layers, weight_scale, bias, output_act
        )

        self.premap_mode = premap_mode
        if not self.premap_mode ==None: 
            self.premap_layer = FeatureMapping(in_features,mode = premap_mode, **kwargs)
        in_features = self.premap_layer.dim # update the nf in features 

        self.filters = nn.ModuleList(
            [
                GaborLayer(
                    in_features,
                    hidden_features,
                    input_scale / np.sqrt(num_hidden_layers + 1),
                    alpha / (num_hidden_layers + 1),
                    beta,
                )
                for _ in range(num_hidden_layers + 1)
            ]
        )
