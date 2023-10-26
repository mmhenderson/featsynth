import numpy as np

from typing import List, Tuple

import torch

import utilities

# this code is originally from:
# https://github.com/honzukka/texture-synthesis-pytorch
# modified by MMH in 2022 

def gram_matrix_spatweighted(activations: torch.Tensor, spatial_weights: torch.Tensor = None, do_sqrt=True) -> torch.Tensor:
    b, n, x, y = activations.size()
    # print((b,n,x,y))
    activation_matrix = activations.view(b * n, x * y)

    if do_sqrt:
        # print('doing sqrt')
        activation_matrix_weighted = activation_matrix * torch.sqrt(spatial_weights[None,:])
    else:
        # print('skipping sqrt')
        activation_matrix_weighted = activation_matrix * spatial_weights[None,:]
    
    G = torch.mm(activation_matrix_weighted, activation_matrix_weighted.t())    # gram product
    return G.div(b * n * x * y)     # normalization


class Model:
    def __init__(
        self, path: str, device: torch.device, target_images: List[torch.Tensor],
        layer_weights: List[float] = [1e09, 1e09, 1e09, 1e09, 1e09],
        important_layers: List[str] = [
            'relu1_1', 'pool1', 'pool2', 'pool3', 'pool4'
        ],
        spatial_weights_list: List[torch.Tensor] = None,
        do_sqrt=True
    ):
        self.net = utilities.load_model(path).to(device).eval()
        self.device = device
        self.target_images = [t.to(device) for t in target_images]
        self.layer_weights = layer_weights
        self.important_layers = important_layers
        self.do_sqrt=do_sqrt
        
        
        if spatial_weights_list is None:
            self.spatial_weights_list = [None for l in self.important_layers]
            self.n_total_grid = 1
        else:
            self.spatial_weights_list = spatial_weights_list
            self.spatial_weights_list = [torch.Tensor(sw).to(self.device) for sw in self.spatial_weights_list]
            self.n_total_grid = self.spatial_weights_list[0].shape[0]
            assert(len(self.spatial_weights_list)==len(self.important_layers))
           
            
        # extract Gram matrices of the target image
        
        gram_hook = GramHook(self.spatial_weights_list, do_sqrt = self.do_sqrt)
            
        gram_hook_handles = []
        
        for name, layer in self.net.named_children():
            if name in self.important_layers:
               
                handle = layer.register_forward_hook(gram_hook)
                gram_hook_handles.append(handle)
            
        for t in self.target_images:
            self.net(t)
        
        target_gram_matrices = gram_hook.gram_matrices
        print(len(target_gram_matrices), len(target_gram_matrices[0]))
        
        target_matrix_avg = self.__combine_target_gram_matrices(target_gram_matrices)
        
        # register Gram loss hook
        self.gram_loss_hook = GramLossHook(
            target_matrix_avg, 
            self.layer_weights, \
            self.important_layers, self.spatial_weights_list, do_sqrt = self.do_sqrt
        )
        
        for handle in gram_hook_handles:    # Gram hook is not needed anymore
            handle.remove()

        for name, layer in self.net.named_children():
            if name in self.important_layers:
                # print('adding loss hook for %s'%name)
                handle = layer.register_forward_hook(self.gram_loss_hook)

        # remove unnecessary layers
        i = 0
        for name, layer in self.net.named_children():
            if name == important_layers[-1]:
                break
            i += 1
        self.net = self.net[:(i + 1)]
        # print([name for [name, l] in self.net.named_children()])
        
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        self.gram_loss_hook.clear()

        return self.net(image)

    def get_loss(self) -> torch.Tensor:
        # return sum(self.gram_loss_hook.losses)
        return torch.stack(self.gram_loss_hook.losses, dim=0).sum(dim=0)

    
    def __combine_target_gram_matrices(self, target_gram_matrices):
        
        # going to average the gram matrix for each layer across all the target images
        # NOTE i haven't tested this for the spatially-constrained synthesis procedure.
        # if trying it for that, the n_layers dim here should include the different 
        # grid positions too. should work similarily
        n_images = len(target_gram_matrices)
        n_layers = len(target_gram_matrices[0]) 
        
        concat_gram_matrices = [ torch.dstack([target_gram_matrices[ii][ll] \
                                            for ii in range(n_images)]) \
                                for ll in range(n_layers)]
        # each element is [n_feat x n_feat x n_images]
        
        # averaging gram matrix for each layer, across target images
        avg_gram_matrices = [torch.mean(c, dim=2) for c in concat_gram_matrices]
        
        return avg_gram_matrices
        
class GramHook:
    def __init__(self, 
                 spatial_weights_list: List[torch.Tensor],
                 do_sqrt=True):
        self.gram_matrices = []
        self.spatial_weights_list = spatial_weights_list
        self.total_counter = -1;
        self.n_layers = len(self.spatial_weights_list)
        # self.layer_counter = -1;
        self.do_sqrt=do_sqrt;
        
    def __call__(
        self, layer: torch.nn.Module, layer_in: Tuple[torch.Tensor],
        layer_out: torch.Tensor
    ):
        self.total_counter+=1
        # figure out what layer and image we are on 
        # image is on outside of loop, layer on inside
        ll = int(np.mod(self.total_counter, self.n_layers))
        
        if ll==0:
            self.gm_this_image = []
           
        if self.spatial_weights_list[ll] is None:
            n_grid_total = 1
        else:
            n_grid_total = self.spatial_weights_list[ll].shape[0]
            
        for gg in range(n_grid_total):

            lout = layer_out.detach()

            if self.spatial_weights_list[ll] is None:
                spatial_weights = torch.ones(size=[lout.shape[2]*lout.shape[3]]).to(lout.device)
            else:
                spatial_weights = self.spatial_weights_list[ll][gg,:]
            
            gram_matrix = gram_matrix_spatweighted(lout, spatial_weights, self.do_sqrt)
           
            self.gm_this_image.append(gram_matrix)
            
        if ll==(self.n_layers-1):
            
            self.gram_matrices.append(self.gm_this_image)
           
        
class GramLossHook:
    def __init__(
        self, target_gram_matrices: List[torch.Tensor],
        layer_weights: List[float], layer_names: List[str],
        spatial_weights_list: List[torch.Tensor],
        do_sqrt=True, 
    ):
        self.target_gram_matrices = target_gram_matrices
        self.layer_weights = [
            weight * (1.0 / 4.0) for weight in layer_weights
        ]
        self.layer_names = layer_names
        self.losses: List[torch.Tensor] = []
        self.spatial_weights_list = spatial_weights_list
        self.layer_counter = -1;
        self.do_sqrt=do_sqrt
        
    def __call__(
        self, layer: torch.nn.Module, layer_in: Tuple[torch.Tensor],
        layer_out: torch.Tensor
    ):
        self.layer_counter+=1
        ll = self.layer_counter
        # print('layer_counter = %d'%self.layer_counter)
        assert ll < len(self.layer_weights)
        assert ll < len(self.target_gram_matrices)
        assert not torch.isnan(layer_out).any()

        if self.spatial_weights_list[ll] is None:
            n_grid_total = 1
        else:
            n_grid_total = self.spatial_weights_list[ll].shape[0]
         
        for gg in range(n_grid_total):
            
            if self.spatial_weights_list[ll] is None:
                spatial_weights = torch.ones(size=[layer_out.shape[2]*layer_out.shape[3]]).to(layer_out.device)
            else:
                spatial_weights = self.spatial_weights_list[ll][gg,:]
                
            tt = ll * n_grid_total + gg
            # print(ll, gg, tt)
            opt_gram_matrix = gram_matrix_spatweighted(layer_out, 
                                                       spatial_weights, \
                                                       self.do_sqrt)
            
            # print(opt_gram_matrix.shape)
            # print(self.target_gram_matrices[tt].shape)
            loss = self.layer_weights[ll] * (
                (opt_gram_matrix - self.target_gram_matrices[tt])**2
            ).sum()
            # print(tt, len(self.losses), loss)
            self.losses.append(loss)

    def clear(self):
        self.losses = []
        self.layer_counter=-1
        
        
def rand_symmetric(N):

    r = torch.zeros([N,N])

    i,j = torch.triu_indices(N,N)

    randvals = torch.randn(i.shape)

    r[i,j] = randvals
    r.T[i,j] = randvals
    
    return r