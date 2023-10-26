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
        do_sqrt=True, 
        add_noise=False, 
        noise_levels=[],
    ):
        self.net = utilities.load_model(path).to(device).eval()
        self.device = device
        self.target_images = [t.to(device) for t in target_images]
        self.layer_weights = layer_weights
        self.important_layers = important_layers
        self.do_sqrt=do_sqrt
        self.add_noise=add_noise
        self.noise_levels=noise_levels
        # print(self.noise_levels)
        # print(self.important_layers)
        
        self.spatial_weights_list = [torch.Tensor(sw).to(self.device) for sw in spatial_weights_list]
        if self.spatial_weights_list is not None:
            self.n_total_grid = self.spatial_weights_list[0].shape[0]
            assert(len(self.spatial_weights_list)==len(self.important_layers))
            
        # extract Gram matrices of the target image
        
        gram_hook = GramHook(self.spatial_weights_list, do_sqrt = self.do_sqrt, \
                            add_noise=self.add_noise, \
                            noise_levels=self.noise_levels)
            
        gram_hook_handles = []
        
        for name, layer in self.net.named_children():
            if name in self.important_layers:
               
                handle = layer.register_forward_hook(gram_hook)
                gram_hook_handles.append(handle)
        
        for t in self.target_images:
            self.net(t)
        
        
        target_gram_matrices = gram_hook.gram_matrices
        # print(len(target_gram_matrices))
        # print(len(target_gram_matrices[0]))
        # should be [images x layers]
        
        target_matrix_avg = self.__combine_target_gram_matrices(target_gram_matrices)
#         if self.add_noise:
#             target_gram_matrices = self.__perturb_target_gram_matrices(target_gram_matrices)

#         print(target_gram_matrices[0])
        
        # print('gram matrices:')
        # print(len(gram_hook.gram_matrices), [gm.shape for gm in gram_hook.gram_matrices])
        # register Gram loss hook
        self.gram_loss_hook = GramLossHook(
            target_matrix_avg, 
            self.layer_weights, \
            self.important_layers, self.spatial_weights_list, do_sqrt = self.do_sqrt
        )
        # print('sizes of gram matrices')
        # for ii, mat in enumerate(gram_hook.gram_matrices):
        #     print(ii, mat.shape)
        for handle in gram_hook_handles:    # Gram hook is not needed anymore
            handle.remove()

        for name, layer in self.net.named_children():
            if name in self.important_layers:
                # print('adding loss hook for %s'%name)
                handle = layer.register_forward_hook(self.gram_loss_hook)

        # print([name for [name, l] in self.net.named_children()])
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
        
        n_images = len(target_gram_matrices)
        n_layers = len(target_gram_matrices[0])
        
        concat_gram_matrices = [ torch.dstack([target_gram_matrices[ii][ll] \
                                            for ii in range(n_images)]) \
                                for ll in range(n_layers)]
        # each element is [n_feat x n_feat x n_images]
        # print([c.shape for c in concat_gram_matrices])
        
        avg_gram_matrices = [torch.mean(c, dim=2) for c in concat_gram_matrices]
        # print([a.shape for a in avg_gram_matrices ])
        
        return avg_gram_matrices
    
    
#     def __perturb_target_gram_matrices(self, target_gram_matrices):
        
#         # adding random gaussian noise to the target gram matrix
#         noise_add = [rand_symmetric(m.shape[0]) * self.noise_levels[0] for m in target_gram_matrices]
#         target_gram_matrices = [m + n for m, n in zip(target_gram_matrices, noise_add)]
        
#         return target_gram_matrices
        
class GramHook:
    def __init__(self, 
                 spatial_weights_list: List[torch.Tensor],
                 do_sqrt=True, \
                 add_noise=False, \
                 noise_levels=[]):
        
        self.gram_matrices = []
        self.spatial_weights_list = spatial_weights_list
        self.total_counter = -1;
        self.n_layers = len(self.spatial_weights_list)
        # print('n_layers=%d'%self.n_layers)
        
        self.do_sqrt=do_sqrt;
        self.add_noise=add_noise;
        self.noise_levels=noise_levels;
        
    def __call__(
        self, layer: torch.nn.Module, layer_in: Tuple[torch.Tensor],
        layer_out: torch.Tensor
    ):
        self.total_counter+=1
        # figure out what layer and image we are on 
        # image is on outside of loop, layer on inside
        ll = int(np.mod(self.total_counter, self.n_layers))
        # ii = int(np.floor(self.total_counter/self.n_layers))
        # print([self.total_counter, ii, ll])
        
        if ll==0:
            # print('making self.gm_this_image')
            self.gm_this_image = []
            
        n_grid_total = self.spatial_weights_list[0].shape[0]
        for gg in range(n_grid_total):
            spatial_weights = self.spatial_weights_list[ll][gg,:]
            # print('size of spatial weights:')
            
            lout = layer_out.detach()
            
            # print(lout.shape)
            # print(lout[0, 0:2,0,0])
            
            gram_matrix = gram_matrix_spatweighted(lout, spatial_weights, self.do_sqrt)
            # print('size of gram matrix:')
            # print(gram_matrix.shape)
            self.gm_this_image.append(gram_matrix)
            
        if ll==(self.n_layers-1):
            # print('appending self.gm_this_image to self.gram_matrices')
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

#         print(self.target_gram_matrices[0][0:2,0:2])
        n_grid_total = self.spatial_weights_list[0].shape[0]
        for gg in range(n_grid_total):
            # print('len of layer_names: %d'%len(self.layer_names))
            
            tt = ll * n_grid_total + gg
            # print(ll, gg, tt)
            opt_gram_matrix = gram_matrix_spatweighted(layer_out, 
                                                       self.spatial_weights_list[ll][gg,:],
                                                       self.do_sqrt)
            
            # print(opt_gram_matrix.shape)
            # print(self.target_gram_matrices[tt].shape)
            # print(opt_gram_matrix.dtype)
            # print(self.target_gram_matrices[tt].dtype)
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