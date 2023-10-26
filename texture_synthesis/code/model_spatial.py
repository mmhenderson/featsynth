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
        self, path: str, device: torch.device, target_image: torch.Tensor,
        layer_weights: List[float] = [1e09, 1e09, 1e09, 1e09, 1e09],
        important_layers: List[str] = [
            'relu1_1', 'pool1', 'pool2', 'pool3', 'pool4'
        ],
        spatial_weights_list: List[torch.Tensor] = None,
        do_sqrt=True, 
    ):
        self.net = utilities.load_model(path).to(device).eval()
        self.device = device
        self.target_image = target_image.to(device)
        self.layer_weights = layer_weights
        self.important_layers = important_layers
        self.do_sqrt=do_sqrt

        print('matching layers:')
        print(self.important_layers)
        
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
            
        self.net(self.target_image)
        
        target_gram_matrices = gram_hook.gram_matrices
        
        # print(target_gram_matrices[0])
        
        # print('gram matrices:')
        # print(len(gram_hook.gram_matrices), [gm.shape for gm in gram_hook.gram_matrices])
        # register Gram loss hook
        self.gram_loss_hook = GramLossHook(
            target_gram_matrices, self.layer_weights, \
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

class GramHook:
    def __init__(self, 
                 spatial_weights_list: List[torch.Tensor],
                 do_sqrt=True):
        self.gram_matrices = []
        self.spatial_weights_list = spatial_weights_list
        self.layer_counter = -1;
        self.do_sqrt=do_sqrt;
        
    def __call__(
        self, layer: torch.nn.Module, layer_in: Tuple[torch.Tensor],
        layer_out: torch.Tensor
    ):
        self.layer_counter+=1
        ll = self.layer_counter
        
        if self.spatial_weights_list[ll] is None:
            n_grid_total = 1
        else:
            n_grid_total = self.spatial_weights_list[ll].shape[0]
            
        for gg in range(n_grid_total):

            lout = layer_out.detach()

            # print(lout.shape)
            # print(lout[0, 0:2,0,0])
            
            if self.spatial_weights_list[ll] is None:
                spatial_weights = torch.ones(size=[lout.shape[2]*lout.shape[3]]).to(lout.device)
            else:
                spatial_weights = self.spatial_weights_list[ll][gg,:]
            # print('size of spatial weights:')
            # print(spatial_weights.shape)
            # print('size of lout:')
            # print(lout.shape)
            
            
            gram_matrix = gram_matrix_spatweighted(lout, spatial_weights, self.do_sqrt)
            # print('size of gram matrix:')
            # print(gram_matrix.shape)
            self.gram_matrices.append(gram_matrix)

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