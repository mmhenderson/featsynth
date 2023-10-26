import time
import numpy as np
from argparse import Namespace
from typing import Callable, List

import torch
import torchvision      # type: ignore
import PIL.Image        # type: ignore

import utilities
import model_spatial

# this code is originally from:
# https://github.com/honzukka/texture-synthesis-pytorch
# modified by MMH in 2022 

class Optimizer:
    def __init__(self, model: model_spatial.Model, args: Namespace):
        self.model = model
        self.n_steps = args.n_steps
        self.checkpoint_every = args.checkpoint_every
        self.max_iter = args.max_iter
        self.lr = args.lr
        
        minvals = [float(t.min()) for t in model.target_images]
        minval = np.min(minvals)
        maxvals = [float(t.max()) for t in model.target_images]
        maxval = np.max(maxvals)
        
        # initialize the image to be optimized
        # ----------------------
        # We use a sigmoid function to keep the optimized values
        # within the bounds of the target image via reparametrization.
        # In order to initialize our guess with noise of the same magnitude
        # as in Gatys et al. (2015), we need to apply inverse sigmoid
        # to the noise we want.
        inv_reparam_func = self.get_inv_reparam_func(minval, maxval)
        if args.rndseed is not None:
            torch.manual_seed(args.rndseed)
        desired_noise = torch.randn_like(
            model.target_images[0]
        ).clamp(minval, maxval)
        
        self.opt_image = inv_reparam_func(desired_noise)
        self.opt_image = self.opt_image.contiguous().requires_grad_()
        # ----------------------

        self.reparam_func = self.get_reparam_func(minval, maxval)
        self.last_checkpoint_time = time.time()
        self.to_pil = torchvision.transforms.ToPILImage()

    def optimize(self) -> torch.Tensor:
        optimizer = torch.optim.LBFGS(
            # [self.opt_image.contiguous().requires_grad_()],
            [self.opt_image],
            lr=self.lr, max_iter=self.max_iter, tolerance_grad=0.0,
            history_size=50,
            tolerance_change=0.0, line_search_fn='strong_wolfe'
        )

        step = 0
        self.losses: List[float] = []
        self.opt_images: List[PIL.Image.Image] = []
        while step < self.n_steps:
            def closure():
                optimizer.zero_grad()
                self.model(self.reparam_func(self.opt_image))
                loss = self.model.get_loss()
                loss.backward()

                return loss

            optimizer.step(closure)

            if step % self.checkpoint_every == self.checkpoint_every - 1:
                self.checkpoint(step + 1, self.model.get_loss().item())
            step += 1
        return self.reparam_func(self.opt_image).detach().cpu()

    def checkpoint(self, step: int, loss: float):
        time_delta = time.time() - self.last_checkpoint_time
        print('step: {}, loss: {} ({:.2f}s)'.format(
            step, loss, time_delta
        ))

        self.losses.append(loss)
        self.opt_images.append(
            utilities.postprocess_image_quick(
                self.reparam_func(self.opt_image.clone().detach())
                # self.reparam_func(self.opt_image.clone().detach().cpu())
            )
        )

        self.last_checkpoint_time = time.time()

    @staticmethod
    def get_reparam_func(
        minval, maxval, 
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        minimum = minval
        value_range = maxval - minimum
        return lambda x: (utilities.sigmoid(x) * value_range) + minimum

    @staticmethod
    def get_inv_reparam_func(
        minval, maxval, 
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        minimum = minval
        value_range = maxval - minval

        return lambda y: utilities.inv_sigmoid(
            (1.0 / value_range) * (y - minimum)
        )
