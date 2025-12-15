"""
This module is developed based on the work:
"Conditional neural field latent diffusion model for generating spatiotemporal turbulence"
(Nature Communications 15, 10416, 2024).

Original Repository: https://github.com/jx-wang-s-group/CoNFiLD
Original Authors: P. Du, M. H. Parikh, X. Fan, X.-Y. Liu, J.-X. Wang

Acknowledgement:
We extend our special thanks to the original authors for their generous sharing.
If you intend to use this specific module, please prioritize referring to the original version of the code and the corresponding literature.
"""

from os.path import exists
from os import mkdir
import sys
from basicutility import ReadInput as ri

from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from einops import rearrange

from cnf.utils.normalize import Normalizer_ts
from cnf.utils import readdata
from cnf import nf_networks
from functools import partial


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt
import seaborn as sns


def plot_distributions(target, prediction, num_samples=500, filename="distribution_comparison.jpg"):
    N = target.shape[0]
    indices = np.random.choice(N, num_samples, replace=False)

    sampled_target = target[indices]
    sampled_prediction = prediction[indices]

    flattened_target = sampled_target.cpu().numpy().flatten()
    flattened_prediction = sampled_prediction.cpu().numpy().flatten()
    xmin = min(flattened_target.min(), flattened_prediction.min())
    xmax = max(flattened_target.max(), flattened_prediction.max())

    plt.figure(figsize=(10, 6))

    sns.histplot(flattened_target, bins=50, kde=True, color='blue', label='Target', alpha=0.5,  binrange=(xmin, xmax))
    sns.histplot(flattened_prediction, bins=50, kde=True, color='red', label='Prediction', alpha=0.5, binrange=(xmin, xmax))

    plt.title('Distribution of Target and Prediction')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(filename)
    plt.close()


class basic_set(Dataset):
    def __init__(self, fois, coord, extra_siren_in = None) -> None:
        super().__init__()
        self.fois = fois
        self.total_samples = fois.shape[0]
        
        self.coords = coord

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if hasattr(self, "extra_in"):
            extra_id = idx % self.fois.shape[1]
            idb = idx // self.fois.shape[1]
            return (self.coords, self.extra_in[extra_id]), self.fois[idb, extra_id], idx
        else:
            return self.coords, self.fois[idx], idx


class LatentContainer(torch.nn.Module):
    """
    a model container that stores latents for multi GPU
    """

    def __init__(
        self,
        N_samples,
        N_features,
        dims,
        lumped=False
    ):
        super().__init__()
        self.expand_dims = " ".join(["1" for _ in range(dims)]) if not lumped else "1"
        self.expand_dims = f"N f -> N {self.expand_dims} f"
        self.latents = torch.nn.Parameter(
            torch.zeros((N_samples, N_features), dtype=torch.float32) + 0.1
        )

    def forward(self, batch_ids):
        return rearrange(self.latents[batch_ids], self.expand_dims)


def rMAE(prediction, target, dims=(1, 2)):
    return torch.abs(prediction - target).mean(dim=dims) / torch.abs(target).mean(
        dim=dims
    )


class trainer(object):

    def __init__(self, hyper_para: ri.basic_input, infer_mode=False, infer_dps=False) -> None:
        '''
        Initialize the training module for the Conditional Neural Field model.
        For propogating gradient through the model (e.g. in DPS), set self.nf.eval() 
        Args:
            hyper_para (basic_input): The hyperparameters for the model.
            infer_mode (bool, optional): Flag indicating whether the module is in inference mode. Defaults to False.
                
        '''
        self.world_size = hyper_para.multiGPU

        if not infer_mode:
            ###### read data - fois ######
            if hasattr(hyper_para, "load_data_fn"):
                if type(hyper_para.load_data_fn) == str:
                    load_data_fn = getattr(readdata, hyper_para.load_data_fn)
                    load_params = {}
                elif type(hyper_para.load_data_fn) == dict:
                    load_data_fn = getattr(readdata, hyper_para.load_data_fn["name"])
                    load_params = hyper_para.load_data_fn["kwargs"]
                fois = load_data_fn(hyper_para.data_path, **load_params)

            else:
                fois = np.load(f"{hyper_para.data_path}")

            assert (
                rearrange(fois, f"{hyper_para.readin_data_shape} -> {hyper_para.readin_data_shape}")
                == fois
            ).all(), f"data shape is {tuple(fois.shape)}, which is inconsistant \
            with the fois_shape ({hyper_para.readin_data_shape}) specified in yaml file."

            fois = rearrange(
                fois, f"{hyper_para.readin_data_shape} -> {hyper_para.batch_shape}"
            )

            if "kwargs" in hyper_para.NF:
                assert (
                    hyper_para.NF["kwargs"]["out_features"] == fois.shape[-1]
                ), "NF_out_features is not consistent with fois shape"
            else:
                assert (
                    hyper_para.NF["out_features"] == fois.shape[-1]
                ), "NF_out_features is not consistent with fois shape"
            
            if hasattr(hyper_para, "extra_siren_in"):
                assert isinstance(hyper_para.extra_siren_in, int) or len(hyper_para.extra_siren_in) >= 3
                if isinstance(hyper_para.extra_siren_in, int):
                    extra_siren_in = torch.linspace(0,1,hyper_para.extra_siren_in)
                elif len(hyper_para.extra_siren_in) == 3:
                    extra_siren_in = torch.linspace(*hyper_para.extra_siren_in)
                elif len(hyper_para.extra_siren_in) > 3:
                    extra_siren_in = torch.tensor(hyper_para.extra_siren_in)
                extra_siren_in_flag = True
            else:
                extra_siren_in_flag = False

            self.spatio_shape = fois.shape[1:-1]
            self.spatio_axis = list(
                range(fois.ndim if isinstance(fois, np.ndarray) else fois.dim())
            )[1:-1]
            if extra_siren_in_flag: 
                self.spatio_shape = self.spatio_shape[1:]
                self.spatio_axis = self.spatio_axis[:-1]
                
            ###### read data - coordinate ######
            if hasattr(hyper_para, "coor_path"):
                coord = np.load(f"{hyper_para.coor_path}")
                #print(coord.shape)
                #aa
                assert (
                    coord.shape[:-1] == self.spatio_shape
                ), "coordinate shape is not consistent with fois shape"
                assert (
                    coord.shape[-1] == hyper_para.dims
                ), "coordinate dimension is not consistent with dims in yaml file"
            else:
                Warning(
                    "No coordinate data is provided, using cartisian coordinate inferred from fois data"
                )
                coord = [np.linspace(0, 1, i) for i in self.spatio_shape]
                coord = np.stack(np.meshgrid(*coord, indexing="ij"), axis=-1)
                
            self.train_coord = torch.tensor(coord, dtype=torch.float32)
            ###### convert to tensor ######
            fois = (
                torch.tensor(fois, dtype=torch.float32)
                if not isinstance(fois, torch.Tensor)
                else fois
            )
            self.N_samples = fois.shape[0] if not extra_siren_in_flag else fois.shape[0] * fois.shape[1]
            print(f"total training samples: {self.N_samples}")
            coord = (
                torch.tensor(coord, dtype=torch.float32)
                if not isinstance(coord, torch.Tensor)
                else coord
            )

        ###### normalizer ######
        self.in_normalizer = Normalizer_ts(**hyper_para.normalizer)
        self.out_normalizer = Normalizer_ts(**hyper_para.normalizer)
        if hasattr(hyper_para, "extra_siren_in"):
            self.extra_in_normalizer = Normalizer_ts(**hyper_para.normalizer)
            extra_siren_in_flag = True
        else:
            extra_siren_in_flag = False
            
        if not exists(f"{hyper_para.save_path}") and (not infer_mode):
            mkdir(hyper_para.save_path)
            
        if exists(f"{hyper_para.save_path}/normalizer_params.pt"):
            print(
                f"loading normalizer parameters from {hyper_para.save_path}/normalizer_params.pt"
            )
            norm_params = torch.load(f"{hyper_para.save_path}/normalizer_params.pt")
            self.in_normalizer.params = norm_params["x_normalizer_params"]
            self.out_normalizer.params = norm_params["y_normalizer_params"]
            if extra_siren_in_flag:
                self.extra_in_normalizer.params = norm_params["extra_normalizer_params"]

        elif not infer_mode:
            print("No noramlization file found! Calculating normalizer parameters")
            self.in_normalizer.fit_normalize(
                coord if hyper_para.lumped_latent else coord.flatten(0, hyper_para.dims-1) 
            )
            if extra_siren_in_flag:
                self.out_normalizer.fit_normalize(fois.flatten(0, hyper_para.dims+1))
            else:
                self.out_normalizer.fit_normalize(
                    fois if hyper_para.lumped_latent else fois.flatten(0, hyper_para.dims)
                )
            if extra_siren_in_flag:
                self.extra_in_normalizer.fit_normalize(extra_siren_in.flatten())
            print(
                f"Saving normalizer parameters to {hyper_para.save_path}/normalizer_params.pt"
            )
            toSave = {
                "x_normalizer_params": self.in_normalizer.get_params(),
                "y_normalizer_params": self.out_normalizer.get_params(),
            }
            if extra_siren_in_flag:
                toSave["extra_normalizer_params"] = self.extra_in_normalizer.get_params()
            torch.save(toSave, hyper_para.save_path + "/normalizer_params.pt",)
        else:
            raise FileNotFoundError(
                f"{hyper_para.save_path}/normalizer_params.pt does not exist"
            )

        if not infer_mode:
            normed_coords = self.in_normalizer.normalize(coord)     ###note
            normed_fois =  self.out_normalizer.normalize(fois)
            if extra_siren_in_flag:
                normed_extra_siren_in = self.extra_in_normalizer.normalize(extra_siren_in)
            else:
                normed_extra_siren_in = None


        ###### nf ######
        if "kwargs" not in hyper_para.NF:
            self.nf: torch.nn.Module = getattr(nf_networks, hyper_para.NF["name"])(
                in_coord_features=hyper_para.dims if not extra_siren_in_flag else hyper_para.dims+1,
                in_latent_features=hyper_para.hidden_size,
                out_features=hyper_para.NF["out_features"],
                num_hidden_layers=hyper_para.NF["num_hidden_layers"],
                hidden_features=hyper_para.NF["hidden_features"],
            )
        else:
            self.nf: torch.nn.Module = getattr(nf_networks, hyper_para.NF["name"])(
                hyper_para.NF["kwargs"]
            )

        ###### latents ######
        if not infer_mode:
            self.latents = LatentContainer(
                self.N_samples, hyper_para.hidden_size, hyper_para.dims, hyper_para.lumped_latent
            )

        self.hyper_para = hyper_para

        if not infer_mode:
            if hasattr(hyper_para, "dataset"):
                raise NotImplementedError

            else:
                self.dataset = basic_set(normed_fois, normed_coords, normed_extra_siren_in)

            self.test_criteria = partial(
                getattr(sys.modules[__name__], hyper_para.test_criteria),
                dims=self.spatio_axis,
            )

        if infer_mode and not infer_dps:
            self.infer = torch.no_grad(self.infer)

    def infer(
        self,
        coord: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        if coord is None:
            print("Using default training query points")
        coord = coord if coord is not None else self.train_coord
        coord = self.in_normalizer.normalize(coord)
        if len(coord.shape) > 2:
            latents = latents[:, None, None]
        else:
            latents = latents[:, None]
        out = self.nf(coord.to(latents.device), latents)
        return self.out_normalizer.denormalize(out)

    def train(self, fix_nf = False):
        self.epoches = self.hyper_para.epochs
        self.criterion = getattr(torch.nn, self.hyper_para.loss_fn)()
        self.lr = self.hyper_para.lr
        self.save_dict = {
            "save_path": self.hyper_para.save_path,
            "save_every": self.hyper_para.save_every,
        }
        optim_dict = self.optim_dict if hasattr(self, "optim_dict") else {}
        start_epoch = self.start_epoch if hasattr(self, "start_epoch") else 0

        if self.world_size > 1:
            print(f"Data Parallel training, using {self.world_size} GPUs")

            p = mp.spawn(
                self._single_trainer,
                args=(
                    self.nf,
                    self.latents,
                    self.criterion,
                    self.dataset,
                    self.hyper_para,
                    self.save,
                    self.world_size,
                    optim_dict,
                    start_epoch,
                    self.out_normalizer,
                    self.test_criteria,
                    fix_nf,
                ),
                nprocs=self.world_size,
                join=False,
            )
            p.join()

        else:
            print(f"Single GPU training")
            self._single_trainer(
                0,
                self.nf,
                self.latents,
                self.criterion,
                self.dataset,
                self.hyper_para,
                self.save,
                self.world_size,
                optim_dict,
                start_epoch,
                self.out_normalizer,
                self.test_criteria,
                fix_nf
            )

    @staticmethod
    def _single_trainer(
        rank,
        model,
        latents,
        criterion,
        dataset,
        hyper_para,
        savefn,
        world_size=1,
        optim_dict={},
        start_epoch=0,
        out_normalizer=None,
        test_criteria=None,
        fix_nf = False,
    ):
        # print(model)

        model.to(rank)
        latents.to(rank)
        if world_size > 1:

            dist.init_process_group("nccl", rank=rank, world_size=world_size)

            model = DDP(model, device_ids=[rank])
            latents = DDP(latents, device_ids=[rank])

            train_loader = DataLoader(
                dataset,
                batch_size=hyper_para.batch_size,
                shuffle=False,
                sampler=DistributedSampler(dataset),
            )
            test_loader = DataLoader(
                dataset,
                batch_size=hyper_para.test_batch_size,
                shuffle=False,
                sampler=DistributedSampler(dataset),
            )
        else:
            train_loader = DataLoader(
                dataset, batch_size=hyper_para.batch_size, shuffle=False
            )
            test_loader = DataLoader(
                dataset, batch_size=hyper_para.test_batch_size, shuffle=False
            )


        if rank == 0:
            logger = SummaryWriter(hyper_para.save_path)

        optim_net_dec = torch.optim.Adam(model.parameters(), lr=hyper_para.lr["nf"])
        
        if fix_nf: model.eval()
        #criterion = torch.nn.L1Loss() 
        for i in tqdm(range(start_epoch, start_epoch + hyper_para.epochs)):
            if world_size > 1:
                train_loader.sampler.set_epoch(i)

            train_ins_error = []
            latents = []

            for batch_coords, batch_fois, idx in train_loader:

                batch_fois = batch_fois.to(rank)
                if isinstance(batch_coords, list): 
                    batch_coords = [i.to(rank) for i in batch_coords]
                else:
                    batch_coords = batch_coords.to(rank)
                batch_output = model(batch_coords, batch_fois)                
                latent = model.gen_latent(batch_coords, batch_fois)
                
                latents.append(latent)

                loss = criterion(batch_output, batch_fois)
                optim_net_dec.zero_grad()
                loss.backward()
                optim_net_dec.step()
                train_ins_error.append(loss)
            #print(batch_output[0], batch_fois[0])
            epoch_loss = torch.stack(train_ins_error).mean()  # .to(rank)
            if world_size > 1:
                torch.distributed.reduce(
                    epoch_loss, op=torch.distributed.ReduceOp.AVG, dst=0
                )
            if rank == 0:
                mean_loss = epoch_loss.item()
                tqdm.write(str(mean_loss))
                logger.add_scalar("loss", mean_loss, i)

            if i % hyper_para.save_every == 0 or i == start_epoch + hyper_para.epochs-1:  # and i > 0:

                test_error = []
                targets = []
                predictions = []
                model.eval()
                with torch.no_grad():
                    if world_size > 1:
                        test_loader.sampler.set_epoch(i)
                    for test_coords, test_fois, idx in test_loader:
                        if isinstance(test_coords, list): 
                            test_coords = [i.to(rank) for i in test_coords]
                        else:
                            test_coords = test_coords.to(rank)
                        test_fois = test_fois.to(rank)
                        prediction = out_normalizer.denormalize(
                            model(test_coords, test_fois)
                        )
                        target = out_normalizer.denormalize(test_fois.to(rank))
                        #prediction = model(test_coords, test_fois)
                        #target = test_fois.to(rank)
                        
                        error = test_criteria(prediction, target)
                        test_error.append(error.detach())
                        targets.append(target)
                        predictions.append(prediction)

                test_error = torch.cat(test_error).mean(dim=0)
                print(test_error)
                predictions = torch.cat(predictions, dim=0).squeeze(1)
                targets = torch.cat(targets,dim=0).squeeze(1)

                #print(target[0], prediction[0]) 
                p_speed = (predictions[:,:,0]**2 + predictions[:,:,1]**2 ) **0.5 
                t_speed = (targets[:,:,0]**2 + targets[:,:,1]**2 ) **0.5
                
                p_pressure = predictions[:,:,2]
                t_pressure = targets[:,:,2]

                plot_distributions(p_speed, t_speed,500,"speed_distribution.jpg")
                plot_distributions(p_pressure, t_pressure, 500,"pressure_distribution.jpg")

                print(prediction.shape)
                
                latents = torch.cat(latents, dim=0).squeeze(1)
                print(latents.shape)
                torch.save(latents, "../latent/train_latents.pt")


                if world_size > 1:
                    torch.distributed.reduce(
                        test_error, op=torch.distributed.ReduceOp.AVG, dst=0
                    )
                if not fix_nf: model.train()
                #print(latents)
                if rank == 0:
                    for k in range(test_error.shape[-1]):
                        logger.add_scalar(f"test_error_{k}", test_error[k], i)

                    if world_size > 1:
                        savefn(
                            model.module, latents.module, i, optim_net_dec
                        )
                    else:
                        savefn(model, latents, i, optim_net_dec)

    def save(self, model, latents, epoch, optim_nf,):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_net_dec_dict": optim_nf.state_dict(),
                "epoch": epoch,
            },
            f"{self.hyper_para.save_path}/checkpoint_{epoch}.pt",
        )

    def load(self, checkpoint_id: int, siren_only=False):

        if checkpoint_id == -1:
            import glob

            checkpoint_list = glob.glob(f"{self.hyper_para.save_path}/checkpoint_*.pt")
            checkpoint_list = [
                int(i.split("_")[-1].split(".")[0]) for i in checkpoint_list
            ]
            try:
                checkpoint_id = max(checkpoint_list)
            except ValueError:
                print(
                    f"*No checkpoint found in {self.hyper_para.save_path}*, starting from scratch"
                )
                return 

        print(
            f"loading checkpoint from {self.hyper_para.save_path}/checkpoint_{checkpoint_id}.pt"
        )

        checkpoint = torch.load(
            f"{self.hyper_para.save_path}/checkpoint_{checkpoint_id}.pt"
        )
        self.nf.load_state_dict(checkpoint["model_state_dict"])

        self.start_epoch = checkpoint["epoch"]

        if not siren_only:

            self.optim_dict = {
                k: checkpoint[k] for k in ["optim_net_dec_dict",]
            }
            
            return self.nf, self.latents, self.optim_dict, checkpoint["epoch"]

        else:
            self.optim_dict = {}


if __name__ == "__main__":
    import sys
    import os

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    hp = ri.basic_input(sys.argv[1])
    mytrainer = trainer(hp)
    mytrainer.load(-1)
    mytrainer.train()
