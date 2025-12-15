import torch 
import numpy as np
import pyvista as pv
import matplotlib
import matplotlib.pyplot as plt
from os.path import join as osj
from os.path import exists as ose
from tqdm import tqdm

def train_regular_transformer(model, device, train_loader, optimizer, criterion, scheduler = None , x_normalizer = None,y_normalizer = None):
    
    model.train()
    train_ins_error = []
    for batch_idx, batch in enumerate(train_loader):
        inputs0 = batch['input'][0]
        inputs1 = batch['input'][1]
        labels = batch['output']
        if x_normalizer!= None:
            # print('normalizing x')
            inputs0 = x_normalizer.normalize(inputs0)
            inputs1 = x_normalizer.normalize(inputs1)
        if y_normalizer!= None:
            # print('normalizing y')
            labels = y_normalizer.normalize(labels)
        # print(batch_idx)
        inputs0 = inputs0.to(device)
        inputs1= inputs1.to(device)
        labels = labels.to(device)

        # print('inputbatchshape:', inputs.shape)
        # print('labelsbatchshape:',labels.shape)

        optimizer.zero_grad()
        tgt_mask = model.get_tgt_mask(inputs1.shape[1]).to(device)
        output = model(inputs0,inputs1,tgt_mask=tgt_mask)
        # print('outputbatchshape:',output.shape)
        # print('debug:',output.size(),labels.size())
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()
        train_ins_error.append(loss)
    return torch.mean(torch.stack(train_ins_error))

def train_regular(model, device, train_loader, optimizer, criterion, scheduler = None , x_normalizer = None,y_normalizer = None):
    
    model.train()
    train_ins_error = []
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch['input']
        labels = batch['output']
        if x_normalizer!= None:
            # print('normalizing x')
            inputs = x_normalizer.normalize(inputs)
        if y_normalizer!= None:
            # print('normalizing y')
            labels = y_normalizer.normalize(labels)
        # print(batch_idx)
        inputs = inputs.to(device)
        labels = labels.to(device)
        # print('batch: ',inputs.shape,labels.shape)
        # print('batch: ',torch.max(inputs),torch.max(labels))
        # print('batch: ',torch.min(inputs),torch.min(labels))
        # print('inputbatchshape:', inputs.shape)
        # print('labelsbatchshape:',labels.shape)

        optimizer.zero_grad()
        output = model(inputs)
        # print('outputbatchshape:',output.shape)
        # print('debug:',output.size(),labels.size())
        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()
        train_ins_error.append(loss)
    return torch.mean(torch.stack(train_ins_error))

def train_regular_with_JH(model, device, train_loader, optimizer, criterion, scheduler = None , x_normalizer = None,y_normalizer = None):
    
    model.train()
    train_ins_error = []
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch['input']
        labels = batch['output']
        if x_normalizer!= None:
            # print('normalizing x')
            inputs = x_normalizer.normalize(inputs)
        if y_normalizer!= None:
            # print('normalizing y')
            labels = y_normalizer.normalize(labels)
        # print(batch_idx)
        inputs = inputs.to(device)
        labels = labels.to(device)
        # print('batch: ',inputs.shape,labels.shape)
        # print('batch: ',torch.max(inputs),torch.max(labels))
        # print('batch: ',torch.min(inputs),torch.min(labels))
        # print('inputbatchshape:', inputs.shape)
        # print('labelsbatchshape:',labels.shape)

        optimizer.zero_grad()
        output, J_loss, H_loss = model(inputs)
        # print('outputbatchshape:',output.shape)
        # print('debug:',output.size(),labels.size())
        
        loss = criterion(output, labels) + J_loss + H_loss
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()
        train_ins_error.append(loss)
    return torch.mean(torch.stack(train_ins_error))


def train_DFN(model, device, train_loader, optimizer, criterion, scheduler = None , x_normalizer = None,y_normalizer = None):
    
    model.train()
    train_ins_error = []
    for batch_idx, batch in enumerate(train_loader):
        labels = batch.y
        if x_normalizer!= None:
            # print('normalizing x')
            inputs = x_normalizer.normalize(inputs)
        if y_normalizer!= None:
            # print('normalizing y')
            labels = y_normalizer.normalize(labels)
        
        batch.x = batch.x.to(device)
        batch.mass = batch.mass.to(device)
        batch.L = batch.L.to(device)
        batch.evals = batch.evals.to(device)
        batch.evecs = batch.evecs.to(device)
        batch.gradX = batch.gradX.to(device)
        batch.gradY = batch.gradY.to(device)
        batch.face = batch.face.to(device)
        labels = labels.to(device)

        # print('batch: ',inputs.shape,labels.shape)
        # print('batch: ',torch.max(inputs),torch.max(labels))
        # print('batch: ',torch.min(inputs),torch.min(labels))
        # print('inputbatchshape:', inputs.shape)
        # print('labelsbatchshape:',labels.shape)

        optimizer.zero_grad()
        output = model(batch.x, 
               batch.mass, 
               L=batch.L, 
               evals=batch.evals, 
               evecs=batch.evecs, 
               gradX=batch.gradX, 
               gradY=batch.gradY, 
               faces=batch.face.T)
        # print('outputbatchshape:',output.shape)
        # print('debug:',output.size(),labels.size())
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()
        train_ins_error.append(loss)
    return torch.mean(torch.stack(train_ins_error))


def train_regular_autoregressive(model, device, train_loader, optimizer, criterion, scheduler = None , x_normalizer = None,y_normalizer = None):
    # TBD
    model.train()
    train_ins_error = []
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch['input']
        labels = batch['output']
        if x_normalizer!= None:
            # print('normalizing x')
            inputs = x_normalizer.normalize(inputs)
        if y_normalizer!= None:
            # print('normalizing y')
            labels = y_normalizer.normalize(labels)
        # print(batch_idx)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # print('inputbatchshape:', inputs.shape)
        # print('labelsbatchshape:',labels.shape)

        optimizer.zero_grad()
        output = model(inputs)
        # print('outputbatchshape:',output.shape)
        # print('debug:',output.size(),labels.size())
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()
        train_ins_error.append(loss)
    return torch.mean(torch.stack(train_ins_error))

def train(model, device, train_loader, optimizer, criterion, scheduler = None , x_normalizer = None,y_normalizer = None):
    
    model.train()
    train_ins_error = []
    for batch_idx, batch in enumerate(train_loader):
        if x_normalizer!= None:
            # print('normalizing x')
            batch.x = x_normalizer.normalize(batch.x)
        if y_normalizer!= None:
            # print('normalizing y')
            batch.y = y_normalizer.normalize(batch.y)
        # print(batch_idx)
        batch = batch.to(device)
        print('batchshape:', batch.shape)
        optimizer.zero_grad()
        output = model(batch)
        labels = batch.y
        # print('debug:',output.size(),labels.size())
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()
        train_ins_error.append(loss)
    return torch.mean(torch.stack(train_ins_error))

def train_mgn(model, device, train_loader, optimizer, criterion, scheduler = None, x_normalizer = None,y_normalizer = None,e_normalizer = None):
    
    model.train()
    train_ins_error = []
    for batch_idx, batch in enumerate(train_loader):
        # print(batch_idx)
        if x_normalizer!= None:
            # print('normalizing x')
            batch.x = x_normalizer.normalize(batch.x)
        if y_normalizer!= None:
            # print('normalizing y')
            batch.y = y_normalizer.normalize(batch.y)
        if e_normalizer!= None:
            # print('normalizing e')
            batch.edge_attr = e_normalizer.normalize(batch.edge_attr)
        batch = batch.to(device)
        # print(batch)
        optimizer.zero_grad()
        output = model(batch.x,batch.edge_index, batch.edge_attr)
        labels = batch.y
        # print('debug:',output.size(),labels.size())
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()
        train_ins_error.append(loss)
    return torch.mean(torch.stack(train_ins_error))
    
def test_regular(model, device, test_loader, criterion, x_normalizer = None,y_normalizer = None):
    
    model.eval()
    test_error = []
    for batch_idx, batch in enumerate(test_loader):
        inputs = batch['input']
        labels = batch['output']
        if x_normalizer!= None:
            # print('normalizing x')
            inputs = x_normalizer.normalize(inputs)
        if y_normalizer!= None:
            # print('normalizing y')
            labels = y_normalizer.normalize(labels)
        # print(batch_idx)
        inputs = inputs.to(device)
        labels = labels.to(device)

        print('inputbatchshape:', inputs.shape)
        print('labelsbatchshape:',labels.shape)

        with torch.no_grad():
            output = model(inputs)
            print('outputbatchshape:',output.shape)
        # print('debug:',output.size(),labels.size())
            loss = criterion(output, labels)
            test_error.append(loss)
    test_error = torch.stack(test_error)
    return torch.mean(test_error)


def test_DFN_output(model, device, test_loader, raw_mesh_path, x_normalizer = None,y_normalizer = None, foi = 'P'):
    
    model.eval()
    for batch_idx, batch in enumerate(test_loader):
        labels = batch.y
        if x_normalizer!= None:
            # print('normalizing x')
            inputs = x_normalizer.normalize(inputs)
        if y_normalizer!= None:
            # print('normalizing y')
            labels = y_normalizer.normalize(labels)
        
        batch.x = batch.x.to(device)
        batch.mass = batch.mass.to(device)
        batch.L = batch.L.to(device)
        batch.evals = batch.evals.to(device)
        batch.evecs = batch.evecs.to(device)
        batch.gradX = batch.gradX.to(device)
        batch.gradY = batch.gradY.to(device)
        batch.face = batch.face.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output = model(batch.x, 
                batch.mass, 
                L=batch.L, 
                evals=batch.evals, 
                evecs=batch.evecs, 
                gradX=batch.gradX, 
                gradY=batch.gradY, 
                faces=batch.face.T)
        mesh = pv.read(osj(raw_mesh_path,'mesh{:d}.vtp'.format(batch.idx.item())))
        R2, MAE, RMSE = cal_error(y_normalizer.denormalize(output.detach().cpu()),batch.y.detach().cpu())
        mesh.point_data['{:s}_norm_prediction'.format(foi)] = output.detach().cpu().numpy()
        mesh.point_data['{:s}_norm_label'.format(foi)] = labels.detach().cpu().numpy()
        mesh.point_data['{:s}_prediction'.format(foi)] = y_normalizer.denormalize(output.detach().cpu()).numpy()
        mesh.point_data['{:s}_label'.format(foi)] = batch.y.detach().cpu().numpy()
        mesh.point_data['{:s}_diff'.format(foi)] = mesh.point_data['P_label'] - mesh.point_data['P_prediction']
        mesh.point_data['R2']=R2.numpy()
        mesh.point_data['MAE']=MAE.numpy()
        mesh.point_data['RMSE']=RMSE.numpy()
        _,MAE_base,_ = cal_error((batch.y*0).detach().cpu(),batch.y.detach().cpu())
        mesh.point_data['RelativeMAE']=MAE.numpy()/MAE_base.numpy()

        for i in range(batch.x.shape[-1]):
            mesh.point_data['x{:d}'.format(i)] =batch.x[:,i].detach().cpu().numpy()
    return mesh, (R2, MAE, RMSE)

def test_DFN(model, device, test_loader, criterion, x_normalizer = None,y_normalizer = None):
    
    model.eval()
    test_error = []
    for batch_idx, batch in enumerate(test_loader):
        labels = batch.y
        if x_normalizer!= None:
            # print('normalizing x')
            inputs = x_normalizer.normalize(inputs)
        if y_normalizer!= None:
            # print('normalizing y')
            labels = y_normalizer.normalize(labels)
        
        batch.x = batch.x.to(device)
        batch.mass = batch.mass.to(device)
        batch.L = batch.L.to(device)
        batch.evals = batch.evals.to(device)
        batch.evecs = batch.evecs.to(device)
        batch.gradX = batch.gradX.to(device)
        batch.gradY = batch.gradY.to(device)
        batch.face = batch.face.to(device)
        labels = labels.to(device)

        # print('batch: ',inputs.shape,labels.shape)
        # print('batch: ',torch.max(inputs),torch.max(labels))
        # print('batch: ',torch.min(inputs),torch.min(labels))
        # print('inputbatchshape:', inputs.shape)
        # print('labelsbatchshape:',labels.shape)

        with torch.no_grad():
            output = model(batch.x, 
                batch.mass, 
                L=batch.L, 
                evals=batch.evals, 
                evecs=batch.evecs, 
                gradX=batch.gradX, 
                gradY=batch.gradY, 
                faces=batch.face.T)
            # print('outputbatchshape:',output.shape)
            # print('debug:',output.size(),labels.size())
            loss = criterion(output, labels)
            test_error.append(loss)
    return torch.mean(torch.stack(test_error))

def test(model, device, test_loader, criterion, x_normalizer = None,y_normalizer = None):
    # define trainloader
    model.eval()
    test_error = []
    for batch_idx, batch in enumerate(test_loader):
        if x_normalizer!= None:
            # print('normalizing x')
            batch.x = x_normalizer.normalize(batch.x)
        if y_normalizer!= None:
            # print('normalizing y')
            batch.y = y_normalizer.normalize(batch.y)
        batch = batch.to(device)
        with torch.no_grad():
            output = model(batch)
            labels = batch.y
            loss = criterion(output, labels)
        test_error.append(loss)
    test_error = torch.stack(test_error)
    return torch.mean(test_error)

def test_mgn(model, device, test_loader, criterion,x_normalizer = None,y_normalizer = None,e_normalizer = None):
    model.eval()
    
    # define trainloader
    test_error = []
    for batch_idx, batch in enumerate(test_loader):
        if x_normalizer!= None:
            # print('normalizing x')
            batch.x = x_normalizer.normalize(batch.x)
        if y_normalizer!= None:
            # print('normalizing y')
            batch.y = y_normalizer.normalize(batch.y)
        if e_normalizer!= None:
            # print('normalizing e')
            batch.edge_attr = e_normalizer.normalize(batch.edge_attr)
        batch = batch.to(device)
        with torch.no_grad():
            output = model(batch.x,batch.edge_index, batch.edge_attr)
            labels = batch.y
            loss = criterion(output, labels)
        test_error.append(loss)
    test_error = torch.stack(test_error)
    return torch.mean(test_error)

def test_mgn_output(model, device, test_loader, criterion,x_normalizer = None,y_normalizer = None,e_normalizer = None):
    model.eval()
    # define trainloader
    test_error = []
    for batch_idx, batch in enumerate(test_loader):
        if x_normalizer!= None:
            # print('normalizing x')
            batch.x = x_normalizer.normalize(batch.x)
        # if y_normalizer!= None:
        #     # print('normalizing y')
        #     batch.y = y_normalizer.normalize(batch.y)
        if e_normalizer!= None:
            # print('normalizing e')
            batch.edge_attr = e_normalizer.normalize(batch.edge_attr)
        batch = batch.to(device)
        with torch.no_grad():
            temp_output = model(batch.x,batch.edge_index, batch.edge_attr) # normed
            labels = batch.y  # physical

            # convert normed to physical 
            output = x_normalizer.denormalize(temp_output)
            loss = criterion(output, labels) # loss in physical domain 
            
        test_error.append(loss)
    test_error = torch.stack(test_error)
    return torch.mean(test_error), output


def test_output(model, device, test_loader, criterion, normalizer, path_for_mesh, path_to_raw, mesh_name = 'mesh', data_name = 'p', original_id =[] ):
    # define trainloader
    test_error = []
    R2s, MAEs, RMSEs = [],[],[]
    for batch_idx, batch in tqdm(enumerate(test_loader)):
        batch = batch.to(device)
        with torch.no_grad():
            output = model(batch)
            labels = batch.y
            label_denorm = normalizer.denormalize(labels)
            output_denorm = normalizer.denormalize(output)
            R2, MAE, RMSE = cal_error(output_denorm,label_denorm)
            loss = criterion(output, labels)
        test_error.append(loss)
        R2s.append(R2);MAEs.append(MAE);RMSEs.append(RMSE)
        idx = batch.idx.detach().cpu().numpy()[0]
        # print(idx)
        mesh = pv.read(osj(path_to_raw,'raw/mesh{:d}.vtp'.format(idx)))
        mesh.point_data['l_'+data_name] = label_denorm.cpu()
        mesh.point_data['p_'+data_name] = output_denorm.detach().cpu()
        mesh.point_data['d_'+data_name] = output_denorm.detach().cpu()-label_denorm.cpu()
        if len(original_id) == 0:
            mesh.save(osj(path_for_mesh, mesh_name+'trainid_{:d}.vtk'.format(idx)))
        else:
            mesh.save(osj(path_for_mesh, mesh_name+'_{:d}_{:s}.vtk'.format(original_id[idx],data_name)))
    test_error = torch.stack(test_error)
    R2s = torch.stack(R2s);MAEs = torch.stack(MAEs);RMSEs = torch.stack(RMSEs)
    torch.save({
        'R2s':R2s,
        'MAEs':MAEs,
        'RMSEs':RMSEs,
        'test_error':test_error,},
        osj(path_for_mesh,mesh_name+'errors_R2_{:.2E}_MAE_{:.2E}_RSME{:.2E}_loss{:.2E}.pt'.format(torch.mean(R2s),
                                                                              torch.mean(MAEs),
                                                                              torch.mean(RMSEs),
                                                                              torch.mean(test_error)))
        )
    return torch.mean(test_error)



def test_output_rollout(model, device, criterion, dataset,normalizer, path_for_mesh, path_to_raw, mesh_name = 'mesh', data_name = 'p' ):
    # define trainloader
    mesh = pv.read(osj(path_to_raw,'raw/mesh{:d}.vtk'.format(0)))
    test_error = [torch.tensor(0,dtype =torch.float32,device = device)]
    R2s, MAEs, RMSEs = [torch.tensor(1,dtype =torch.float32,device = device)],[torch.tensor(0,dtype =torch.float32,device = device)],[torch.tensor(0,dtype =torch.float32,device = device)]
    foi_i = dataset[0].clone()
    for i in tqdm(range(len(dataset)-1)):
        with torch.no_grad():
            foi_i =foi_i.to(device)
            output = model(foi_i)
        # apply recursive
        foi_i.x = output
        
        labels = (dataset[i+1].x).to(device)       
        label_denorm = normalizer.denormalize(labels)
        output_denorm = normalizer.denormalize(output)
        R2, MAE, RMSE = cal_error(output_denorm,label_denorm)
        loss = criterion(output, labels)
        test_error.append(loss)
        R2s.append(R2);MAEs.append(MAE);RMSEs.append(RMSE)
        idx = dataset[i+1].idx
        # print(idx)
        mesh.point_data['l_'+data_name] = label_denorm.cpu()
        mesh.point_data['p_'+data_name] = output_denorm.detach().cpu()
        mesh.point_data['d_'+data_name] = output_denorm.detach().cpu()-label_denorm.cpu()
        mesh.save(osj(path_for_mesh, mesh_name+'_{:d}.vtk'.format(idx)))
        
    test_error = torch.stack(test_error)
    # print(R2s,MAEs,RMSEs)
    R2s = torch.stack(R2s);MAEs = torch.stack(MAEs);RMSEs = torch.stack(RMSEs)
    torch.save({
        'R2s':MAEs,
        'MAEs':MAEs,
        'RMSEs':RMSEs,
        'test_error':test_error,},
        osj(path_for_mesh,mesh_name+'errors_R2_{:.2E}_MAE_{:.2E}_RSME{:.2E}_loss{:.2E}.pt'.format(torch.max(R2s),
                                                                              torch.max(MAEs),
                                                                              torch.max(RMSEs),
                                                                              torch.max(test_error)))
        )
    return R2s.cpu().numpy(),MAEs.cpu().numpy(),RMSEs.cpu().numpy(), test_error.cpu().numpy()




# from torcheval.metrics import R2Score
def cal_error(p_y,l_y):
    # print(p_y.size(), l_y.size())
    assert l_y.shape == p_y.shape
    N = len(l_y)
    # metric = R2Score()
    # metric.update(p_y,l_y)
    # R2 = metric.compute()
    MAE = torch.sum(torch.abs(l_y-p_y))/N
    RMSE = (torch.sum((l_y-p_y)**2)/N)**0.5
    return MAE, MAE, RMSE






from collections.abc import Iterable
from math import log, cos, pi, floor

from torch.optim.lr_scheduler import _LRScheduler


class CyclicCosineDecayLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 init_decay_epochs,
                 min_decay_lr,
                 restart_interval=None,
                 restart_interval_multiplier=None,
                 restart_lr=None,
                 warmup_epochs=None,
                 warmup_start_lr=None,
                 last_epoch=-1,
                 verbose=False):
        """
        Initialize new CyclicCosineDecayLR object.
        :param optimizer: (Optimizer) - Wrapped optimizer.
        :param init_decay_epochs: (int) - Number of initial decay epochs.
        :param min_decay_lr: (float or iterable of floats) - Learning rate at the end of decay.
        :param restart_interval: (int) - Restart interval for fixed cycles.
            Set to None to disable cycles. Default: None.
        :param restart_interval_multiplier: (float) - Multiplication coefficient for geometrically increasing cycles.
            Default: None.
        :param restart_lr: (float or iterable of floats) - Learning rate when cycle restarts.
            If None, optimizer's learning rate will be used. Default: None.
        :param warmup_epochs: (int) - Number of warmup epochs. Set to None to disable warmup. Default: None.
        :param warmup_start_lr: (float or iterable of floats) - Learning rate at the beginning of warmup.
            Must be set if warmup_epochs is not None. Default: None.
        :param last_epoch: (int) - The index of the last epoch. This parameter is used when resuming a training job. Default: -1.
        :param verbose: (bool) - If True, prints a message to stdout for each update. Default: False.
        """

        if not isinstance(init_decay_epochs, int) or init_decay_epochs < 1:
            raise ValueError("init_decay_epochs must be positive integer, got {} instead".format(init_decay_epochs))

        if isinstance(min_decay_lr, Iterable) and len(min_decay_lr) != len(optimizer.param_groups):
            raise ValueError("Expected len(min_decay_lr) to be equal to len(optimizer.param_groups), "
                             "got {} and {} instead".format(len(min_decay_lr), len(optimizer.param_groups)))

        if restart_interval is not None and (not isinstance(restart_interval, int) or restart_interval < 1):
            raise ValueError("restart_interval must be positive integer, got {} instead".format(restart_interval))

        if restart_interval_multiplier is not None and \
                (not isinstance(restart_interval_multiplier, float) or restart_interval_multiplier <= 0):
            raise ValueError("restart_interval_multiplier must be positive float, got {} instead".format(
                restart_interval_multiplier))

        if isinstance(restart_lr, Iterable) and len(restart_lr) != len(optimizer.param_groups):
            raise ValueError("Expected len(restart_lr) to be equal to len(optimizer.param_groups), "
                             "got {} and {} instead".format(len(restart_lr), len(optimizer.param_groups)))

        if warmup_epochs is not None:
            if not isinstance(warmup_epochs, int) or warmup_epochs < 1:
                raise ValueError(
                    "Expected warmup_epochs to be positive integer, got {} instead".format(type(warmup_epochs)))

            if warmup_start_lr is None:
                raise ValueError("warmup_start_lr must be set when warmup_epochs is not None")

            if not (isinstance(warmup_start_lr, float) or isinstance(warmup_start_lr, Iterable)):
                raise ValueError("warmup_start_lr must be either float or iterable of floats, got {} instead".format(
                    warmup_start_lr))

            if isinstance(warmup_start_lr, Iterable) and len(warmup_start_lr) != len(optimizer.param_groups):
                raise ValueError("Expected len(warmup_start_lr) to be equal to len(optimizer.param_groups), "
                                 "got {} and {} instead".format(len(warmup_start_lr), len(optimizer.param_groups)))

        group_num = len(optimizer.param_groups)
        self._warmup_start_lr = [warmup_start_lr] * group_num if isinstance(warmup_start_lr, float) else warmup_start_lr
        self._warmup_epochs = 0 if warmup_epochs is None else warmup_epochs
        self._init_decay_epochs = init_decay_epochs
        self._min_decay_lr = [min_decay_lr] * group_num if isinstance(min_decay_lr, float) else min_decay_lr
        self._restart_lr = [restart_lr] * group_num if isinstance(restart_lr, float) else restart_lr
        self._restart_interval = restart_interval
        self._restart_interval_multiplier = restart_interval_multiplier
        super(CyclicCosineDecayLR, self).__init__(optimizer, last_epoch, verbose=verbose)

    def get_lr(self):

        if self._warmup_epochs > 0 and self.last_epoch < self._warmup_epochs:
            return self._calc(self.last_epoch,
                              self._warmup_epochs,
                              self._warmup_start_lr,
                              self.base_lrs)

        elif self.last_epoch < self._init_decay_epochs + self._warmup_epochs:
            return self._calc(self.last_epoch - self._warmup_epochs,
                              self._init_decay_epochs,
                              self.base_lrs,
                              self._min_decay_lr)
        else:
            if self._restart_interval is not None:
                if self._restart_interval_multiplier is None:
                    cycle_epoch = (self.last_epoch - self._init_decay_epochs - self._warmup_epochs) % self._restart_interval
                    lrs = self.base_lrs if self._restart_lr is None else self._restart_lr
                    return self._calc(cycle_epoch,
                                      self._restart_interval,
                                      lrs,
                                      self._min_decay_lr)
                else:
                    n = self._get_n(self.last_epoch - self._warmup_epochs - self._init_decay_epochs)
                    sn_prev = self._partial_sum(n)
                    cycle_epoch = self.last_epoch - sn_prev - self._warmup_epochs - self._init_decay_epochs
                    interval = self._restart_interval * self._restart_interval_multiplier ** n
                    lrs = self.base_lrs if self._restart_lr is None else self._restart_lr
                    return self._calc(cycle_epoch,
                                      interval,
                                      lrs,
                                      self._min_decay_lr)
            else:
                return self._min_decay_lr

    def _calc(self, t, T, lrs, min_lrs):
        return [min_lr + (lr - min_lr) * ((1 + cos(pi * t / T)) / 2)
                for lr, min_lr in zip(lrs, min_lrs)]

    def _get_n(self, epoch):
        _t = 1 - (1 - self._restart_interval_multiplier) * epoch / self._restart_interval
        return floor(log(_t, self._restart_interval_multiplier))

    def _partial_sum(self, n):
        return self._restart_interval * (1 - self._restart_interval_multiplier ** n) / (
                    1 - self._restart_interval_multiplier)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def cal_epoches(ii, 
                init_decay_epochs=500,
                restart_interval=300,
                warmup_epochs=200,
                restart_interval_multiplier=1.5):
    epoch = init_decay_epochs + warmup_epochs
    for i in range(ii):
        epoch += int(restart_interval*restart_interval_multiplier**i)
    return epoch 

def epoch2cosepcoh(my_epoch,**kargs):
    ii=0
    epoch = 0
    epoch_list = [epoch]
    while epoch<= my_epoch:
        ii+=1
        epoch = cal_epoches(ii, **kargs)+1   
        epoch_list.append(epoch)
    return epoch_list[-1], epoch_list
