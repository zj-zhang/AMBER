import numpy as np
import torch
from pdb import set_trace as bp


__all__ = ['get_ntk']


def get_ntk(inputs, targets, network, criterion=torch.nn.BCELoss(reduction='none'), train_mode=True):
    if not criterion:
        criterion = torch.nn.BCELoss(reduction='none')
    device = torch.cuda.current_device()
    if train_mode:
        network.train()
    else:
        network.eval()
    grads_x = [] # size: #training samples. grads of all W from each sample
    # device management should happen outside of get_ntk
    #inputs = torch.Tensor(inputs).cuda(device=device, non_blocking=True)
    #targets = torch.Tensor(targets).cuda(device=device, non_blocking=True)

    ch = 16
    for idx in np.arange(0, len(inputs), ch):
        logit = network(inputs[idx:idx+ch])
        # choose specific class for loss
        loss = criterion(logit, targets[idx:idx+ch])
        for _idx in range(len(inputs[idx:idx+ch])):
            # logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
            # use criterion to get gradient
            loss[_idx:_idx+1].backward(torch.ones_like(loss[_idx:_idx+1]), retain_graph=_idx < len(inputs[idx:idx+ch])-1)
            grad = []
            for name, W in network.named_parameters():
                if 'classifier' in name or 'fc' in name or 'out' in name: continue
                if 'weight' in name and W.grad is not None:
                    grad.append(W.grad.view(-1).detach())
            grads_x.append(torch.cat(grad, -1).detach())
            network.zero_grad()
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    _grads_x = torch.stack([item.cuda() for item in grads_x], 0)
    torch.cuda.empty_cache()
    ntk = torch.einsum('nc,mc->nm', [_grads_x, _grads_x])
    del _grads_x
    try:
        # eigenvalues, _ = torch.symeig(ntk)  # ascending
        eigenvalues, _ = torch.linalg.eigh(ntk, UPLO='L')
    except:
        ntk[ntk == float("Inf")] = 0
        ntk[ntk == 0] = ntk.max() # TODO avoid inf in ntk
        eigenvalues, _ = torch.linalg.eigh(ntk + ntk.mean().item() * torch.eye(ntk.shape[0]).cuda() * 1e-4, UPLO='L')  # ascending
    _cond = torch.div(eigenvalues[-1], eigenvalues[0])
    if torch.isnan(_cond):
        return -1, float(loss.mean().item) # bad gradients
    else:
        return _cond.item(), float(loss.mean().item())
