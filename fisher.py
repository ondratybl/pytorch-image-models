import torch
from torch.func import functional_call, vmap, grad, jacrev
from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC, PMatDiag
from contextlib import suppress


def get_ntk_tenas(model, output):
    # based on https://github.com/VITA-Group/TENAS/blob/main/prune_tenas.py
    grads = []
    for _idx in range(len(output)):
        output[_idx:_idx+1].backward(torch.ones_like(output[_idx:_idx+1]), retain_graph=True)
        grad = []
        for name, W in model.named_parameters():
            if 'weight' in name and W.grad is not None:
                grad.append(W.grad.view(-1).detach())
        grads.append(torch.cat(grad, -1))
        model.zero_grad()
    grads = torch.stack(grads, 0)
    ntk = torch.einsum('nc,mc->nm', [grads, grads])
    return torch.linalg.eigvalsh(ntk)


def cholesky_covariance(output):

    # Cholesky decomposition of covariance matrix (notation from Theorem 1 in https://sci-hub.se/10.2307/2345957)
    alpha = 0.000001  # label smoothing for stability

    alpha = torch.tensor(alpha, dtype=torch.float16, device=output.device)

    prob = torch.nn.functional.softmax(output, dim=1) * (1 - alpha) + alpha / output.shape[1]
    q = torch.ones_like(prob) - torch.cumsum(prob, dim=1)
    q[:, -1] = torch.zeros_like(q[:, -1])
    q_shift = torch.roll(q, shifts=1, dims=1)
    q_shift[:, 0] = torch.ones_like(q_shift[:, 0])
    d = torch.sqrt(prob * q / q_shift)

    L = -torch.matmul(torch.unsqueeze(prob, dim=2), 1 / torch.transpose(torch.unsqueeze(q, dim=2), dim0=1, dim1=2))
    L = torch.nan_to_num(L, neginf=0.)
    L = L * (1 - torch.eye(L.shape[1], device=output.device, dtype=output.dtype).repeat(L.shape[0], 1, 1)) + \
        torch.eye(L.shape[1], device=output.device, dtype=output.dtype).repeat(L.shape[0], 1,
                                                                               1)  # replace diagonal elements by 1.
    L = L * (1 - torch.triu(torch.ones(L.shape[1], L.shape[2], device=output.device, dtype=output.dtype),
                            diagonal=1).repeat(L.shape[0], 1, 1))  # replace upper diagonal by 0
    L = torch.matmul(L, torch.diag_embed(d))  # multiply columns

    # Test
    cov_true = torch.diag_embed(prob) - torch.matmul(torch.unsqueeze(prob, dim=2),
                                                     torch.transpose(torch.unsqueeze(prob, dim=2), dim0=1, dim1=2))
    cov_cholesky = torch.matmul(L, torch.transpose(L, dim0=1, dim1=2))

    max_error = torch.abs(cov_true - cov_cholesky).max().item()
    if max_error > 1.0e-4:
        print(f'Cholesky decomposition back-test error with max error {max_error}')
    return L.detach()


def gradient_batch(model, input):

    # Use vmap to compute per-sample gradient wrt model parameters, model has one output dimension
    def compute_prediction(params, buffers, sample):
        return functional_call(model, (params, buffers), (sample.unsqueeze(0),))
    ft_compute_grad = grad(compute_prediction)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0))
    grads = list(ft_compute_sample_grad(
        {k: v.detach() for k, v in model.named_parameters()},
        {k: v.detach() for k, v in model.named_buffers()},
        input
    ).values())
    return torch.cat([torch.flatten(i, start_dim=1, end_dim=-1) for i in grads], dim=1)  # (batch_size, model_param_count)


def jacobian_batch_efficient(model, input):

    model.zero_grad()

    params_grad = {k: v.detach() for k, v in model.named_parameters() if ('weight' in k and 'bn' not in k)}
    #params_grad = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def jacobian_sample(sample):
        def compute_prediction(params_grad):
            params = params_grad.copy()
            params.update({k: v.detach() for k, v in model.named_parameters() if k not in params_grad.keys()})
            return torch.softmax(functional_call(model, (params, buffers), (sample.unsqueeze(0),)).squeeze(0), dim=1)

        return jacrev(compute_prediction)(params_grad)

    jacobian_dict = vmap(jacobian_sample)(input)
    ret = torch.cat([torch.flatten(v, start_dim=2, end_dim=-1) for v in jacobian_dict.values()], dim=2)

    return ret.detach()


def jacobian_batch(model, input, num_classes=1000):
    import time
    start_time = time.time()
    class WrappedModel(torch.nn.Module):
        def __init__(self, original_model, dimension):
            super(WrappedModel, self).__init__()
            self.original_model = original_model
            self.dimension = dimension

        def forward(self, x):
            original_output = self.original_model(x)
            # Extract the first dimension of the output
            first_dimension_output = original_output[:, self.dimension].squeeze(0)
            return first_dimension_output

    # Get per-sample jacobian of multidimensional output wrt model parameters
    grads = []
    for dim in range(num_classes):  # for each output dim compute (batch_size, model_param_count) tensor
        model.zero_grad()
        grads.append(gradient_batch(WrappedModel(model, dim), input))

    print(time.time() - start_time)
    return torch.transpose(torch.stack(grads), dim0=0, dim1=1)  # (batch_size, num_classes, model_param_count)


def get_fisher(model, loader, num_classes):

    # works only for models with CNN and linear layers only

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    f_diag = FIM(
        model=model,
        loader=loader,
        representation=PMatDiag,
        n_output=num_classes,
        variant='classif_logits',
        device=device
    ).frobenius_norm().detach().item()

    f_fkac = FIM(
        model=model,
        loader=loader,
        representation=PMatKFAC,
        n_output=num_classes,
        variant='classif_logits',
        device=device
    ).frobenius_norm().detach().item()

    return f_fkac, f_diag


def get_eigenvalues(model, input, output, ntk_old, batch):

    # output.dtype == torch.float16
    # input.dtype == torch.float32

    output = output.float()

    # ntk = A*A^T, fisher = A^T*A
    #cholesky = cholesky_covariance(output)  # torch.float16
    jacobian = jacobian_batch_efficient(model, input)  # RuntimeError: Input type (torch.cuda.HalfTensor) and weight type (torch.cuda.FloatTensor) should be the same

    #A = torch.matmul(cholesky, jacobian).detach()
    ntk = torch.mean(torch.matmul(jacobian, torch.transpose(jacobian, dim0=1, dim1=2)), dim=0).detach()

    # get eigenvalues
    try:
        eig_ntk = torch.linalg.eigvalsh(ntk).detach()  # per population ntk
    except Exception as e:
        print(f"Per population NTK failed in batch {batch}: {e}")
        eig_ntk = torch.full((1000,), float('nan'), device=output.device)
    ntk_new = ((ntk_old * batch + ntk) / (batch + 1)).detach()

    return eig_ntk, ntk_new