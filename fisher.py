import torch


def cholesky_covariance(logits):

    # Cholesky decomposition (notation from Theorem 1 in https://sci-hub.se/10.2307/2345957)
    prob = torch.nn.functional.softmax(logits, dim=1)
    q = torch.ones(prob.shape) - torch.cumsum(prob, dim=1)
    q[:, -1] = torch.zeros(q[:, -1].shape)
    q_shift = torch.roll(q, shifts=1, dims=1)
    q_shift[:, 0] = torch.ones(q_shift[:, 0].shape)
    d = torch.sqrt(prob * q / q_shift)

    L = - torch.matmul(torch.unsqueeze(prob, dim=2), 1 / torch.transpose(torch.unsqueeze(q, dim=2), dim0=1, dim1=2))
    L = torch.nan_to_num(L, neginf=0.)
    L = L * (1 - torch.eye(L.shape[1]).repeat(L.shape[0], 1, 1)) + torch.eye(L.shape[1]).repeat(L.shape[0], 1,
                                                                                                1)  # replace diagonal elements by 1.
    L = L * (1 - torch.triu(torch.ones(L.shape[1], L.shape[2]), diagonal=1).repeat(L.shape[0], 1,
                                                                                   1))  # replace upper diagonal by 0
    L = torch.matmul(L, torch.diag_embed(d))  # multiply columns

    # Test
    cov_true = torch.diag_embed(prob) - torch.matmul(torch.unsqueeze(prob, dim=2),
                                                     torch.transpose(torch.unsqueeze(prob, dim=2), dim0=1, dim1=2))
    cov_cholesky = torch.matmul(L, torch.transpose(L, dim0=1, dim1=2))
    if torch.abs(cov_true - cov_cholesky).max().item() > 1.0e-5:
        print('Cholesky decomposition back-test error.')

    return L


def gradient_batch(model, input):

    from torch.func import functional_call, vmap, grad

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


def jacobian_batch(model, input, num_classes=1000):
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

    grads = []
    for dim in range(num_classes):
        grads.append(gradient_batch(WrappedModel(model, dim), input))
    return torch.transpose(torch.stack(grads), dim0=0, dim1=1)  # (batch_size, num_classes, model_param_count)


def get_eigenvalues(model, input, output, num_classes=20):
    A = torch.matmul(cholesky_covariance(output)[:, :num_classes, :num_classes], jacobian_batch(model, input, num_classes=num_classes))
    ntk = torch.mean(torch.matmul(A, torch.transpose(A, dim0=1, dim1=2)), dim=0)
    return torch.linalg.eigvalsh(ntk)