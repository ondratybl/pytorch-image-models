from xautodl.datasets.DownsampledImageNet import ImageNet16
from nats_bench import create
from xautodl.models import get_cell_based_tiny_net
from fisher import cholesky_covariance, jacobian_batch_efficient, get_ntk_tenas_new, get_ntk_tenas_new_probs
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import argparse
import wandb
import random
import gc
import numpy as np


def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


def get_model(api, dataset, index, seed, hp, pretrained):
    prepare_seed(
        seed)  # we call similarly as in https://github.com/D-X-Y/AutoDL-Projects/blob/main/xautodl/procedures/funcs_nasbench.py#L82
    model = get_cell_based_tiny_net(api.get_net_config(index, dataset))
    if pretrained:  # weights overridden by pretrained ones
        params = {seed: api.get_net_param(index, dataset, seed=seed, hp=hp)}  # seed=None returns all seeds
        model.load_state_dict(next(iter(params.values())))
        model.eval()
    else:
        model.train()
        torch.func.replace_all_batch_norm_modules_(model)  # TODO: discuss with Lukas

    class ModelWrapper(torch.nn.Module):
        def __init__(self, original_model):
            super(ModelWrapper, self).__init__()
            self.original_model = original_model

        def forward(self, x):
            output = self.original_model(x)
            return output[1]  # Return the second element of the output tuple

    return ModelWrapper(model)


def get_condition_number(matrix):
    try:
        lambdas = torch.linalg.eigvalsh(matrix).detach()
    except torch._C._LinAlgError as e:
        error_message = str(e)
        if "torch.linalg.eigvalsh: The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues" in error_message:
            print(
                "Error in eigenvalue calculation: The input matrix is ill-conditioned or has too many repeated eigenvalues.")
        else:
            print("Unexpected torch._C._LinAlgError occurred:")
            print(error_message)

    if lambdas.min().item() > 0:
        return lambdas.max().item() / lambdas.min().item()
    else:
        return None


def get_ntk(model, input):

    cholesky = cholesky_covariance(model(input))
    jacobian = jacobian_batch_efficient(model, input)

    ntk = torch.mean(torch.matmul(jacobian, torch.transpose(jacobian, dim0=1, dim1=2)), dim=0).detach()
    A = torch.matmul(cholesky, jacobian).detach()

    del jacobian, cholesky
    gc.collect()
    torch.cuda.empty_cache()

    ntk_p = torch.mean(torch.matmul(A, torch.transpose(A, dim0=1, dim1=2)), dim=0).detach()

    del A
    gc.collect()
    torch.cuda.empty_cache()

    return torch.linalg.eigvalsh(ntk), torch.linalg.eigvalsh(ntk_p)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FIM for NATS-bench', add_help=False)
    # Add arguments
    parser.add_argument('--dataset', type=str, default='Data/ImageNet16-120', help='Dataset path.')
    parser.add_argument('--models', type=str, default='NATS-tss-v1_0-3ffb9-simple', help='Models path.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights.')
    parser.add_argument('--epochs_trained', type=str, default='200', help='Number of training epochs.')
    parser.add_argument('--n_model_min', type=int, default=0, help='Model from.')
    parser.add_argument('--n_model_max', type=int, default=10, help='Model to.')
    parser.add_argument('--num-iterations', type=int, default=1, help='Number of iterations.')
    parser.add_argument('--use-train', action='store_true', help='Use train split, not test split.')
    parser.add_argument('--name-wandb', default='default_wandb_name', type=str, metavar='NAME',
                        help='Name of wandb experiment to be shown in the interface')
    parser.add_argument('--notes-wandb', default='', type=str, metavar='NAME',
                        help='Longer description of the run, like a -m commit message in git')
    parser.add_argument('--tags-wandb', default='default', type=str, metavar='NAME',
                        help='tags of the run')
    parser.add_argument('--num-batches', default=10, type=int, metavar='NAME',
                        help='Number of batches to be watched')

    # Parse the arguments
    args = parser.parse_args()
    args.dataset_name = args.dataset.split('/')[-1]

    # wandb
    wandb.init(
        project=None,
        config=args,
        name=args.name_wandb,
        notes=args.notes_wandb,
        tags=[args.tags_wandb],
    )

    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # API
    prepare_seed(1)
    loader = DataLoader(ImageNet16(args.dataset, False, transforms.Compose([transforms.ToTensor()]), 120), batch_size=args.batch_size,
                        shuffle=True)
    api = create(args.models, 'tss', fast_mode=True, verbose=True)

    # Iterate models and seeds
    for index in range(args.n_model_min, args.n_model_max):
        try:
            for seed in api.get_net_param(index, args.dataset_name, None, hp=args.epochs_trained).keys():

                model = get_model(api, args.dataset_name, index, seed, args.epochs_trained, args.pretrained).to(device)

                # get train statistics
                info_per = api.get_more_info(index, args.dataset_name, hp=args.epochs_trained, is_random=seed)
                info_cost = api.get_cost_info(index, args.dataset_name, hp=args.epochs_trained)

                # get compute & log
                for batch, (input, _) in enumerate(loader)[: args.num_batches]:

                    input = input.to(device)

                    # tenas
                    eig_tenas = get_ntk_tenas_new(model, model(input)).detach()
                    eig_tenas_p = get_ntk_tenas_new_probs(model, model(input)).detach()

                    info_tenas = {
                        'tenas_cond': eig_tenas.max() / eig_tenas.min() if eig_tenas.min() > 0 else None,
                        'tenas_max': eig_tenas.max(),
                        'tenas_coef': eig_tenas.std() / eig_tenas.mean() if eig_tenas.mean() > 0 else None,

                        'tenas_p_cond': eig_tenas_p.max() / eig_tenas_p.min() if eig_tenas_p.min() > 0 else None,
                        'tenas_p_max': eig_tenas_p.max(),
                        'tenas_p_coef': eig_tenas_p.std() / eig_tenas_p.mean() if eig_tenas_p.mean() > 0 else None,
                    }

                    # ntk
                    eig_ntk, eig_ntk_p = get_ntk(model, input)

                    if torch.isnan(eig_ntk).sum().item() > 0 or torch.isnan(eig_ntk_p).sum().item():
                        print(f'NTK contains NaN. Output is stored.')
                        np.savetxt(f'output/output_{index}_{seed}_{batch}.csv', model(input).detach().cpu().numpy(),
                                   delimiter=',')

                    info_ntk = {
                        'ntk_cond': eig_ntk.max() / eig_ntk.min() if eig_ntk.min() > 0 else None,
                        'ntk_max': eig_ntk.max(),
                        'ntk_coef': eig_ntk.std() / eig_ntk.mean() if eig_ntk.mean() > 0 else None,

                        'ntk_p_cond': eig_ntk_p.max() / eig_ntk_p.min() if eig_ntk_p.min() > 0 else None,
                        'ntk_p_max': eig_ntk_p.max(),
                        'ntk_p_coef': eig_ntk_p.std() / eig_ntk_p.mean() if eig_ntk_p.mean() > 0 else None,
                    }

                    # log
                    combined_dict = {'index': index, 'seed': seed, 'hp': args.epochs_trained, 'batch': batch}
                    combined_dict.update(info_ntk)
                    combined_dict.update(info_tenas)
                    combined_dict.update(info_per)
                    combined_dict.update(info_cost)
                    wandb.log(combined_dict)

        except AssertionError as e:
            print(f"Model not found: {e}")
