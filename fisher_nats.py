from xautodl.datasets.DownsampledImageNet import ImageNet16
from nats_bench import create
from xautodl.models import get_cell_based_tiny_net
from fisher import get_eigenvalues, get_ntk_tenas_new
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import argparse
import wandb


def get_model(api, dataset, index, seed, hp):

    model = get_cell_based_tiny_net(api.get_net_config(index, dataset))
    params = {seed: api.get_net_param(index, dataset, seed=seed, hp=hp)}  # seed=None returns all seeds
    model.load_state_dict(next(iter(params.values())))

    class ModelWrapper(torch.nn.Module):
        def __init__(self, original_model):
            super(ModelWrapper, self).__init__()
            self.original_model = original_model

        def forward(self, x):
            output = self.original_model(x)
            return output[1]  # Return the second element of the output tuple

    return ModelWrapper(model)


def compute(model, index, seed, loader, num_fisher, num_tenas, device):

    model.eval()

    # NTK
    ntk = torch.zeros(120, 120, device=device)
    for batch, data in enumerate(loader):

        if batch >= num_fisher:
            break

        input, _ = data
        input = input.to(device)
        ntk = get_eigenvalues(model, input, model(input), ntk, batch)

        if batch % 100 == 0:
            print(f'Index {index} seed {seed} batch {batch}')
            if torch.cuda.is_available():
                print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
                print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
                print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
                torch.cuda.reset_peak_memory_stats()

    ntk = ntk.float() / 1000000

    print(f'Index {index} seed {seed} before TENAS')
    if torch.cuda.is_available():
        print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
        torch.cuda.reset_peak_memory_stats()

    # TENAS
    output = []
    for input, _ in list(loader)[:num_tenas]:
        input = input.to(device)
        output.append(model(input).squeeze(0))
    output = torch.stack(output)
    eig_tenas = get_ntk_tenas_new(model, output).detach()

    print(f'Index {index} seed {seed} after TENAS')
    if torch.cuda.is_available():
        print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
        torch.cuda.reset_peak_memory_stats()

    return {
        'ntk_fro': torch.linalg.matrix_norm(ntk, ord='fro').item(),
        'ntk_nuc': torch.linalg.matrix_norm(ntk, ord='nuc').item(),
        'ntk_sing': torch.linalg.matrix_norm(ntk, ord=2).item(),
        'tenas_max': eig_tenas.max().item(),
        'tenas_sum': eig_tenas.sum().item(),
        'tenas_sum2': torch.square(eig_tenas).sum().item(),
        'tenas_std': eig_tenas.std().item(),
        'tenas_cond': eig_tenas.max().item() / eig_tenas.min().item(),
        'params_total': sum(p.numel() for n, p in model.named_parameters()),
        'params_used': sum(p.numel() for n, p in model.named_parameters() if ('weight' in n and 'bn' not in n)),
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FIM for NATS-bench', add_help=False)
    # Add arguments
    parser.add_argument('--dataset', type=str, default='Data/ImageNet16-120', help='Dataset path.')
    parser.add_argument('--models', type=str, default='NATS-tss-v1_0-3ffb9-full', help='Models path.')
    parser.add_argument('--epochs_trained', type=str, default='200', help='Number of training epochs.')
    parser.add_argument('--n_models', type=int, default=100, help='Number of models to assess.')
    parser.add_argument('--num-iterations', type=int, default=1, help='Number of iterations.')
    parser.add_argument('--use-train', action='store_true', help='Use train split, not test split.')
    parser.add_argument('--name-wandb', default='default_wandb_name', type=str, metavar='NAME',
                       help='Name of wandb experiment to be shown in the interface')
    parser.add_argument('--notes-wandb', default='', type=str, metavar='NAME',
                       help='Longer description of the run, like a -m commit message in git')
    parser.add_argument('--tags-wandb', default='default', type=str, metavar='NAME',
                       help='tags of the run')
    parser.add_argument('--num-fisher', default=4096, type=int, metavar='NAME',
                       help='Number of samples to be watched by FIM')
    parser.add_argument('--num-tenas', default=32, type=int, metavar='NAME',
                       help='Number of samples to be watched by TENAS')

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
    loader = DataLoader(ImageNet16(args.dataset, False, transforms.Compose([transforms.ToTensor()]), 120), batch_size=1, shuffle=False)
    api = create(args.models, 'tss', fast_mode=True, verbose=True)

    # Iterate models and seeds
    for index in range(1, args.n_models):
        try:
            for seed in api.get_net_param(index, args.dataset_name, None, hp=args.epochs_trained).keys():

                # get FIM and TENAS
                model = get_model(api, args.dataset_name, index, seed, args.epochs_trained).to(device)
                info_ntk = compute(model, index, seed, loader, args.num_fisher, args.num_tenas, device)

                # get train statistics
                info_per = api.get_more_info(index, args.dataset_name, hp=args.epochs_trained, is_random=seed)
                info_cost = api.get_cost_info(index, args.dataset_name, hp=args.epochs_trained)

                # log
                combined_dict = {'index': index, 'seed': seed, 'hp': args.epochs_trained}
                combined_dict.update(info_ntk)
                combined_dict.update(info_per)
                combined_dict.update(info_cost)
                wandb.log(combined_dict)
        except AssertionError as e:
            print(f"Model not found: {e}")
