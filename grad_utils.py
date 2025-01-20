import json
import os
from hashlib import md5
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm
# from trak.projectors import BasicProjector, CudaProjector, ProjectionType



def center_score(influence_scores):
    """ Center the influence scores. """
    max_score = influence_scores.max()
    min_score = influence_scores.min()
    center_score = (influence_scores - min_score) / (max_score - min_score + 1e-8)
    return center_score

def prepare_batch(batch, device=torch.device("cuda")):
    """ Move the batch to the device. """
    for key in batch:
        batch[key] = batch[key].to(device)



# def get_trak_projector(device: torch.device, verbose=True):
#     """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
#     try:
#         num_sms = torch.cuda.get_device_properties(
#             device.index).multi_processor_count
#         import fast_jl
#
#         # test run to catch at init time if projection goes through
#         fast_jl.project_rademacher_8(torch.zeros(
#             8, 1_000, device=device), 512, 0, num_sms)
#         projector = CudaProjector
#         # if verbose:
#         #     print("Using CudaProjector")
#     except:
#         projector = BasicProjector
#         # if verbose:
#         #     print("Using BasicProjector")
#     return projector


def get_number_of_params(model, verbose=True):
    num_params = sum([p.numel() 
                      for p in model.parameters() if p.requires_grad])
    if verbose:
        print(f"Total number of parameters that require gradients: {num_params}")
    return num_params


def obtain_gradients(model):
    """ obtain gradients. """

    vectorized_grads = torch.cat(
        [p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    return vectorized_grads


def obtain_gradients_with_adam(model, avg, avg_sq):
    """ obtain gradients with adam optimizer states. """
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08

    vectorized_grads = torch.cat(
        [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])

    updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
    updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads ** 2
    vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)

    return vectorized_grads


def prepare_optimizer_state(model, optimizer_state, device):
    names = [i for i, t in enumerate(model.named_parameters()) if t[1].requires_grad]
    avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in names])
    avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1)
                       for n in names])
    avg = avg.to(device)
    avg_sq = avg_sq.to(device)
    return avg, avg_sq


def collect_train_grads(dataloader,
                  model,
                  accelerator,
                  proj_dim=8192,
                  adam_optimizer_state=None,):


    model_id = 0  # model_id is used to draft the random seed for the projectors
    block_size = 128  # fixed block size for the projectors
    projector_batch_size = 16  # batch size for the projectors

    verbose = accelerator.is_main_process

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype


    assert adam_optimizer_state is not None
    # first and second moment estimates
    m, v = prepare_optimizer_state(model, adam_optimizer_state, device)

    # projector = get_trak_projector(device, verbose)
    # number_of_params = get_number_of_params(model, verbose)
    #
    # proj = projector(grad_dim=number_of_params,
    #                  proj_dim=proj_dim,
    #                  seed=0,
    #                  proj_type=ProjectionType.rademacher,
    #                  device=device,
    #                  dtype=dtype,
    #                  block_size=block_size,
    #                  max_batch_size=projector_batch_size)

    model.zero_grad()
    total_steps = len(dataloader)
    for batch in tqdm(dataloader, total=total_steps, disable=not verbose):
        prepare_batch(batch, device)
        loss = model(batch).loss
        loss = loss / total_steps
        accelerator.backward(loss)

    accelerator.wait_for_everyone()
    vectorized_grads = obtain_gradients_with_adam(model, m, v)
    vectorized_grads = vectorized_grads.unsqueeze(0)
    # projected_grads = proj.project(vectorized_grads, model_id=model_id)
    projected_grads = vectorized_grads


    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()


    return projected_grads


def collect_valid_grads(dataloader,
                        model,
                        accelerator,
                        proj_dim=8192,):
    model_id = 0  # model_id is used to draft the random seed for the projectors
    block_size = 128  # fixed block size for the projectors
    projector_batch_size = 16  # batch size for the projectors

    verbose = accelerator.is_main_process

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # projector = get_trak_projector(device, verbose)
    # number_of_params = get_number_of_params(model, verbose)
    #
    # proj = projector(grad_dim=number_of_params,
    #                  proj_dim=proj_dim,
    #                  seed=0,
    #                  proj_type=ProjectionType.rademacher,
    #                  device=device,
    #                  dtype=dtype,
    #                  block_size=block_size,
    #                  max_batch_size=projector_batch_size)

    model.zero_grad()
    total_steps = len(dataloader)

    for batch in tqdm(dataloader, total=total_steps, disable=not verbose):
        prepare_batch(batch, device)
        loss = model(batch).loss
        loss = loss / total_steps
        accelerator.backward(loss)

    accelerator.wait_for_everyone()
    vectorized_grads = obtain_gradients(model)
    vectorized_grads = vectorized_grads.unsqueeze(0)
    # projected_grads = proj.project(vectorized_grads, model_id=model_id)
    projected_grads = vectorized_grads

    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()

    return projected_grads



def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):
    """Calculate the influence score.

    Args:
        training_info (torch.Tensor): training info (gradients/representations) stored in a tensor of shape N x N_DIM
        validation_info (torch.Tensor): validation info (gradients/representations) stored in a tensor of shape N_VALID x N_DIM
    """
    # training_info = normalize(training_info, dim=1)
    # validation_info = normalize(validation_info, dim=1)
    # N x N_VALID
    influence_scores = torch.matmul(
        training_info, validation_info.transpose(0, 1))
    return influence_scores
