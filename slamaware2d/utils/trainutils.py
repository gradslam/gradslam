import os

import numpy as np
from colorama import Fore, Style

import torch
from globalmap import GlobalMap
from weights import load_weights


def pose_to_quaternion_fmt(pose):
    # print (pose)
    quat = quaternion_from_matrix(pose)[np.array([1, 2, 3, 0])]
    tr = pose[:3, 3]

    return np.hstack((tr, quat))


def save_checkpoint(model, optimizer, epoch, miou, args):
    """Saves the model in a specified directory with a specified name.save

    Keyword arguments:
    - model (``nn.Module``): The model to save.
    - optimizer (``torch.optim``): The optimizer state to save.
    - epoch (``int``): The current epoch for the model.
    - miou (``float``): The mean IoU obtained by the model.
    - args (``ArgumentParser``): An instance of ArgumentParser which contains
    the arguments used to train ``model``. The arguments are written to a text
    file in ``args.save_dir`` named "``args.name``_args.txt".

    """
    name = args.name
    save_dir = args.save_dir

    assert os.path.isdir(save_dir), 'The directory "{0}" doesn\'t exist.'.format(
        save_dir
    )

    # Save model
    model_path = os.path.join(save_dir, name)
    checkpoint = {
        "epoch": epoch,
        "miou": miou,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, model_path)

    # Save arguments
    summary_filename = os.path.join(save_dir, name + "_summary.txt")
    with open(summary_filename, "w") as summary_file:
        sorted_args = sorted(vars(args))
        summary_file.write("ARGUMENTS\n")
        for arg in sorted_args:
            arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
            summary_file.write(arg_str)

        summary_file.write("\nBEST VALIDATION\n")
        summary_file.write("Epoch: {0}\n".format(epoch))
        summary_file.write("Mean IoU: {0}\n".format(miou))


def load_enet_checkpoint(model, model_path, verbose=True):
    """Loads enet from the specified checkpoint path

    Keyword arguments:
    - model (``nn.Module``): The stored model state is copied to this model
    instance.
    - model_path (``string``): The path to the saved model checkpoint.
    - verbose (``bool``): 

    Returns:
    The epoch, mean IoU, ``model``, and ``optimizer`` loaded from the
    checkpoint.

    """
    folder_dir = os.path.dirname(model_path)
    filename = os.path.basename(model_path)
    assert os.path.isdir(folder_dir), 'The directory "{0}" doesn\'t exist.'.format(
        folder_dir
    )
    assert os.path.isfile(model_path), 'The model file "{0}" doesn\'t exist.'.format(
        filename
    )

    # Load the stored model parameters to the model instance
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]
    miou = checkpoint["miou"]

    if verbose:
        print(f"-{Fore.GREEN}Pretrained ENet loaded{Style.RESET_ALL}")

    return model, optimizer, epoch, miou


def load_fcrn_weights(model, weights_file_path, dtype, use_bn, verbose=True):
    """Loads fcrn from the specified weights file path

    Keyword arguments:
    - model (``nn.Module``): The stored model state is copied to this model
    instance.
    - model_path (``string``): The path to the saved model weights.
    - dtype
    - verbose (``bool``): 

    Returns:
    The epoch, mean IoU, ``model``, and ``optimizer`` loaded from the
    checkpoint.

    """
    # Load the stored model weights to the model instance
    if os.path.isfile(weights_file_path):
        model.load_state_dict(load_weights(model, weights_file_path, dtype, use_bn))
        if verbose:
            print(f"Pretrained FCRN: {Fore.GREEN}Loaded{Style.RESET_ALL}")
    elif verbose:
        print(f"Pretrained FCRN: {Fore.RED}Not Loaded{Style.RESET_ALL}")

    return model


def normalize_grads(gradients, method="abs", per_frame=True, use_log=False, eps=1e-25):
    r"""Normalizes gradients to be in [0-1] range

    Args:
        gradients (torch.Tensor): Gradient tensor
                                  (of shape: batch_size x seq_len x C x H x W)
        method (str, Optional): Method for performing the normalization.
                                Options:
                                'abs' -> take absolute value
                                'shift' -> shift min to 0
        per_frame (bool, Optional): Do the normalization per frame or per
                                    entire sequence.
        use_log (bool, Optional): Get logarithm of gradients after
                                  normalization to have larger gradient
                                  scale (e.g. for visualization).
        eps (float, Optional): Epsilon for numerical stability.
    """
    assert (method == "abs") or (method == "shift"), "invalid method"
    assert not (use_log and (method == "shift")), (
        "use_log intended for 'abs'" " method only"
    )
    if torch.sum(torch.abs(gradients)) < 1e-35:
        return torch.abs(gradients) if method == "abs" else (gradients + 0.5)

    abs_gradients = torch.abs(gradients)
    if not per_frame:
        max_abs_val = torch.max(abs_gradients)
    else:
        max_abs_val = torch.max(torch.max(torch.max(abs_gradients, 2)[0], 2)[0], 2)[0]
        max_abs_val = max_abs_val.unsqueeze(2).unsqueeze(3).unsqueeze(4)

    if method == "abs":
        gradients = abs_gradients / (max_abs_val + eps)
        if use_log:
            gradients = torch.log2(1 + gradients)
        assert torch.max(gradients) == 1.0, "" "gradients were not scaled correctly"

    elif method == "shift":
        gradients = gradients / ((2 * max_abs_val) + eps)
        gradients += 0.5
        assert (torch.max(gradients) == 1.0) or (torch.min(gradients) == 0), (
            "" "gradients were not scaled correctly"
        )
    return gradients


def predict_depth(model, color_batch, depth_batch, pred_last_frame_only):
    if pred_last_frame_only:
        gt_first_depths = depth_batch[:, :-1]
        pred_final_depth = model(color_batch[:, -1]).unsqueeze(1)
        pred_depth_batch = torch.cat([gt_first_depths, pred_final_depth], 1)

        zeros_mask = torch.zeros_like(gt_first_depths)
        ones_mask = torch.ones_like(pred_final_depth)
        pred_mask = torch.cat([zeros_mask, ones_mask], 1).type(torch.bool)
    else:
        N, S, C, H, W = color_batch.shape
        color_batch_flat = color_batch.view(-1, C, H, W).contiguous()
        pred_depth_batch_flat = model(color_batch_flat)
        pred_depth_batch = pred_depth_batch_flat.view(N, S, 1, H, W).contiguous()
        pred_mask = torch.ones_like(pred_depth_batch).type(torch.bool)

    # remove missing values
    known_values_mask = depth_batch != 0
    pred_mask = pred_mask & known_values_mask
    pred_depth_batch = pred_depth_batch * known_values_mask.float()

    return pred_depth_batch, pred_mask


def pointfusion(
    frames,
    transform_batch,
    dist2_th,
    dot_th,
    use_GT_pose=True,
    use_grad_LM=False,
    pred_last_frame_only=False,
    *,
    ds_ratio=4,
    accumulateMap=False,
):
    device = frames[0][0].color_map.device
    global_map_list = []
    seq_len = len(frames)
    batch_size = len(frames[0])
    for t in range(seq_len):
        # TODO: Remove second for loop by vectorizing most ICP/fusion stuff
        for n in range(batch_size):
            if t == 0:
                global_map = GlobalMap(init_frame=frames[0][n])
                global_map_list.append(global_map)
            else:
                # get current frame and global map
                new_frame = frames[t][n]
                global_map = global_map_list[n]

                # get pose transform
                if use_GT_pose or (pred_last_frame_only and t < (seq_len - 1)):
                    transform = transform_batch[n, t]
                else:
                    initial_transform = torch.eye(4).to(device)
                    transform, correspondence = ICP(
                        global_map, new_frame, initial_transform, use_grad_LM, ds_ratio,
                    )

                    # def diff(x, y):
                    #     return torch.mean((x - y) ** 2) / torch.mean(y ** 2)
                    # print("ICP transform loss t{}: {}".format(t, diff(transform, transform_batch[0, t])))

                # update global map
                if not accumulateMap:
                    global_map.updateMapFusion(
                        new_frame, transform.to(device), dist2_th, dot_th
                    )
                else:
                    global_map.updateMapAccumulate(new_frame, transform.to(device))

    return global_map_list
