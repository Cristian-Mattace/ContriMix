import torch

def update_average(prev_avg, prev_counts, curr_avg, curr_counts):
    denom = prev_counts + curr_counts
    if isinstance(curr_counts, torch.Tensor):
        denom += (denom==0).float()
    elif isinstance(curr_counts, int) or isinstance(curr_counts, float):
        if denom==0:
            return 0.
    else:
        raise ValueError('Type of curr_counts not recognized')
    prev_weight = prev_counts/denom
    curr_weight = curr_counts/denom
    return prev_weight*prev_avg + curr_weight*curr_avg

def move_to(obj, device):
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to(v, device) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        # Assume obj is a Tensor or other type
        # (like Batch, for MolPCBA) that supports .to(device)
        return obj.to(device)

def detach_and_clone(obj):
    if torch.is_tensor(obj):
        return obj.detach().clone()
    elif isinstance(obj, dict):
        return {k: detach_and_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_and_clone(v) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        raise TypeError("Invalid type for detach_and_clone")