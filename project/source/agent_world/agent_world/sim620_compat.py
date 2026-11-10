"""Zero-copy boundary from Lab 3 ProxyArray/Warp data to the original Torch controller."""
def torch_data(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "torch"):
        return value.torch
    import warp as wp
    if isinstance(value, wp.array):
        return wp.to_torch(value)
    return value
