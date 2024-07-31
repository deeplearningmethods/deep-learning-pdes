import gc
import torch

def get_all_active_pytorch_objects():
    # Collect all PyTorch objects that are currently tracked by the garbage collector
    pytorch_objects = [obj for obj in gc.get_objects() if isinstance(obj, (torch.nn.Module, torch.Tensor))]
    return pytorch_objects

def get_memory_usage(obj):
    def calculate_memory(obj):
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            return obj.element_size() * obj.nelement()
        return 0

    # Calculate memory usage recursively if obj is a container (e.g., nn.Module)
    if isinstance(obj, (list, tuple, set)):
        return sum(get_memory_usage(item) for item in obj)
    elif isinstance(obj, dict):
        return sum(get_memory_usage(item) for item in obj.values())
    elif isinstance(obj, torch.nn.Module):
        # Calculate memory usage for model parameters and buffers only once
        param_buffer_memory = sum(p.data.element_size() * p.data.nelement() for p in obj.parameters())
        param_buffer_memory += sum(b.data.element_size() * b.data.nelement() for b in obj.buffers())
        return param_buffer_memory

    return calculate_memory(obj)

def tensor_summary(tensor):
    shape = tuple(tensor.size())
    dtype = tensor.dtype
    description = f"Tensor - Shape: {shape}, Dtype: {dtype}"
    return description

def estimate_total_memory_usage(pytorch_objects):
    processed_objects = set()
    total_memory_usage = 0

    for obj in pytorch_objects:
        if obj not in processed_objects:
            memory_usage = get_memory_usage(obj)
            total_memory_usage += memory_usage
            processed_objects.add(obj)

    return total_memory_usage

if __name__ == "__main__":
    # Create some PyTorch objects
    tensor1 = torch.randn(100, 100).cuda()
    tensor2 = torch.randn(100, 100).cuda()
    tensor3 = torch.randn(100, 100).cuda()
    tensor4 = torch.randn(100, 100).cuda()

    model = torch.nn.Sequential(
        torch.nn.Linear(100, 100).cuda(),
        torch.nn.Linear(100, 100).cuda(),
        torch.nn.Linear(100, 100).cuda(),
        torch.nn.Linear(100, 100).cuda(),
    )

    active_pytorch_objects = get_all_active_pytorch_objects()
    total_memory_usage = estimate_total_memory_usage(active_pytorch_objects)
    print(f"Estimated total GPU memory usage: {total_memory_usage / (1024**2):.2f} MB")

    # Print a list of tensors and models with their summaries and memory usages
    print("List of active PyTorch objects:")
    for obj in active_pytorch_objects:
        memory_usage = get_memory_usage(obj)
        if memory_usage > 0:
            obj_summary = tensor_summary(obj) if isinstance(obj, torch.Tensor) else f"{type(obj).__name__}"
            print(f"{obj_summary} - Memory Usage: {memory_usage / (1024**2):.2f} MB")





