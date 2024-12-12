import torch, gc

"""To clear my CUDA cache and release the memory used from training my model"""
torch.cuda.empty_cache()
gc.collect()