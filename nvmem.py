import torch
import gc


def printMemoryUsed(abbr=False):
    print()
    print(torch.cuda.memory_summary(device=torch.cuda.current_device(), abbreviated=True))
    print()

def printInfoCUDA():
    print()
    print("CUDA version: ", torch.version.cuda)
    print("CUDA available: ", torch.cuda.is_available())
    print("Available GPUs: ", torch.cuda.device_count())
    print("Current device: ", torch.cuda.current_device())
    print()

def clearMemory():
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    printInfoCUDA()
    printMemoryUsed(True)


