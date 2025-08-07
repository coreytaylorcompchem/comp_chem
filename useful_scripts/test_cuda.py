import torch

def check_cuda():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

        # Test tensor operation on GPU
        x = torch.rand(3, 3).to('cuda')
        y = torch.rand(3, 3).to('cuda')
        z = x + y
        print("Tensor operation successful on GPU:\n", z)
    else:
        print("CUDA is not available. Check your installation.")

if __name__ == "__main__":
    check_cuda()
