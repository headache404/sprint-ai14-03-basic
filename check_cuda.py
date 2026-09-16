import torch

# nvidia-smi

print("torch:", torch.__version__)
print("CUDA 사용 가능:", torch.cuda.is_available())

print(torch.cuda.is_available())      # True면 사용 가능
print(torch.cuda.get_device_name(0))  # GPU 이름 출력
print(torch.version.cuda)             # PyTorch가 인식한 CUDA 버전