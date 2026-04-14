import torch

print(f"Версия PyTorch: {torch.__version__}")

cuda_available = torch.cuda.is_available()
print(f"CUDA: {cuda_available}")

if cuda_available:
    gpu_count = torch.cuda.device_count()
    print(f"Количество GPU: {gpu_count}")

    gpu_name = torch.cuda.get_device_name(0)
    print(f"Имя GPU: {gpu_name}")
    
    cuda_version = torch.version.cuda
    print(f"Версия CUDA (PyTorch): {cuda_version}")
    
    x = torch.randn(3, 3).cuda()
    y = torch.randn(3, 3).cuda()
    
    print(f"Тензор x на устройстве: {x.device}")
    print(f"Тензор y на устройстве: {y.device}")
    
    z = torch.matmul(x, y)
    print(f"Результат умножения: {z.device}")
    
    result = (x + y).mean()
    print(f"Среднее значение x + y: {result.item():.4f}")
else:
    print("CUDA НЕ доступна.")