import torch

print(torch.cuda.is_available())  # Moet True teruggeven als CUDA correct is geïnstalleerd

print(torch.cuda.get_device_name(0))  # Geeft de naam van de GPU terug
