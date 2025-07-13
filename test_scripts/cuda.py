import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_capability())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))