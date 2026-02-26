import torch
model = torch.load(r"C:\Users\vigne\Downloads\archive26feb\pretrained_model\pretrained_model\model_100_epoch.pth", map_location="cpu")
print(type(model))
