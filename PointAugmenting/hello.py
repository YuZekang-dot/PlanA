import torch
checkpoint = torch.load('./output/pp/v1/checkpoint_epoch_15.pth', map_location="cpu")
print(checkpoint)

