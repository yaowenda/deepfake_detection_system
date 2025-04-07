import torch

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)