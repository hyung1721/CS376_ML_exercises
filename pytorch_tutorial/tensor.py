import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device="cuda", requires_grad=True)

print(x)
print(x.device)