import torch 
t = torch.tensor([[1, 2], [3, 4]])

v = t.gather(1, torch.tensor([[0], [0]]))

print(v)