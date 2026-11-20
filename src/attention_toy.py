import torch

x = torch.tensor([[1.0, 0.0, 1.0, 0.0],      # token 1
                  [0.0, 1.0, 0.0, 1.0],      # token 2
                  [1.0, 0.0, 1.0, 1.0]])     # token 3

d = x.shape[1]                                # 4 numbers per token
scores = x @ x.T / (d ** 0.5)                 # how much each token likes each other token
weights = torch.softmax(scores, dim=-1)       # turn scores into percentages (rows add up to 1)
out = weights @ x                             # each token becomes a mix of all tokens

print("scores:\n", scores)
print("weights:\n", weights)
print("row sums:", weights.sum(dim=1))
print("output:\n", out)