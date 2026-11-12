import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)

# data: 200 points of sin(x), shaped as a column (200, 1)
x = torch.linspace(0, 2 * torch.pi, 200).unsqueeze(1)
y_true = torch.sin(x)

# network: 1 input -> 32 -> 32 -> 1 output
model = nn.Sequential(
    nn.Linear(1, 32), nn.Tanh(),
    nn.Linear(32, 32), nn.Tanh(),
    nn.Linear(32, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(5000):
    optimizer.zero_grad()            # clear old gradients
    y_pred = model(x)                # network's guess
    loss = ((y_pred - y_true) ** 2).mean()
    loss.backward()                  # autograd: how should each weight change?
    optimizer.step()                 # nudge the weights
    if step % 500 == 0:
        print(step, loss.item())

print("final loss:", loss.item())

plt.plot(x, y_true, label="sin x")
plt.plot(x, y_pred.detach(), "--", label="network")
plt.legend()
plt.savefig("figures/fit_sine.png")
xg = x.clone().requires_grad_(True)
u = model(xg)
du = torch.autograd.grad(u, xg, torch.ones_like(u), create_graph=True)[0]
d2u = torch.autograd.grad(du, xg, torch.ones_like(du), create_graph=True)[0]
print("max |d2u + sin x|:", (d2u + torch.sin(x)).abs().max().item())