import torch
import matplotlib.pyplot as plt

x = torch.linspace(0, 2 * torch.pi, 200, requires_grad=True)
y = torch.sin(x)

# first derivative
dy = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

# second derivative: same pattern, but differentiate dy instead of y
d2y = torch.autograd.grad(dy, x, grad_outputs=torch.ones_like(dy), create_graph=True)[0]

err1 = (dy - torch.cos(x)).abs().max()
err2 = (d2y - (-torch.sin(x))).abs().max()
print(f"max error dy: {err1:.2e}, d2y: {err2:.2e}")

plt.plot(x.detach(), y.detach(), label="sin x")
plt.plot(x.detach(), dy.detach(), label="autograd dy/dx")
plt.plot(x.detach(), d2y.detach(), label="autograd d2y/dx2")
plt.legend()
plt.savefig("figures/autograd_check.png")