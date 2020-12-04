import torch

from .reps import reps_dual


def test_reps_dual_derivative():
    eta = torch.tensor(0.5, requires_grad=True)
    theta = torch.tensor([2.0, 1.0, 3.0], requires_grad=True)
    features = torch.tensor([[1.0, 2.0, 4.0], [1.0, 2.0, 5.0], [1.0, 2.0, 6.0]])
    features_next = torch.tensor([[2.0, 3.0, 2.0], [2.0, 3.0, 1.0], [2.0, 3.0, 3.0]])
    rewards = torch.tensor([0.0, 2.0, 0.5])
    epsilon = torch.tensor(1e-5, dtype=torch.get_default_dtype())

    dual_f, d_eta, d_theta = reps_dual(
        eta, theta, features, features_next, rewards, epsilon
    )
    dual_f.backward()
    eta_pytorch_grad = eta.grad
    theta_pytorch_grad = theta.grad
    assert torch.equal(d_theta, theta_pytorch_grad)
    assert torch.equal(d_eta, eta_pytorch_grad)
