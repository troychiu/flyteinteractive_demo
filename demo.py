import torch
from torch import nn
from torch.optim import SGD
from flytekit import task, workflow
from flytekitplugins.flyteinteractive import vscode

from flytekit.core.resources import Resources

def generate_data():
    torch.manual_seed(0)
    x = torch.rand(100, 1) * 10
    y = 2 * x + 1 + torch.randn(100, 1) * 2 
    return x, y

def train_model(x: torch.Tensor, y: torch.Tensor) -> nn.Module:
    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    num_epochs = 100
    for _ in range(num_epochs):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model

def calculate_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    criterion = nn.MSELoss()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    return loss.item()

@task
def get_data() -> (torch.Tensor, torch.Tensor):
    x, y = generate_data()
    return x, y

@task(
    requests=Resources(cpu="2000m", mem="2000Mi")
)

@vscode
def train_model_and_calculate_loss(x: torch.Tensor, y: torch.Tensor) -> float:
    model = train_model(x, y)
    loss = calculate_loss(model, x, y)
    return loss

@workflow
def training_workflow() -> float:
    x, y = get_data()
    loss = train_model_and_calculate_loss(x=x, y=y)
    return loss

