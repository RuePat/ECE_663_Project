import torchvision
from torchvision.datasets import USPS
from torchvision.datasets import KMNIST
from torchvision import transforms
import torch.utils.data as data

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE : ",device)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((24,24)),
     transforms.Normalize((0.5,), (0.5,)),
     ])
usps_dataset = torchvision.datasets.USPS(root='./usps_24', download=True, transform=transform)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((16,16)),
     transforms.Normalize((0.5,), (0.5,)),
     ])
usps_dataset = torchvision.datasets.USPS(root='./usps_16', download=True, transform=transform)