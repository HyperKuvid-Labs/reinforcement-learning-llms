import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter

from advantages import compute_grpo, compute_reinforce, compute_maxrl

batch_size = 256
num_epochs = 20
k = 10
learning_rate = 1e-3
num_classes = 101
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_root = './data'

train_dataset = datasets.Food101(
    root=data_root,
    split='train',
    download=True,
    transform=train_transform
)

val_dataset = datasets.Food101(
    root=data_root,
    split='test',
    download=True,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

def sample_rollouts(logits, K):
    probs = torch.softmax(logits, dim=1)
    samples = torch.multinomial(probs, num_samples=K, replacement=True)
    return samples

def rl_loss(inputs, labels, model, advantage_fn, K):
    logits = model(inputs)
    losses = []
    for i in range(inputs.size(0)):
        logit = logits[i].unsqueeze(0)
        y_star = labels[i]
        y_samples = sample_rollouts(logit, K)

        rewards = (y_samples == y_star).float()
        advantages = advantage_fn(rewards)

        log_probs = torch.log_softmax(logit, dim=1).gather(1, y_samples).squeeze()
        loss_term = -torch.mean(log_probs * advantages)
        losses.append(loss_term)
    return torch.mean(torch.stack(losses))

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

advantage_fns = {
    'grpo': compute_grpo,
    'reinforce': compute_reinforce,
    'maxrl': compute_maxrl,
}

for adv_name, advantage_fn in advantage_fns.items():
    writer = SummaryWriter(log_dir=f'runs/food101_{adv_name}')

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = rl_loss(inputs, labels, model, advantage_fn, k)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            writer.add_scalar('loss/step', loss.item(), global_step)
            global_step += 1

        scheduler.step()
        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar('loss/epoch', avg_epoch_loss, epoch)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

        val_acc = evaluate(model, val_loader)
        writer.add_scalar('acc/val', val_acc, epoch)

    writer.close()

