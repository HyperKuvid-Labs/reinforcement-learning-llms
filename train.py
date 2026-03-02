import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
import torch.amp

from advantages import compute_grpo, compute_reinforce, compute_maxrl

batch_size = 128
num_epochs = 5
k = 4
learning_rate = 0.1
num_classes = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = True

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

data_root = './data'

train_dataset = datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_transform)
val_dataset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

def sample_rollouts(logits, K):
    probs = torch.softmax(logits, dim=1)
    return torch.multinomial(probs, num_samples=K, replacement=True)

def rl_loss(inputs, labels, model, advantage_fn, K):
    with torch.amp.autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
        logits = model(inputs)
        y_samples = sample_rollouts(logits, K)

        rewards = (y_samples == labels.view(-1, 1)).float()
        advantages = advantage_fn(rewards)

        log_probs = torch.log_softmax(logits, dim=1).gather(1, y_samples)
        policy_loss = -(log_probs * advantages).mean()
    return policy_loss

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
                preds = model(inputs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

advantage_fns = {
    'grpo': compute_grpo,
    'reinforce': compute_reinforce,
    'maxrl': compute_maxrl,
}


def train_experiment(adv_name, advantage_fn, callbacks=None):
    cb = callbacks or {}
    _scaler = torch.amp.GradScaler(enabled=use_amp)
    writer = SummaryWriter(log_dir=f'runs/cifar100_{adv_name}')

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = rl_loss(inputs, labels, model, advantage_fn, k)

            _scaler.scale(loss).backward()
            _scaler.step(optimizer)
            _scaler.update()

            epoch_loss += loss.item()
            writer.add_scalar('loss/step', loss.item(), global_step)
            global_step += 1

            if 'on_batch' in cb:
                cb['on_batch'](batch_idx, len(train_loader), loss.item())

        scheduler.step()
        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar('loss/epoch', avg_epoch_loss, epoch)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

        val_acc = evaluate(model, val_loader)
        writer.add_scalar('acc/val', val_acc, epoch)

        if 'on_epoch' in cb:
            cb['on_epoch'](epoch, num_epochs, avg_epoch_loss, val_acc, scheduler.get_last_lr()[0])

    writer.close()
    if 'on_done' in cb:
        cb['on_done'](adv_name)


if __name__ == '__main__':
    for adv_name, advantage_fn in advantage_fns.items():
        print(f"\nStarting training with advantage: {adv_name.upper()}")
        train_experiment(adv_name, advantage_fn)

