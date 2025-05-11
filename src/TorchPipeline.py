import os
import warnings

import torch
import numpy as np
from typing import List, Tuple

from PIL import Image
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm import tqdm

from BasePipeline import BasePipeline


class ImageDataset(Dataset):
    """
    Dataset for loading images from filepaths with labels.

    Args:
        paths: list of image file paths
        labels: numpy array of integer labels
        transform: torchvision transform to apply
    """

    def __init__(self, paths: List[str], labels: np.ndarray, transform: transforms.Compose) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), int(self.labels[idx])


class TorchPipeline(BasePipeline):
    """
    PyTorch pipeline for training and evaluating a ResNet-18 model.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

        self.transform_train = transforms.Compose([
            transforms.Resize(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
        ])
        self.transform_eval = transforms.Compose([
            transforms.Resize(size=230),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Pre-trained backbone (no head yet)
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None

    def _lr_finder(self, loader: DataLoader, init_value: float = 1e-7, final_value: float = 0.2,
                   beta: float = 0.98, return_all: bool = False) -> float | Tuple[List[float], List[float], float]:
        """
        Find an optimal learning rate using an empirical LR range test.

        Returns:
            If `return_all` is True, returns (lrs, losses, suggested_lr),
            otherwise just suggested_lr.
        """
        model = self.model
        optimizer = optim.SGD(model.parameters(), lr=init_value)
        num = len(loader) - 1
        mult = (final_value / init_value) ** (1 / num)
        avg_loss, best_loss = 0.0, float('inf')
        lrs, losses = [], []

        checkpoint = model.state_dict()
        model.train()

        for inputs, labels in tqdm(loader, desc="LR finder", unit="batch"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            loss = self.criterion(model(inputs), labels)
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smooth = avg_loss / (1 - beta ** (len(lrs) + 1))
            if smooth < best_loss:
                best_loss = smooth
            if smooth > 10 * best_loss:
                break
            loss.backward()
            optimizer.step()
            lrs.append(init_value * (mult ** len(lrs)))
            losses.append(smooth)
            for pg in optimizer.param_groups:
                pg['lr'] *= mult

        model.load_state_dict(checkpoint)
        grads = np.gradient(np.array(losses), np.log10(np.array(lrs)))
        idx = int(np.argmin(grads[40:-5]) + 40)

        suggested = lrs[idx]

        if return_all:
            return lrs, losses, suggested
        return suggested

    def find_learning_rate(self, paths: List[str], labels: np.ndarray) -> Tuple[List[float], List[float], float]:
        """
        Runs the LR finder on (paths, labels) with train transforms,
        returning (lrs, losses, suggested_lr).
        """

        ds = ImageDataset(paths, labels, transform=self.transform_train)
        loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=min(os.cpu_count(), 4))

        return self._lr_finder(loader, return_all=True)

    def fit(self, paths: List[str], labels: np.ndarray, lr: float = None) -> None:
        """
        Train the model: freeze backbone first two epochs, then fine-tune all layers.
        """
        # Reset final layer to match number of classes
        n_classes = int(np.unique(labels).shape[0])
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=n_classes).to(self.device)

        train_ds = ImageDataset(paths, labels, transform=self.transform_train)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=min(os.cpu_count(), 4))

        lr = lr or self._lr_finder(train_loader)

        # freeze backbone
        backbone_params = []
        head_params = []
        for name, p in self.model.named_parameters():
            if name.startswith("fc."):
                p.requires_grad = True
                head_params.append(p)
            else:
                p.requires_grad = False
                backbone_params.append(p)

        print("Head params:", len(head_params), "Backbone params:", len(backbone_params))

        self.optimizer = optim.SGD(head_params, lr=lr, momentum=0.9, weight_decay=1e-4)

        for epoch in range(1, 6):
            if epoch == 3:
                # unfreeze all
                for param in self.model.parameters():
                    param.requires_grad = True
                self.optimizer = optim.SGD(self.model.parameters(), lr=lr * 0.1, momentum=0.9, weight_decay=1e-4)
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            processed = 0
            all_preds, all_trues = [], []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
            for inputs, lbls in pbar:
                inputs, lbls = inputs.to(self.device), lbls.to(self.device)
                self.optimizer.zero_grad()
                outs = self.model(inputs)
                loss = self.criterion(outs, lbls)
                loss.backward()
                self.optimizer.step()

                # stats
                bs = inputs.size(0)
                running_loss += loss.item() * bs
                preds = outs.argmax(1).cpu().numpy()
                trues = lbls.cpu().numpy()
                running_corrects += (preds == trues).sum()
                processed += bs
                all_preds.extend(preds.tolist())
                all_trues.extend(trues.tolist())

                avg_loss = running_loss / processed
                avg_acc = running_corrects / processed
                avg_f1 = f1_score(all_trues, all_preds, average='macro')
                pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}", f1=f"{avg_f1:.4f}")

        self._is_fitted = True

    def predict(self, paths: List[str]) -> np.ndarray:
        if not self.is_fitted: warnings.warn("Not fitted yet")

        eval_ds = ImageDataset(paths, np.zeros(len(paths)), transform=self.transform_eval)
        eval_loader = DataLoader(eval_ds, batch_size=32, shuffle=False, num_workers=min(os.cpu_count(), 4))

        all_preds = []
        self.model.eval()
        with torch.no_grad():
            for inputs, _ in tqdm(eval_loader, desc="Predict", unit="batch"):
                outs = self.model(inputs.to(self.device))
                all_preds.extend(outs.argmax(1).cpu().numpy().tolist())

        return np.array(all_preds)

    def save(self, path: str) -> None:
        torch.save(self.model, path)

    def load(self, path: str) -> 'TorchPipeline':
        self.model = torch.load(path, map_location=self.device, weights_only=False)
        self.model.eval()
        self._is_fitted = True
        return self
