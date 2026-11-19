"""Carga los recortes de data/pieces/<split>/<clase>/ con aumentos de datos.

Los índices de clase siguen el orden de config.CLASSES (no el alfabético de
ImageFolder), que es el mismo que se exporta a pieces.json para la app.
"""
from collections import Counter

import cv2
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2

from config import (BATCH_SIZE, CLASSES, DATA_DIR, INPUT_H, INPUT_W, MEAN,
                    NUM_WORKERS, STD)

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
PIN = torch.cuda.is_available()


def build_transforms(train: bool) -> v2.Compose:
    base = [v2.Resize((INPUT_H, INPUT_W), antialias=True)]
    if train:
        base += [
            # Pequeños errores de homografía / encuadre
            v2.RandomAffine(degrees=5, translate=(0.05, 0.03), scale=(0.95, 1.05)),
            # Iluminación. Sin cambios de tono: no confundir blancas con negras
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
            # Sin volteo vertical (la pieza siempre "sube"); horizontal sí es válido
            v2.RandomHorizontalFlip(p=0.5),
        ]
    base += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
    return v2.Compose(base)


class PiecesDataset(Dataset):
    def __init__(self, split: str, root=DATA_DIR, train: bool | None = None):
        self.root = root / split
        if not self.root.exists():
            raise FileNotFoundError(f"No existe {self.root}. Ejecuta antes prepare_data.py")
        self.samples = [(p, CLASS_TO_IDX[c])
                        for c in CLASSES
                        for p in sorted((self.root / c).glob("*.png"))]
        if not self.samples:
            raise RuntimeError(f"{self.root} está vacío")
        self.transform = build_transforms(split == "train" if train is None else train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # la app debe hacer lo mismo
        x = torch.from_numpy(img).permute(2, 0, 1)          # HWC uint8 -> CHW
        return self.transform(x), label

    def class_counts(self) -> list[int]:
        cnt = Counter(label for _, label in self.samples)
        return [cnt.get(i, 0) for i in range(len(CLASSES))]


def class_weights(ds: PiecesDataset) -> torch.Tensor:
    """Pesos inversos a la frecuencia, para CrossEntropyLoss(weight=...)."""
    counts = torch.tensor(ds.class_counts(), dtype=torch.float32).clamp(min=1)
    w = counts.sum() / (len(counts) * counts)
    return w / w.mean()


def make_loaders(balanced: bool = True, batch_size: int = BATCH_SIZE,
                 num_workers: int = NUM_WORKERS):
    """Devuelve (train_loader, val_loader, train_dataset).

    balanced=True usa un sampler que muestrea todas las clases por igual
    (alternativa a los pesos en la pérdida; no usar ambos a la vez).
    """
    train_ds, val_ds = PiecesDataset("train"), PiecesDataset("val")

    if balanced:
        counts = train_ds.class_counts()
        sample_w = [1.0 / counts[label] for _, label in train_ds.samples]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        train_dl = DataLoader(train_ds, batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=PIN)
    else:
        train_dl = DataLoader(train_ds, batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=PIN)

    val_dl = DataLoader(val_ds, batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=PIN)
    return train_dl, val_dl, train_ds


if __name__ == "__main__":
    # Comprobación rápida: python training/dataset.py
    for split in ("train", "val", "test"):
        ds = PiecesDataset(split)
        print(f"{split:<6} {len(ds):>7} recortes  " +
              " ".join(f"{c}={n}" for c, n in zip(CLASSES, ds.class_counts())))
    x, y = PiecesDataset("train")[0]
    print("Muestra:", tuple(x.shape), x.dtype, "clase:", CLASSES[y])