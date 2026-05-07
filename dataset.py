from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class PlantVillageDataset(Dataset):
    def __init__(self, split_file, data_dir, class_to_idx, transform=None):
        self.data_dir     = Path(data_dir)
        self.transform    = transform
        self.class_to_idx = class_to_idx
        self.samples      = []
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('/')
                class_name = parts[2]
                label = class_to_idx[class_name]
                self.samples.append((line, label))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        img = Image.open(self.data_dir / rel_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
