# =============================================================================
# The Green Grid — Classificacao de Doencas Vegetais
# TI6 / Grupo 12 / 2026.1
#
# Uso:
#   1 no  (local):   python train.py
#   2 nos (cluster): inicie o head com `ray start --head`
#                    conecte o worker com `ray start --address=<IP_HEAD>:6379`
#                    rode no head: python train.py --address=auto
# =============================================================================

import time
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, balanced_accuracy_score, confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
import joblib
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CAMINHOS E HIPERPARAMETROS
# =============================================================================
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / 'PlantVillage-completo'
SPLITS_DIR  = DATA_DIR / 'splits'
RAW_DIR     = DATA_DIR / 'raw'
CKPT_DIR    = BASE_DIR / 'checkpoints'
RESULTS_DIR = BASE_DIR / 'results'
CKPT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

IMG_SIZE    = 224
BATCH_SIZE  = 128
NUM_EPOCHS  = 10
LR          = 1e-3
NUM_WORKERS = 4       # funciona corretamente em script .py no Windows
SEED        = 42
VERSIONS    = ['color', 'grayscale', 'segmented']

# =============================================================================
# DATASET
# =============================================================================
CLASS_NAMES = sorted([d.name for d in (RAW_DIR / 'color').iterdir() if d.is_dir()])
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

CONFIG = {
    'data_dir':    str(DATA_DIR),
    'splits_dir':  str(SPLITS_DIR),
    'ckpt_dir':    str(CKPT_DIR),
    'img_size':    IMG_SIZE,
    'batch_size':  BATCH_SIZE,
    'num_epochs':  NUM_EPOCHS,
    'lr':          LR,
    'num_workers': NUM_WORKERS,
    'seed':        SEED,
    'class_to_idx': CLASS_TO_IDX,
}


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


# =============================================================================
# RAY ACTOR — ModelTrainer
# Nivel 3 (distribuido): cada actor roda num no do cluster Ray
# Nivel 2 (paralelo CPU): DataLoader com num_workers processos paralelos
# Nivel 1 (paralelo GPU): CUDA processa cada batch em paralelo nos cores
# =============================================================================
@ray.remote(num_gpus=0.33)
class ModelTrainer:

    def __init__(self, version, config):
        import torch
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from pathlib import Path
        from PIL import Image

        torch.manual_seed(config['seed'])

        self.version = version
        self.config  = config
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        data_dir   = Path(config['data_dir'])
        splits_dir = Path(config['splits_dir'])
        img_size   = config['img_size']
        bs         = config['batch_size']
        nw         = config['num_workers']
        pin        = (self.device.type == 'cuda')
        mean       = [0.485, 0.456, 0.406]
        std        = [0.229, 0.224, 0.225]

        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # Importa dataset localmente no actor (cada no tem sua copia)
        from train import PlantVillageDataset

        train_ds = PlantVillageDataset(splits_dir / f'{version}_train.txt', data_dir, config['class_to_idx'], transform=train_tf)
        test_ds  = PlantVillageDataset(splits_dir / f'{version}_test.txt',  data_dir, config['class_to_idx'], transform=eval_tf)

        self.n_train = len(train_ds)
        self.n_test  = len(test_ds)

        # Nivel 2: num_workers processos CPU carregam imagens em paralelo
        self.train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                       num_workers=nw, pin_memory=pin,
                                       persistent_workers=(nw > 0))
        self.val_loader   = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                                       num_workers=nw, pin_memory=pin,
                                       persistent_workers=(nw > 0))

    def _build_model(self):
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        return model.to(self.device)

    def train(self):
        ver         = self.version
        ckpt_dir    = Path(self.config['ckpt_dir'])
        done_path   = ckpt_dir / f'model_{ver}_DONE.pt'
        resume_path = ckpt_dir / f'model_{ver}_resume.pt'
        best_path   = ckpt_dir / f'model_{ver}_best.pt'

        if done_path.exists():
            history = torch.load(done_path, map_location='cpu', weights_only=False)
            print(f'[{ver}] DONE — pulando (melhor val_acc: {max(history["val_acc"]):.4f})', flush=True)
            return history

        model     = self._build_model()
        optimizer = optim.Adam(model.fc.parameters(), lr=self.config['lr'])
        scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
        criterion = nn.CrossEntropyLoss()
        num_epochs = self.config['num_epochs']
        history   = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'best_val_acc': 0.0}
        start_epoch = 0

        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            history     = ckpt['history']
            start_epoch = ckpt['epoch'] + 1
            print(f'[{ver}] Retomando do epoch {start_epoch} — device: {self.device}', flush=True)
        else:
            print(f'[{ver}] Iniciando do zero — device: {self.device}', flush=True)

        for epoch in range(start_epoch, num_epochs):
            t0 = time.time()

            # Nivel 1: CUDA — GPU processa cada batch em paralelo nos cores CUDA
            model.train()
            train_loss, train_correct = 0.0, 0
            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                out  = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                train_loss    += loss.item() * imgs.size(0)
                train_correct += (out.argmax(1) == labels).sum().item()

            model.eval()
            val_correct = 0
            with torch.no_grad():
                for imgs, labels in self.val_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    val_correct += (model(imgs).argmax(1) == labels).sum().item()

            scheduler.step()

            ep_loss      = train_loss    / self.n_train
            ep_train_acc = train_correct / self.n_train
            ep_val_acc   = val_correct   / self.n_test

            history['train_loss'].append(ep_loss)
            history['train_acc'].append(ep_train_acc)
            history['val_acc'].append(ep_val_acc)

            if ep_val_acc > history['best_val_acc']:
                history['best_val_acc'] = ep_val_acc
                torch.save({'model_state_dict': model.state_dict(), 'val_acc': ep_val_acc}, best_path)

            print(f'[{ver}] Epoch {epoch+1}/{num_epochs} | '
                  f'loss={ep_loss:.4f} | train={ep_train_acc:.4f} | '
                  f'val={ep_val_acc:.4f} | {time.time()-t0:.1f}s', flush=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
            }, resume_path)

        resume_path.unlink(missing_ok=True)
        torch.save(history, done_path)
        print(f'[{ver}] Treino concluido — melhor val_acc: {history["best_val_acc"]:.4f}', flush=True)
        return history


def build_model_local(device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', default=None,
                        help='Endereco do cluster Ray (ex: auto ou IP:6379). '
                             'Omitir para rodar em modo local.')
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # -------------------------------------------------------------------------
    # Nivel 3: inicializa cluster Ray
    # -------------------------------------------------------------------------
    if args.address:
        ray.init(address=args.address, ignore_reinit_error=True)
    else:
        ray.init(ignore_reinit_error=True)

    nodes = ray.nodes()
    resources = ray.cluster_resources()
    print(f'\nCluster Ray — {len(nodes)} no(s):')
    for n in nodes:
        status = 'ATIVO' if n['Alive'] else 'INATIVO'
        addr   = n['NodeManagerAddress']
        cpus   = n['Resources'].get('CPU', 0)
        gpus   = n['Resources'].get('GPU', 0)
        print(f'  [{status}] {addr}  CPUs={cpus:.0f}  GPUs={gpus:.0f}')
    print(f'Total — CPUs: {resources.get("CPU",0):.0f}  GPUs: {resources.get("GPU",0):.0f}\n')

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if DEVICE.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f'Head node GPU: {props.name} ({props.total_memory/1e9:.1f} GB)')

    print(f'\n{NUM_CLASSES} classes | IMG={IMG_SIZE} | BATCH={BATCH_SIZE} | EPOCHS={NUM_EPOCHS} | WORKERS={NUM_WORKERS}')
    for ver in VERSIONS:
        n_train = sum(1 for _ in open(SPLITS_DIR / f'{ver}_train.txt'))
        n_test  = sum(1 for _ in open(SPLITS_DIR / f'{ver}_test.txt'))
        print(f'  {ver:12s} train: {n_train:,} | test: {n_test:,}')

    # -------------------------------------------------------------------------
    # EDA e demonstracao ExG (skip se PNGs ja existem)
    # -------------------------------------------------------------------------
    EDA_DIST = RESULTS_DIR / 'eda_distribuicao.png'
    if not EDA_DIST.exists():
        culture_counts = Counter()
        with open(SPLITS_DIR / 'color_train.txt') as f:
            for line in f:
                parts = line.strip().split('/')
                if len(parts) >= 3:
                    culture_counts[parts[2].split('___')[0]] += 1
        cultures = sorted(culture_counts.keys())
        counts   = [culture_counts[c] for c in cultures]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(cultures, counts, color='steelblue')
        ax.set_title('Distribuicao por Cultura (color train set)')
        ax.set_xlabel('Cultura'); ax.set_ylabel('Numero de imagens')
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        plt.savefig(EDA_DIST, dpi=120); plt.close()
        print(f'Salvo: {EDA_DIST}')

    EXG_PATH = RESULTS_DIR / 'exg_demo.png'
    if not EXG_PATH.exists():
        with open(SPLITS_DIR / 'color_train.txt') as f:
            first_line = f.readline().strip().split()[0]
        img_arr = np.array(Image.open(DATA_DIR / first_line).convert('RGB'))
        r, g, b = img_arr[:,:,0]/255., img_arr[:,:,1]/255., img_arr[:,:,2]/255.
        exg      = 2*g - r - b
        exg_norm = (exg - exg.min()) / (exg.max() - exg.min() + 1e-8)
        mask     = exg > 0.1
        seg      = img_arr.copy(); seg[~mask] = 0
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_arr);  axes[0].set_title('Original');      axes[0].axis('off')
        axes[1].imshow(exg_norm, cmap='RdYlGn'); axes[1].set_title('ExG'); axes[1].axis('off')
        axes[2].imshow(seg);      axes[2].set_title('Segmentado');    axes[2].axis('off')
        plt.suptitle('Filtro Excess Green (ExG)'); plt.tight_layout()
        plt.savefig(EXG_PATH, dpi=120); plt.close()
        print(f'Salvo: {EXG_PATH}')

    # -------------------------------------------------------------------------
    # Nivel 3: treino paralelo distribuido via Ray
    # Com 1 GPU: 3 actors compartilham (num_gpus=1 por actor exige cluster real)
    # Com 2 PCs (2 GPUs): Ray aloca 1 actor por GPU automaticamente
    # -------------------------------------------------------------------------
    all_done = all((CKPT_DIR / f'model_{ver}_DONE.pt').exists() for ver in VERSIONS)
    if all_done:
        print('\nTodos os modelos ja treinados — carregando historicos...')
        histories = {}
        for ver in VERSIONS:
            h = torch.load(CKPT_DIR / f'model_{ver}_DONE.pt', map_location='cpu', weights_only=False)
            histories[ver] = h
            print(f'  [{ver}] melhor val_acc: {max(h["val_acc"]):.4f}')
    else:
        print('\nIniciando treino paralelo dos 3 modelos (Ray)...')
        t_start  = time.time()
        trainers = {ver: ModelTrainer.remote(ver, CONFIG) for ver in VERSIONS}
        futures  = [trainers[ver].train.remote() for ver in VERSIONS]
        results  = ray.get(futures)
        histories = dict(zip(VERSIONS, results))
        elapsed  = time.time() - t_start
        print(f'\nTreino concluido em {elapsed/60:.1f} min')
        for ver, h in histories.items():
            print(f'  {ver:12s} melhor val_acc: {h["best_val_acc"]:.4f}')

    # -------------------------------------------------------------------------
    # Curvas de aprendizado
    # -------------------------------------------------------------------------
    CURVES_PATH = RESULTS_DIR / 'curvas_aprendizado.png'
    if not CURVES_PATH.exists():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = {'color': 'tab:green', 'grayscale': 'tab:gray', 'segmented': 'tab:blue'}
        for ver, h in histories.items():
            ep = range(1, len(h['train_loss']) + 1)
            axes[0].plot(ep, h['train_loss'], label=ver, color=colors[ver])
            axes[1].plot(ep, h['val_acc'],    label=ver, color=colors[ver])
        axes[0].set_title('Loss de treino');   axes[0].set_xlabel('Epoch'); axes[0].legend()
        axes[1].set_title('Acc de validacao'); axes[1].set_xlabel('Epoch'); axes[1].legend()
        plt.tight_layout()
        plt.savefig(CURVES_PATH, dpi=120); plt.close()
        print(f'Salvo: {CURVES_PATH}')

    # -------------------------------------------------------------------------
    # Carrega melhores modelos
    # -------------------------------------------------------------------------
    best_models = {}
    for ver in VERSIONS:
        best_path = CKPT_DIR / f'model_{ver}_best.pt'
        if not best_path.exists():
            print(f'[{ver}] best.pt nao encontrado — execute o treino primeiro')
            continue
        model = build_model_local(DEVICE)
        ckpt  = torch.load(best_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        best_models[ver] = model
        print(f'[{ver}] carregado (val_acc: {ckpt["val_acc"]:.4f})')

    # -------------------------------------------------------------------------
    # Avaliacao individual — todos no color_test.txt para comparacao justa
    # -------------------------------------------------------------------------
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    shared_test_ds = PlantVillageDataset(SPLITS_DIR / 'color_test.txt', DATA_DIR, CLASS_TO_IDX, transform=eval_tf)
    shared_loader  = DataLoader(shared_test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True,
                                persistent_workers=True)

    n_batches = len(shared_loader)
    print('\nAvaliacao individual (color_test.txt):')
    ind_results = {}
    for ver in VERSIONS:
        if ver not in best_models:
            continue
        model = best_models[ver]
        model.eval()
        preds, lbls = [], []
        t0 = time.time()
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(shared_loader, 1):
                preds.extend(model(imgs.to(DEVICE)).argmax(1).cpu().numpy())
                lbls.extend(labels.numpy())
                if i % 20 == 0 or i == n_batches:
                    print(f'  [{ver}] batch {i}/{n_batches} ({time.time()-t0:.0f}s)', flush=True)
        ind_results[ver] = {
            'acc':     accuracy_score(lbls, preds),
            'f1':      f1_score(lbls, preds, average='macro', zero_division=0),
            'prec':    precision_score(lbls, preds, average='macro', zero_division=0),
            'rec':     recall_score(lbls, preds, average='macro', zero_division=0),
            'bal_acc': balanced_accuracy_score(lbls, preds),
        }
        r = ind_results[ver]
        print(f'  {ver:12s} acc={r["acc"]:.4f}  f1={r["f1"]:.4f}  '
              f'prec={r["prec"]:.4f}  rec={r["rec"]:.4f}  bal_acc={r["bal_acc"]:.4f}')

    # -------------------------------------------------------------------------
    # Heatmap de metricas
    # -------------------------------------------------------------------------
    HEATMAP_PATH = RESULTS_DIR / 'metricas_heatmap.png'
    if not HEATMAP_PATH.exists():
        metrics = ['acc', 'f1', 'prec', 'rec', 'bal_acc']
        rows    = [ver for ver in VERSIONS if ver in ind_results]
        data    = [[ind_results[ver][m] for m in metrics] for ver in rows]
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.heatmap(data, annot=True, fmt='.4f', xticklabels=metrics,
                    yticklabels=rows, cmap='YlGn', vmin=0, vmax=1, ax=ax)
        ax.set_title('Metricas por versao do dataset')
        plt.tight_layout()
        plt.savefig(HEATMAP_PATH, dpi=120); plt.close()
        print(f'Salvo: {HEATMAP_PATH}')

    # -------------------------------------------------------------------------
    # Ensemble por media de softmax
    # -------------------------------------------------------------------------
    print('\nCalculando ensemble (media de softmax)...')
    test_probs_all  = {}
    test_labels_all = None

    for ver in VERSIONS:
        if ver not in best_models:
            continue
        model = best_models[ver]
        model.eval()
        ps, ls = [], []
        t0 = time.time()
        print(f'  [{ver}] calculando probs...', flush=True)
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(shared_loader, 1):
                out = model(imgs.to(DEVICE))
                ps.append(F.softmax(out, dim=1).cpu().numpy())
                ls.append(labels.numpy())
                if i % 20 == 0 or i == n_batches:
                    print(f'  [{ver}] batch {i}/{n_batches} ({time.time()-t0:.0f}s)', flush=True)
        test_probs_all[ver] = np.concatenate(ps)
        if test_labels_all is None:
            test_labels_all = np.concatenate(ls)
        print(f'  [{ver}] probs calculadas: {test_probs_all[ver].shape}')

    avg_probs      = np.mean([test_probs_all[ver] for ver in VERSIONS if ver in test_probs_all], axis=0)
    ensemble_preds = avg_probs.argmax(axis=1)

    ens_result = {
        'acc':     accuracy_score(test_labels_all, ensemble_preds),
        'f1':      f1_score(test_labels_all, ensemble_preds, average='macro', zero_division=0),
        'prec':    precision_score(test_labels_all, ensemble_preds, average='macro', zero_division=0),
        'rec':     recall_score(test_labels_all, ensemble_preds, average='macro', zero_division=0),
        'bal_acc': balanced_accuracy_score(test_labels_all, ensemble_preds),
    }
    r = ens_result
    print(f'  ensemble      acc={r["acc"]:.4f}  f1={r["f1"]:.4f}  '
          f'prec={r["prec"]:.4f}  rec={r["rec"]:.4f}  bal_acc={r["bal_acc"]:.4f}')

    # -------------------------------------------------------------------------
    # Stacking — meta-classificador treinado nas probs do color_train.txt
    #
    # NOTA: grayscale e segmented recebem imagens coloridas (input "errado"),
    #       porque os splits de treino das 3 versoes sao desalinhados (UUIDs
    #       distintos). Este experimento e exploratorio — estima o potencial
    #       do stacking, mas NAO e uma comparacao justa com os individuais.
    # -------------------------------------------------------------------------
    STACK_PATH    = CKPT_DIR / 'stacking_meta.pkl'
    STACK_CS_PATH = CKPT_DIR / 'stacking_cs_meta.pkl'
    stacking_result    = None
    stacking_cs_result = None
    train_probs_all    = {}
    train_labels_meta  = None

    if STACK_PATH.exists():
        print('\n[stacking] Carregando resultado salvo...')
        stacking_result = joblib.load(STACK_PATH)
        r = stacking_result
        print(f'  stacking      acc={r["acc"]:.4f}  f1={r["f1"]:.4f}  '
              f'prec={r["prec"]:.4f}  rec={r["rec"]:.4f}  bal_acc={r["bal_acc"]:.4f}')
    elif len(best_models) == len(VERSIONS):
        print('\nCalculando stacking (meta-features via color_train.txt)...')
        print('  NOTA: grayscale e segmented recebem imagens coloridas — comparacao nao justa.')

        train_ds_color = PlantVillageDataset(
            SPLITS_DIR / 'color_train.txt', DATA_DIR, CLASS_TO_IDX, transform=eval_tf)
        train_loader_meta = DataLoader(
            train_ds_color, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
        n_meta_batches = len(train_loader_meta)

        for ver in VERSIONS:
            model = best_models[ver]
            model.eval()
            ps, ls = [], []
            t0 = time.time()
            print(f'  [{ver}] probs no train...', flush=True)
            with torch.no_grad():
                for i, (imgs, labels) in enumerate(train_loader_meta, 1):
                    out = model(imgs.to(DEVICE))
                    ps.append(F.softmax(out, dim=1).cpu().numpy())
                    ls.append(labels.numpy())
                    if i % 50 == 0 or i == n_meta_batches:
                        print(f'  [{ver}] batch {i}/{n_meta_batches} ({time.time()-t0:.0f}s)', flush=True)
            train_probs_all[ver] = np.concatenate(ps)
            if train_labels_meta is None:
                train_labels_meta = np.concatenate(ls)
            print(f'  [{ver}] shape: {train_probs_all[ver].shape}')

        X_train = np.hstack([train_probs_all[v] for v in VERSIONS])   # (N_train, 114)
        X_test  = np.hstack([test_probs_all[v]  for v in VERSIONS])   # (N_test,  114)
        print(f'  Meta-features: train={X_train.shape}  test={X_test.shape}')
        print('  Treinando LogisticRegression...', flush=True)

        meta = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                   multi_class='multinomial', n_jobs=-1)
        meta.fit(X_train, train_labels_meta)
        stack_preds = meta.predict(X_test)

        stacking_result = {
            'acc':     accuracy_score(test_labels_all, stack_preds),
            'f1':      f1_score(test_labels_all, stack_preds, average='macro', zero_division=0),
            'prec':    precision_score(test_labels_all, stack_preds, average='macro', zero_division=0),
            'rec':     recall_score(test_labels_all, stack_preds, average='macro', zero_division=0),
            'bal_acc': balanced_accuracy_score(test_labels_all, stack_preds),
            'note':    'input colorido para grayscale/segmented — comparacao nao justa',
        }
        joblib.dump(stacking_result, STACK_PATH)
        joblib.dump(meta, CKPT_DIR / 'stacking_model.pkl')
        r = stacking_result
        print(f'  stacking      acc={r["acc"]:.4f}  f1={r["f1"]:.4f}  '
              f'prec={r["prec"]:.4f}  rec={r["rec"]:.4f}  bal_acc={r["bal_acc"]:.4f}')
        print(f'  ({r["note"]})')
    else:
        print('\n[stacking] Ignorado — nem todos os modelos carregados.')

    # -------------------------------------------------------------------------
    # Stacking color + segmented (justo — ambos treinados em imagens coloridas)
    # color foi treinado em coloridas; segmented tambem aceita coloridas (fundo
    # removido no treino, mas input RGB identico). Comparacao justa com color.
    # -------------------------------------------------------------------------
    CS_VERSIONS = ['color', 'segmented']

    if STACK_CS_PATH.exists():
        print('\n[stacking c+s] Carregando resultado salvo...')
        stacking_cs_result = joblib.load(STACK_CS_PATH)
        r = stacking_cs_result
        print(f'  stacking c+s  acc={r["acc"]:.4f}  f1={r["f1"]:.4f}  '
              f'prec={r["prec"]:.4f}  rec={r["rec"]:.4f}  bal_acc={r["bal_acc"]:.4f}')
    elif all(v in best_models for v in CS_VERSIONS):
        print('\nCalculando stacking color+segmented (color_train.txt)...')

        # Reutiliza probs ja calculadas pelo bloco anterior; recalcula se necessario
        if not train_probs_all:
            train_ds_color = PlantVillageDataset(
                SPLITS_DIR / 'color_train.txt', DATA_DIR, CLASS_TO_IDX, transform=eval_tf)
            train_loader_meta = DataLoader(
                train_ds_color, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
            n_meta_batches = len(train_loader_meta)
            for ver in CS_VERSIONS:
                model = best_models[ver]
                model.eval()
                ps, ls = [], []
                t0 = time.time()
                print(f'  [{ver}] probs no train...', flush=True)
                with torch.no_grad():
                    for i, (imgs, labels) in enumerate(train_loader_meta, 1):
                        out = model(imgs.to(DEVICE))
                        ps.append(F.softmax(out, dim=1).cpu().numpy())
                        ls.append(labels.numpy())
                        if i % 50 == 0 or i == n_meta_batches:
                            print(f'  [{ver}] batch {i}/{n_meta_batches} ({time.time()-t0:.0f}s)', flush=True)
                train_probs_all[ver] = np.concatenate(ps)
                if train_labels_meta is None:
                    train_labels_meta = np.concatenate(ls)
                print(f'  [{ver}] shape: {train_probs_all[ver].shape}')

        X_train_cs = np.hstack([train_probs_all[v] for v in CS_VERSIONS])  # (N_train, 76)
        X_test_cs  = np.hstack([test_probs_all[v]  for v in CS_VERSIONS])  # (N_test,  76)
        print(f'  Meta-features: train={X_train_cs.shape}  test={X_test_cs.shape}')
        print('  Treinando LogisticRegression...', flush=True)

        meta_cs = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                     multi_class='multinomial', n_jobs=-1)
        meta_cs.fit(X_train_cs, train_labels_meta)
        stack_cs_preds = meta_cs.predict(X_test_cs)

        stacking_cs_result = {
            'acc':     accuracy_score(test_labels_all, stack_cs_preds),
            'f1':      f1_score(test_labels_all, stack_cs_preds, average='macro', zero_division=0),
            'prec':    precision_score(test_labels_all, stack_cs_preds, average='macro', zero_division=0),
            'rec':     recall_score(test_labels_all, stack_cs_preds, average='macro', zero_division=0),
            'bal_acc': balanced_accuracy_score(test_labels_all, stack_cs_preds),
        }
        joblib.dump(stacking_cs_result, STACK_CS_PATH)
        joblib.dump(meta_cs, CKPT_DIR / 'stacking_cs_model.pkl')
        r = stacking_cs_result
        print(f'  stacking c+s  acc={r["acc"]:.4f}  f1={r["f1"]:.4f}  '
              f'prec={r["prec"]:.4f}  rec={r["rec"]:.4f}  bal_acc={r["bal_acc"]:.4f}')
    else:
        print('\n[stacking c+s] Ignorado — modelos color ou segmented nao carregados.')

    # -------------------------------------------------------------------------
    # Graficos finais
    # -------------------------------------------------------------------------
    FINAL_PATH = RESULTS_DIR / 'comparacao_final.png'
    if not FINAL_PATH.exists():
        nomes  = list(VERSIONS) + ['softmax']
        accs   = [ind_results[v]['acc'] for v in VERSIONS] + [ens_result['acc']]
        cores  = ['tab:green', 'tab:gray', 'tab:blue', 'tab:orange']
        labels = list(VERSIONS) + ['softmax']
        if stacking_result:
            nomes.append('stacking*')
            accs.append(stacking_result['acc'])
            cores.append('tab:red')
            labels.append('stacking\n(*nao justo)')
        if stacking_cs_result:
            nomes.append('stacking c+s')
            accs.append(stacking_cs_result['acc'])
            cores.append('tab:purple')
            labels.append('stacking\ncolor+seg')
        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.bar(labels, accs, color=cores[:len(labels)])
        ax.bar_label(bars, fmt='%.4f', padding=3)
        ax.set_ylim(0, 1.08)
        ax.set_title('Acuracia: individuais vs ensembles')
        ax.set_ylabel('Acuracia')
        plt.tight_layout()
        plt.savefig(FINAL_PATH, dpi=120); plt.close()
        print(f'Salvo: {FINAL_PATH}')

    CM_PATH = RESULTS_DIR / 'matriz_confusao_ensemble.png'
    if not CM_PATH.exists():
        cm = confusion_matrix(test_labels_all, ensemble_preds)
        fig, ax = plt.subplots(figsize=(18, 16))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title('Matriz de Confusao — Ensemble')
        ax.set_xlabel('Predito'); ax.set_ylabel('Real')
        plt.xticks(rotation=90, fontsize=7); plt.yticks(rotation=0, fontsize=7)
        plt.tight_layout()
        plt.savefig(CM_PATH, dpi=120); plt.close()
        print(f'Salvo: {CM_PATH}')

    # -------------------------------------------------------------------------
    # Resultado final — questao de pesquisa
    # -------------------------------------------------------------------------
    print()
    print('=' * 64)
    print('QUESTAO DE PESQUISA')
    print('=' * 64)
    print('O pipeline distribuido (Ray) com paralelismo (CUDA + DataLoader)')
    print('reduz o tempo de treinamento, e o ensemble supera os modelos individuais?')
    print()

    best_ver  = max(ind_results, key=lambda v: ind_results[v]['acc'])
    best_ind  = ind_results[best_ver]['acc']
    ens_acc   = ens_result['acc']
    gain_soft = (ens_acc - best_ind) * 100

    print(f'Melhor modelo individual : {best_ver:12s} acc={best_ind:.4f}')
    print(f'Softmax ensemble         :              acc={ens_acc:.4f}  ({gain_soft:+.2f} pp)')
    if stacking_result:
        gain_stack = (stacking_result['acc'] - best_ind) * 100
        print(f'Stacking (nao justo)     :              acc={stacking_result["acc"]:.4f}  ({gain_stack:+.2f} pp)')
        print(f'  * grayscale e segmented avaliados em imagens coloridas')
    if stacking_cs_result:
        gain_cs = (stacking_cs_result['acc'] - best_ind) * 100
        print(f'Stacking color+segmented :              acc={stacking_cs_result["acc"]:.4f}  ({gain_cs:+.2f} pp)')
    print()
    if gain_soft > 0.5:
        print('Softmax ensemble: SIM — supera os modelos individuais de forma relevante.')
    elif gain_soft > 0:
        print('Softmax ensemble: MARGINAL — ganho existe mas e pequeno.')
    else:
        print('Softmax ensemble: NAO — nao supera o melhor modelo individual.')
    if stacking_result:
        gain_stack = (stacking_result['acc'] - best_ind) * 100
        if gain_stack > 0.5:
            print('Stacking (estimativa): PROMISSOR — sugere que stacking com splits alinhados poderia superar.')
        elif gain_stack > 0:
            print('Stacking (estimativa): MARGINAL — pequeno ganho mesmo com input subotimo.')
        else:
            print('Stacking (estimativa): NAO supera — meta-classificador nao compensou o input errado.')
    if stacking_cs_result:
        if gain_cs > 0.5:
            print('Stacking color+seg: SIM — supera o melhor modelo individual de forma relevante.')
        elif gain_cs > 0:
            print('Stacking color+seg: MARGINAL — ganho existe mas e pequeno.')
        else:
            print('Stacking color+seg: NAO — nao supera o melhor modelo individual.')
    print('=' * 64)

    ray.shutdown()
