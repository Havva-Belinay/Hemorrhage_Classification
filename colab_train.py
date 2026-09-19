"""
Hemorrhage Binary Classification

İki model:
  1) Custom CNN   (sıfırdan tasarlanmış VGG-style)
  2) ConvNeXt-Tiny (ImageNet pretrained, fine-tuned)

Her ikisi için: EarlyStopping, ReduceLROnPlateau, eğitim grafikleri,
confusion matrix, Accuracy/Precision/Recall/F1

Colab'da çalıştırma:
    !python /content/colab_train.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# ADIM 1 — Veri setini çıkart
# ══════════════════════════════════════════════════════════════════════════════
import os

LOCAL_ZIP = '/content/head-ct-hemorrahage.zip'
DRIVE_ZIP = '/content/drive/MyDrive/head-ct-hemorrahage.zip'

if not os.path.exists('/content/labels.csv'):
    if os.path.exists(LOCAL_ZIP):
        print('Zip /content/ içinde bulundu, çıkartılıyor...')
        os.system(f'unzip -q "{LOCAL_ZIP}" -d /content/')
        print('Tamamlandı.')
    elif os.path.exists(DRIVE_ZIP):
        print("Drive'dan çıkartılıyor...")
        os.system(f'unzip -q "{DRIVE_ZIP}" -d /content/')
        print('Tamamlandı.')
    else:
        raise FileNotFoundError(
            'head-ct-hemorrahage.zip bulunamadı!\n'
            f'  Aranan: {LOCAL_ZIP}\n'
            f'  Aranan: {DRIVE_ZIP}'
        )
else:
    print('Veri seti zaten mevcut.')

# Zip içeriği doğrudan /content/ altına çıkar (labels.csv + head_ct/)
DATA_DIR = '/content'

if os.path.exists('/content/drive/MyDrive'):
    SAVE_DIR = '/content/drive/MyDrive/hemorrhage_detection_results'
else:
    SAVE_DIR = '/content/hemorrhage_detection_results'
os.makedirs(SAVE_DIR, exist_ok=True)

print(f'DATA_DIR : {DATA_DIR}')
print(f'SAVE_DIR : {SAVE_DIR}')

# ══════════════════════════════════════════════════════════════════════════════
# ADIM 2 — Kurulum
# ══════════════════════════════════════════════════════════════════════════════
os.system('pip install -q torchinfo')

# ══════════════════════════════════════════════════════════════════════════════
# ADIM 3 — Import ve sabitler
# ══════════════════════════════════════════════════════════════════════════════
import json, random, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              confusion_matrix, classification_report)

SEED         = 42
IMG_SIZE     = 224
BATCH_SIZE   = 16
EPOCHS       = 60
PATIENCE     = 12          # EarlyStopping patience (val_loss)
CLASS_NAMES  = ['Normal', 'Hemorrhage']

# Hyperparameter configs — deneysel olarak optimize edilmiş değerler
# Custom CNN: LR=3e-4 küçük veri setinde stabil eğitim sağlıyor
# ConvNeXt  : LR=5e-5 + yüksek weight_decay overfitting'i azaltıyor
CUSTOM_CFG   = {'lr': 3e-4, 'weight_decay': 5e-4, 'dropout': 0.5,
                'epochs': EPOCHS, 'patience': PATIENCE}
CONVNEXT_CFG = {'lr': 5e-5, 'weight_decay': 1e-3, 'dropout': 0.5,
                'epochs': EPOCHS, 'patience': PATIENCE}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU   : {torch.cuda.get_device_name(0)}')

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed()
print('Sabitler hazır.')

# ══════════════════════════════════════════════════════════════════════════════
# ADIM 4 — Veri yükleme
# ══════════════════════════════════════════════════════════════════════════════
def load_dataset(data_dir):
    labels_path = os.path.join(data_dir, 'labels.csv')
    images_dir  = os.path.join(data_dir, 'head_ct', 'head_ct')
    df = pd.read_csv(labels_path)
    df.columns = df.columns.str.strip()
    image_paths, labels = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(images_dir, f"{int(row['id']):03d}.png")
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(int(row['hemorrhage']))
    print(f'Toplam görüntü  : {len(image_paths)}')
    print(f'  Normal (0)    : {labels.count(0)}')
    print(f'  Hemorrhage (1): {labels.count(1)}')
    return image_paths, labels

image_paths, labels = load_dataset(DATA_DIR)
image_paths = np.array(image_paths)
labels      = np.array(labels)

# ══════════════════════════════════════════════════════════════════════════════
# ADIM 5 — Stratified 70 / 15 / 15 split
# ══════════════════════════════════════════════════════════════════════════════
X_tv, X_test, y_tv, y_test = train_test_split(
    image_paths, labels, test_size=0.15, stratify=labels, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.15/0.85, stratify=y_tv, random_state=SEED)

print(f'\nStratified split (seed={SEED}):')
print(f'  Train : {len(X_train):3d}  (Normal={( y_train==0).sum()}, Hem={(y_train==1).sum()})')
print(f'  Val   : {len(X_val):3d}  (Normal={( y_val==0).sum()}, Hem={(y_val==1).sum()})')
print(f'  Test  : {len(X_test):3d}  (Normal={( y_test==0).sum()}, Hem={(y_test==1).sum()}) ← DOKUNULMAZ')

# ══════════════════════════════════════════════════════════════════════════════
# ADIM 6 — Dataset sınıfı ve Transforms
# ══════════════════════════════════════════════════════════════════════════════
class HeadCTDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img   = Image.open(self.paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform: img = self.transform(img)
        return img, label

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

def get_train_transform():
    """Augmentation sadece train setine — veri sızıntısını önler."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

def get_eval_transform():
    """Val / Test: augmentation YOK."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

def make_loaders(bs=BATCH_SIZE):
    train_ld = DataLoader(HeadCTDataset(X_train, y_train, get_train_transform()),
                          batch_size=bs, shuffle=True,  num_workers=2, pin_memory=True)
    val_ld   = DataLoader(HeadCTDataset(X_val,   y_val,   get_eval_transform()),
                          batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    test_ld  = DataLoader(HeadCTDataset(X_test,  y_test,  get_eval_transform()),
                          batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    return train_ld, val_ld, test_ld

print('Dataset ve transforms hazır.')

# ══════════════════════════════════════════════════════════════════════════════
# ADIM 7a — Custom CNN (sıfırdan tasarlanmış)
# ══════════════════════════════════════════════════════════════════════════════
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
    def forward(self, x): return self.block(x)

class CustomCNN(nn.Module):
    """
    4-blok VGG-tarzı CNN.
    Girdi : (B, 3, 224, 224)
    Çıktı : (B, 2)
    """
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32),   # → 112×112
            ConvBlock(32,  64),   # →  56× 56
            ConvBlock(64,  128),  # →  28× 28
            ConvBlock(128, 256),  # →  14× 14
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 512), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(512, 128),     nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))

# ══════════════════════════════════════════════════════════════════════════════
# ADIM 7b — ConvNeXt-Tiny (ImageNet pretrained)
# ══════════════════════════════════════════════════════════════════════════════
def build_convnext(num_classes=2, dropout_rate=0.5, freeze_ratio=0.875):
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    n = len(model.features)
    for i, blk in enumerate(model.features):
        for p in blk.parameters():
            p.requires_grad = (i >= int(n * freeze_ratio))
    in_f = model.classifier[2].in_features  # 768
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.LayerNorm(in_f),
        nn.Dropout(dropout_rate),
        nn.Linear(in_f, 256), nn.GELU(),
        nn.Dropout(dropout_rate),
        nn.Linear(256, num_classes),
    )
    return model

# ══════════════════════════════════════════════════════════════════════════════
# ADIM 8 — Eğitim & değerlendirme fonksiyonları
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == lbls).sum().item()
        n        += imgs.size(0)
    return loss_sum / n, 100 * correct / n

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == lbls).sum().item()
        n        += imgs.size(0)
    return loss_sum / n, 100 * correct / n

@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, trues = [], []
    for imgs, lbls in loader:
        preds.extend(model(imgs.to(DEVICE)).argmax(1).cpu().numpy())
        trues.extend(lbls.numpy())
    return np.array(preds), np.array(trues)


def train_model(model, train_ld, val_ld, lr, weight_decay, model_name,
                epochs=30, patience=5, label_smoothing=0.0):
    """Eğitim döngüsü: EarlyStopping + ReduceLROnPlateau."""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_loss, best_weights, no_improve = float('inf'), None, 0

    print(f'\n{"="*60}')
    print(f'  {model_name} eğitimi başlıyor...')
    print(f'  epochs={epochs}, patience={patience}, lr={lr:.1e}')
    print(f'{"="*60}')

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_ld, optimizer, criterion)
        vl_loss, vl_acc = eval_epoch(model,  val_ld,   criterion)
        scheduler.step(vl_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        print(f'  Epoch {epoch:3d}/{epochs} | '
              f'Train {tr_loss:.4f}/{tr_acc:.1f}% | '
              f'Val {vl_loss:.4f}/{vl_acc:.1f}% | '
              f'LR {optimizer.param_groups[0]["lr"]:.1e}')

        if vl_loss < best_loss - 1e-4:
            best_loss = vl_loss
            best_weights = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  Early stopping — epoch {epoch}')
                break

    model.load_state_dict(best_weights)
    print(f'  En iyi Val Loss: {best_loss:.4f}')
    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 9 — Custom CNN eğitimi
# ══════════════════════════════════════════════════════════════════════════════
set_seed()
train_ld, val_ld, test_ld = make_loaders()

custom_model = CustomCNN(num_classes=2, dropout_rate=CUSTOM_CFG['dropout'])
total = sum(p.numel() for p in custom_model.parameters())
CUSTOM_CNN_NAME = 'Custom CNN'
print(f'\n{CUSTOM_CNN_NAME} — Toplam parametre: {total:,}')

custom_model, custom_hist = train_model(
    custom_model, train_ld, val_ld,
    lr=CUSTOM_CFG['lr'], weight_decay=CUSTOM_CFG['weight_decay'],
    model_name=CUSTOM_CNN_NAME,
    epochs=CUSTOM_CFG['epochs'], patience=CUSTOM_CFG['patience']
)

torch.save(custom_model.state_dict(), f'{SAVE_DIR}/custom_cnn_best.pth')

# ══════════════════════════════════════════════════════════════════════════════
# ADIM 10 — ConvNeXt eğitimi
# ══════════════════════════════════════════════════════════════════════════════
set_seed()
train_ld, val_ld, test_ld = make_loaders()

convnext_model = build_convnext(num_classes=2, dropout_rate=CONVNEXT_CFG['dropout'])
total     = sum(p.numel() for p in convnext_model.parameters())
trainable = sum(p.numel() for p in convnext_model.parameters() if p.requires_grad)
print(f'\nConvNeXt — Toplam: {total:,}  |  Eğitilebilir: {trainable:,} ({100*trainable/total:.1f}%)')

convnext_model, convnext_hist = train_model(
    convnext_model, train_ld, val_ld,
    lr=CONVNEXT_CFG['lr'], weight_decay=CONVNEXT_CFG['weight_decay'],
    model_name='ConvNeXt-Tiny',
    epochs=CONVNEXT_CFG['epochs'], patience=CONVNEXT_CFG['patience'],
    label_smoothing=0.1
)

torch.save(convnext_model.state_dict(), f'{SAVE_DIR}/convnext_best.pth')

# ══════════════════════════════════════════════════════════════════════════════
# ADIM 11 — Eğitim grafikleri (her iki model)
# ══════════════════════════════════════════════════════════════════════════════
def plot_history(hist, model_name, save_path):
    ep = range(1, len(hist['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'{model_name} — Eğitim Grafikleri', fontsize=14, fontweight='bold')

    ax1.plot(ep, hist['train_loss'], 'b-', label='Train Loss')
    ax1.plot(ep, hist['val_loss'],   'r-', label='Val Loss')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ep, hist['train_acc'], 'b-', label='Train Acc')
    ax2.plot(ep, hist['val_acc'],   'r-', label='Val Acc')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Grafik kaydedildi: {save_path}')

plot_history(custom_hist,   CUSTOM_CNN_NAME,   f'{SAVE_DIR}/custom_cnn_history.png')
plot_history(convnext_hist, 'ConvNeXt-Tiny', f'{SAVE_DIR}/convnext_history.png')

# ══════════════════════════════════════════════════════════════════════════════
# ADIM 12 — Test değerlendirmesi (her iki model)
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_and_report(model, test_ld, model_name, save_dir):
    preds, trues = predict(model, test_ld)

    acc  = accuracy_score(trues, preds) * 100
    prec = precision_score(trues, preds, zero_division=0) * 100
    rec  = recall_score(trues, preds, zero_division=0) * 100
    f1   = f1_score(trues, preds, zero_division=0) * 100

    print(f'\n{"="*60}')
    print(f'  {model_name} — TEST SONUÇLARI')
    print(f'{"="*60}')
    print(f'  Accuracy  : {acc:.2f}%')
    print(f'  Precision : {prec:.2f}%')
    print(f'  Recall    : {rec:.2f}%')
    print(f'  F1-Score  : {f1:.2f}%')
    print()
    print(classification_report(trues, preds, target_names=CLASS_NAMES))

    # Confusion matrix
    cm = confusion_matrix(trues, preds)
    _, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f'{model_name}\nConfusion Matrix', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    cm_path = f'{save_dir}/{model_name.replace(" ", "_").replace("-", "_")}_confusion_matrix.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix kaydedildi: {cm_path}')

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

custom_metrics   = evaluate_and_report(custom_model,   test_ld, CUSTOM_CNN_NAME,    SAVE_DIR)
convnext_metrics = evaluate_and_report(convnext_model, test_ld, 'ConvNeXt-Tiny', SAVE_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# ADIM 13 — Karşılaştırma tablosu
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print('  MODEL KARŞILAŞTIRMA TABLOSU (Test Seti)')
print(f'{"="*60}')
print(f'  {"Metrik":<12} {"Custom CNN":>12} {"ConvNeXt":>12}')
print(f'  {"-"*38}')
for key in ['accuracy', 'precision', 'recall', 'f1']:
    print(f'  {key:<12} {custom_metrics[key]:>11.2f}%  {convnext_metrics[key]:>11.2f}%')

# JSON kaydet
results = {
    'custom_cnn'  : custom_metrics,
    'convnext'    : convnext_metrics,
    'hyperparams' : {
        'custom_cnn': CUSTOM_CFG,
        'convnext'  : CONVNEXT_CFG,
        'batch_size': BATCH_SIZE,
    }
}
with open(f'{SAVE_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nTüm dosyalar kaydedildi: {SAVE_DIR}')
print('  custom_cnn_best.pth')
print('  convnext_best.pth')
print('  custom_cnn_history.png')
print('  convnext_history.png')
print('  Custom_CNN_confusion_matrix.png')
print('  ConvNeXt_Tiny_confusion_matrix.png')
print('  results.json')
print('\nEğitim tamamlandı!')