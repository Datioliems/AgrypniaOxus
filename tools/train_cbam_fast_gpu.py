"""Fast GPU CBAM training with PyTorch CUDA, then Keras/TFLite export.

The original CBAM notebook uses TensorFlow/Keras. On native Windows,
TensorFlow 2.11+ cannot use NVIDIA CUDA, while this project environment has
PyTorch CUDA. This script trains the same small CNN+CBAM idea with PyTorch on
the RTX GPU, initializes shared CNN layers from the existing baseline Keras
models when available, then copies weights into an equivalent Keras model for
TFLite export.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Task:
    name: str
    data_dir: Path
    out_dir: Path
    tflite_out: Path
    baseline_keras: Path | None
    expected_classes: list[str]


TASKS = {
    "eye": Task(
        name="eye",
        data_dir=ROOT / "dataset_mrl",
        out_dir=ROOT / "outputs" / "cnn_eye_cbam",
        tflite_out=ROOT / "app" / "src" / "main" / "assets" / "drowsiness_cbam.tflite",
        baseline_keras=ROOT / "outputs" / "training" / "drowsiness_model.keras",
        expected_classes=["eyes_closed", "eyes_open"],
    ),
    "yawn": Task(
        name="yawn",
        data_dir=ROOT / "dataset_yawn",
        out_dir=ROOT / "outputs" / "cnn_yawn_cbam",
        tflite_out=ROOT / "app" / "src" / "main" / "assets" / "yawn_cbam.tflite",
        baseline_keras=ROOT / "outputs" / "cnn_yawn" / "best_model.keras",
        expected_classes=["no_yawn", "yawn"],
    ),
}


def load_friend_params() -> dict:
    params_path = ROOT / "data_pipeline" / "best_params.json"
    if params_path.exists():
        return json.loads(params_path.read_text(encoding="utf-8"))
    return {}


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        hidden = max(channels // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.fc2(self.relu(self.fc1(self.avg_pool(x).view(b, c))))
        mx = self.fc2(self.relu(self.fc1(self.max_pool(x).view(b, c))))
        scale = torch.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


class CbamBlock(nn.Module):
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        self.channel = ChannelAttention(channels, ratio)
        self.spatial = SpatialAttention()

    def forward(self, x):
        return self.spatial(self.channel(x))


class FastCbamCNN(nn.Module):
    def __init__(self, num_classes: int = 2, ratio: int = 8, dropout: float = 0.30):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-3)
        self.cbam1 = CbamBlock(32, ratio)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-3)
        self.cbam2 = CbamBlock(64, ratio)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-3)
        self.cbam3 = CbamBlock(128, ratio)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.cbam1(self.bn1(torch.relu(self.conv1(x)))))
        x = self.pool2(self.cbam2(self.bn2(torch.relu(self.conv2(x)))))
        x = self.cbam3(self.bn3(torch.relu(self.conv3(x))))
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def load_baseline_weights(model: FastCbamCNN, baseline_path: Path | None) -> bool:
    if baseline_path is None or not baseline_path.exists():
        return False
    import tensorflow as tf

    keras_model = tf.keras.models.load_model(str(baseline_path), compile=False)
    conv_i = 0
    bn_i = 0
    dense_loaded = False

    convs = [model.conv1, model.conv2, model.conv3]
    bns = [model.bn1, model.bn2, model.bn3]

    with torch.no_grad():
        for layer in keras_model.layers:
            cls = layer.__class__.__name__
            weights = layer.get_weights()
            if cls == "Conv2D" and conv_i < len(convs):
                w, b = weights
                tw = torch.from_numpy(np.transpose(w, (3, 2, 0, 1)))
                tb = torch.from_numpy(b)
                if tuple(tw.shape) != tuple(convs[conv_i].weight.shape) or tuple(tb.shape) != tuple(convs[conv_i].bias.shape):
                    return False
                convs[conv_i].weight.copy_(tw)
                convs[conv_i].bias.copy_(tb)
                conv_i += 1
            elif cls == "BatchNormalization" and bn_i < len(bns):
                gamma, beta, mean, var = weights
                if tuple(gamma.shape) != tuple(bns[bn_i].weight.shape):
                    return False
                bns[bn_i].weight.copy_(torch.from_numpy(gamma))
                bns[bn_i].bias.copy_(torch.from_numpy(beta))
                bns[bn_i].running_mean.copy_(torch.from_numpy(mean))
                bns[bn_i].running_var.copy_(torch.from_numpy(var))
                bn_i += 1
            elif cls == "Dense" and not dense_loaded:
                w, b = weights
                tw = torch.from_numpy(w.T)
                tb = torch.from_numpy(b)
                if tuple(tw.shape) != tuple(model.fc.weight.shape) or tuple(tb.shape) != tuple(model.fc.bias.shape):
                    return False
                model.fc.weight.copy_(tw)
                model.fc.bias.copy_(tb)
                dense_loaded = True

    return conv_i == 3 and bn_i == 3 and dense_loaded


def make_balanced_subset(dataset, fraction: float, max_per_class: int | None, seed: int, split_name: str):
    if fraction >= 1.0 and not max_per_class:
        return dataset, {
            "split": split_name,
            "mode": "full",
            "total": len(dataset),
            "per_class": {
                class_name: sum(1 for _, target in dataset.samples if target == class_idx)
                for class_name, class_idx in dataset.class_to_idx.items()
            },
        }

    targets = np.array([target for _, target in dataset.samples])
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    per_class: dict[str, int] = {}
    for class_name, class_idx in dataset.class_to_idx.items():
        idxs = np.flatnonzero(targets == class_idx)
        rng.shuffle(idxs)
        take = len(idxs)
        if fraction < 1.0:
            take = max(1, int(round(len(idxs) * fraction)))
        if max_per_class:
            take = min(take, max_per_class)
        take = min(take, len(idxs))
        selected.extend(idxs[:take].tolist())
        per_class[class_name] = int(take)

    rng.shuffle(selected)
    return Subset(dataset, selected), {
        "split": split_name,
        "mode": "balanced_subset",
        "fraction": fraction,
        "max_per_class": max_per_class,
        "total": len(selected),
        "per_class": per_class,
    }


def make_loaders(task: Task, args, params: dict):
    pp = params.get("preprocess", {})
    aug = pp.get("augmentation", {})
    image_size = int(pp.get("image_size", 64))
    train_steps = [
        transforms.Resize((image_size, image_size)),
    ]
    if aug.get("random_flip") == "horizontal":
        train_steps.append(transforms.RandomHorizontalFlip(p=0.5))
    if "random_rotation" in aug:
        train_steps.append(transforms.RandomRotation(degrees=float(aug["random_rotation"]) * 360.0))
    if "random_zoom" in aug:
        z = float(aug["random_zoom"])
        train_steps.append(transforms.RandomResizedCrop((image_size, image_size), scale=(1.0 - z, 1.0), ratio=(0.95, 1.05)))
    if "random_contrast" in aug:
        c = float(aug["random_contrast"])
        train_steps.append(transforms.ColorJitter(contrast=c))
    train_steps.append(transforms.ToTensor())

    train_tfm = transforms.Compose(train_steps)
    eval_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    train_full = datasets.ImageFolder(task.data_dir / "train", transform=train_tfm)
    val_full = datasets.ImageFolder(task.data_dir / "val", transform=eval_tfm)
    test_full = datasets.ImageFolder(task.data_dir / "test", transform=eval_tfm)
    classes = train_full.classes
    if classes != task.expected_classes:
        raise RuntimeError(f"{task.name}: class order {classes} != {task.expected_classes}")
    train_ds, train_info = make_balanced_subset(
        train_full, args.train_fraction, args.max_train_per_class, args.seed, "train"
    )
    val_ds, val_info = make_balanced_subset(
        val_full, args.val_fraction, args.max_eval_per_class, args.seed + 1, "val"
    )
    test_ds, test_info = make_balanced_subset(
        test_full, args.test_fraction, args.max_eval_per_class, args.seed + 2, "test"
    )
    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
    return classes, train_loader, val_loader, test_loader, {
        "train": train_info,
        "val": val_info,
        "test": test_info,
    }


@torch.inference_mode()
def evaluate(model, loader, device, class_names: list[str]):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    cm = torch.zeros(2, 2, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(1)
        total += y.numel()
        correct += (pred == y).sum().item()
        loss_sum += loss.item() * y.numel()
        for t, p in zip(y.cpu(), pred.cpu()):
            cm[int(t), int(p)] += 1
    per_class = {}
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum().item() - tp)
        fn = int(cm[i, :].sum().item() - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[name] = {
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "support": int(cm[i, :].sum().item()),
        }
    return {
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def train(task: Task, args):
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available; cannot do fast GPU CBAM training.")

    task.out_dir.mkdir(parents=True, exist_ok=True)
    task.tflite_out.parent.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    params = load_friend_params()
    classes, train_loader, val_loader, test_loader, data_selection = make_loaders(task, args, params)

    model = FastCbamCNN(num_classes=len(classes), ratio=args.cbam_ratio, dropout=args.dropout)
    baseline_loaded = False if args.skip_baseline_init else load_baseline_weights(model, task.baseline_keras)
    model = model.to(device, memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=args.amp)

    best_state = None
    best_val = -1.0
    patience_left = args.patience
    history = []
    start = time.perf_counter()

    print(f"\n=== {task.name.upper()} CBAM FAST GPU ===")
    print(f"device={torch.cuda.get_device_name(0)} batch={args.batch_size} amp={args.amp}")
    print(f"classes={classes} baseline_init={baseline_loaded}")
    print(f"data_selection={json.dumps(data_selection, ensure_ascii=False)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = correct = 0
        loss_sum = 0.0
        epoch_start = time.perf_counter()
        num_batches = len(train_loader)
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=args.amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pred = logits.argmax(1)
            total += y.numel()
            correct += (pred == y).sum().item()
            loss_sum += loss.item() * y.numel()
            if args.progress_every and (batch_idx == 1 or batch_idx % args.progress_every == 0 or batch_idx == num_batches):
                partial = {
                    "model": f"DrowsyCNN_CBAM_{task.name}",
                    "status": "training_epoch",
                    "epoch": epoch,
                    "epochs_total": args.epochs,
                    "batch": batch_idx,
                    "batches_total": num_batches,
                    "train_accuracy_so_far": round(float(correct / max(total, 1)), 6),
                    "train_loss_so_far": round(float(loss_sum / max(total, 1)), 6),
                    "elapsed_seconds": round(time.perf_counter() - start, 1),
                    "device": torch.cuda.get_device_name(0),
                    "data_selection": data_selection,
                }
                (task.out_dir / "summary_running.json").write_text(
                    json.dumps(partial, indent=2), encoding="utf-8"
                )
                print(
                    f"epoch {epoch:02d}/{args.epochs} batch {batch_idx:03d}/{num_batches} "
                    f"train_acc_so_far={partial['train_accuracy_so_far']*100:.2f}% "
                    f"loss={partial['train_loss_so_far']:.4f}",
                    flush=True,
                )

        train_acc = correct / total
        train_loss = loss_sum / total
        validating_summary = {
            "model": f"DrowsyCNN_CBAM_{task.name}",
            "status": "epoch_train_completed_validating",
            "epoch": epoch,
            "epochs_total": args.epochs,
            "train_loss": round(float(train_loss), 6),
            "train_accuracy": round(float(train_acc), 6),
            "epoch_train_seconds": round(time.perf_counter() - epoch_start, 1),
            "elapsed_seconds": round(time.perf_counter() - start, 1),
            "device": torch.cuda.get_device_name(0),
            "data_selection": data_selection,
        }
        (task.out_dir / "summary_running.json").write_text(
            json.dumps(validating_summary, indent=2), encoding="utf-8"
        )
        print(
            f"epoch {epoch:02d}/{args.epochs} train done: "
            f"train_acc={train_acc*100:.2f}% train_loss={train_loss:.4f}; validating...",
            flush=True,
        )
        val = evaluate(model, val_loader, device, classes)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
        }
        history.append(row)
        (task.out_dir / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        running_summary = {
            "model": f"DrowsyCNN_CBAM_{task.name}",
            "status": "running_or_interrupted",
            "epochs_completed": epoch,
            "best_val_accuracy_so_far": round(float(max(r["val_accuracy"] for r in history)), 6),
            "last_epoch": row,
            "device": torch.cuda.get_device_name(0),
            "data_selection": data_selection,
            "note": "This file is updated after every epoch so interrupted runs keep progress.",
        }
        (task.out_dir / "summary_running.json").write_text(
            json.dumps(running_summary, indent=2), encoding="utf-8"
        )
        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"train_acc={train_acc*100:.2f}% val_acc={val['accuracy']*100:.2f}% "
            f"val_loss={val['loss']:.4f}"
        )

        if val["accuracy"] > best_val + args.min_delta:
            best_val = val["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, task.out_dir / "best_cbam_torch_state.pt")
            best_snapshot = dict(running_summary)
            best_snapshot["status"] = "best_checkpoint_saved"
            best_snapshot["best_epoch"] = epoch
            best_snapshot["best_val_accuracy"] = round(float(best_val), 6)
            (task.out_dir / "summary_best_checkpoint.json").write_text(
                json.dumps(best_snapshot, indent=2), encoding="utf-8"
            )
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stopping at epoch {epoch}")
                break

        if best_val >= args.target_acc:
            print(f"target val accuracy reached: {best_val*100:.2f}%")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test = evaluate(model, test_loader, device, classes)
    elapsed = time.perf_counter() - start
    print(f"test_acc={test['accuracy']*100:.2f}% elapsed={elapsed/60:.1f} min")

    torch.save(model.state_dict(), task.out_dir / "final_cbam_torch_state.pt")
    export_tflite_from_torch(task, model.cpu(), classes, args.cbam_ratio, args.dropout)

    summary = {
        "model": f"DrowsyCNN_CBAM_{task.name}",
        "classes": classes,
        "image_size": 64,
        "cbam_ratio": args.cbam_ratio,
        "dropout": args.dropout,
        "epochs_ran": len(history),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "baseline_initialized": baseline_loaded,
        "device": torch.cuda.get_device_name(0),
        "data_selection": data_selection,
        "best_val_accuracy": round(float(best_val), 6),
        "test_accuracy": round(float(test["accuracy"]), 6),
        "test_loss": round(float(test["loss"]), 6),
        "confusion_matrix": test["confusion_matrix"],
        "per_class": test["per_class"],
        "preprocess_source": "data_pipeline/best_params.json" if params else "script defaults",
        "augmentation": params.get("preprocess", {}).get("augmentation", {}),
        "tflite_path": str(task.tflite_out),
        "tflite_kb": round(task.tflite_out.stat().st_size / 1024, 1),
        "elapsed_seconds": round(elapsed, 1),
    }
    (task.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (task.out_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (task.out_dir / "class_names.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    return summary


def keras_cbam_model(num_classes: int, ratio: int, dropout: float):
    import tensorflow as tf

    def channel_attention(x, channels, block_name):
        hidden = max(channels // ratio, 1)
        avg = tf.keras.layers.GlobalAveragePooling2D(name=f"{block_name}_ca_avg")(x)
        mx = tf.keras.layers.GlobalMaxPooling2D(name=f"{block_name}_ca_max")(x)
        fc1 = tf.keras.layers.Dense(hidden, activation="relu", name=f"{block_name}_ca_fc1")
        fc2 = tf.keras.layers.Dense(channels, name=f"{block_name}_ca_fc2")
        avg = fc2(fc1(avg))
        mx = fc2(fc1(mx))
        scale = tf.keras.layers.Activation("sigmoid", name=f"{block_name}_ca_sigmoid")(
            tf.keras.layers.Add(name=f"{block_name}_ca_add")([avg, mx])
        )
        scale = tf.keras.layers.Reshape((1, 1, channels), name=f"{block_name}_ca_reshape")(scale)
        return tf.keras.layers.Multiply(name=f"{block_name}_ca_multiply")([x, scale])

    def spatial_attention(x, block_name):
        avg = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True), name=f"{block_name}_sa_avg")(x)
        mx = tf.keras.layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True), name=f"{block_name}_sa_max")(x)
        concat = tf.keras.layers.Concatenate(axis=-1, name=f"{block_name}_sa_concat")([avg, mx])
        scale = tf.keras.layers.Conv2D(1, 7, padding="same", use_bias=False, activation="sigmoid", name=f"{block_name}_sa_conv")(concat)
        return tf.keras.layers.Multiply(name=f"{block_name}_sa_multiply")([x, scale])

    def block(x, channels, block_name, pool=True):
        x = tf.keras.layers.Conv2D(channels, 3, padding="same", activation="relu", name=f"{block_name}_conv")(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-3, name=f"{block_name}_bn")(x)
        x = channel_attention(x, channels, block_name)
        x = spatial_attention(x, block_name)
        if pool:
            x = tf.keras.layers.MaxPooling2D(name=f"{block_name}_pool")(x)
        return x

    inputs = tf.keras.Input(shape=(64, 64, 3), name="input")
    x = block(inputs, 32, "b1", pool=True)
    x = block(x, 64, "b2", pool=True)
    x = block(x, 128, "b3", pool=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs, outputs, name="FastCbamCNN_KerasExport")


def set_keras_weights_from_torch(kmodel, tmodel: FastCbamCNN):
    modules = [
        ("b1", tmodel.conv1, tmodel.bn1, tmodel.cbam1),
        ("b2", tmodel.conv2, tmodel.bn2, tmodel.cbam2),
        ("b3", tmodel.conv3, tmodel.bn3, tmodel.cbam3),
    ]
    for name, conv, bn, cbam in modules:
        kmodel.get_layer(f"{name}_conv").set_weights([
            conv.weight.detach().numpy().transpose(2, 3, 1, 0),
            conv.bias.detach().numpy(),
        ])
        kmodel.get_layer(f"{name}_bn").set_weights([
            bn.weight.detach().numpy(),
            bn.bias.detach().numpy(),
            bn.running_mean.detach().numpy(),
            bn.running_var.detach().numpy(),
        ])
        kmodel.get_layer(f"{name}_ca_fc1").set_weights([
            cbam.channel.fc1.weight.detach().numpy().T,
            cbam.channel.fc1.bias.detach().numpy(),
        ])
        kmodel.get_layer(f"{name}_ca_fc2").set_weights([
            cbam.channel.fc2.weight.detach().numpy().T,
            cbam.channel.fc2.bias.detach().numpy(),
        ])
        kmodel.get_layer(f"{name}_sa_conv").set_weights([
            cbam.spatial.conv.weight.detach().numpy().transpose(2, 3, 1, 0),
        ])

    kmodel.get_layer("classifier").set_weights([
        tmodel.fc.weight.detach().numpy().T,
        tmodel.fc.bias.detach().numpy(),
    ])


def export_tflite_from_torch(task: Task, tmodel: FastCbamCNN, classes: list[str], ratio: int, dropout: float):
    import tensorflow as tf

    kmodel = keras_cbam_model(len(classes), ratio, dropout)
    set_keras_weights_from_torch(kmodel, tmodel)
    kmodel.save(task.out_dir / "best_model.keras")

    converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    export_copy = task.out_dir / task.tflite_out.name
    export_copy.write_bytes(tflite_bytes)
    try:
        task.tflite_out.write_bytes(tflite_bytes)
    except PermissionError:
        shutil.copy2(export_copy, task.tflite_out)

    interp = tf.lite.Interpreter(model_path=str(task.tflite_out))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"TFLite: {task.tflite_out}")
    print(f"  input={inp['shape']} {inp['dtype'].__name__}")
    print(f"  output={out['shape']} {out['dtype'].__name__}")
    if inp["shape"].tolist() != [1, 64, 64, 3] or inp["dtype"].__name__ != "float32":
        raise RuntimeError("Exported TFLite is not Android-compatible.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["eye", "yawn"], choices=sorted(TASKS))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--cbam-ratio", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--target-acc", type=float, default=0.985)
    parser.add_argument("--train-fraction", type=float, default=1.0, help="Use a balanced fraction of each training class.")
    parser.add_argument("--val-fraction", type=float, default=1.0, help="Use a balanced fraction of each validation class.")
    parser.add_argument("--test-fraction", type=float, default=1.0, help="Use a balanced fraction of each test class.")
    parser.add_argument("--max-train-per-class", type=int, default=0, help="Cap training images per class; 0 means no cap.")
    parser.add_argument("--max-eval-per-class", type=int, default=0, help="Cap validation/test images per class; 0 means no cap.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-baseline-init", action="store_true", help="Skip importing TensorFlow to warm-start from baseline Keras weights.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print and save in-epoch progress every N batches; 0 disables.")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    args.amp = not args.no_amp
    args.max_train_per_class = args.max_train_per_class or None
    args.max_eval_per_class = args.max_eval_per_class or None
    for name in ["train_fraction", "val_fraction", "test_fraction"]:
        value = getattr(args, name)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0 and <= 1")

    summaries = []
    for task_name in args.tasks:
        summaries.append(train(TASKS[task_name], args))
    print("\n=== CBAM FAST GPU DONE ===")
    for s in summaries:
        print(f"{s['model']}: val={s['best_val_accuracy']*100:.2f}% test={s['test_accuracy']*100:.2f}% tflite={s['tflite_kb']} KB")


if __name__ == "__main__":
    main()
