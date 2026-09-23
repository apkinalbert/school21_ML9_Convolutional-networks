import copy
from typing import Callable

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from model import my_LeNet


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = True,
    mix: Callable | None = None,
) -> float:
    """Run one training epoch and return the average loss."""
    model.train()
    amp = use_amp and device.type == "cuda"

    total_loss = torch.zeros((), device=device)
    total_samples = 0

    for images, targets in train_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mix is not None:
            images, targets = mix(images, targets)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(images)
            loss = loss_fn(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.detach() * batch_size
        total_samples += batch_size

    return total_loss.item() / total_samples


def validate(
    model: nn.Module,
    valid_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    tta: bool = False,
) -> tuple[float, float]:
    """Run one pass over the validation set and return (loss, ROC AUC).

    If tta=True, each batch is also predicted flipped horizontally, and the
    two predictions are averaged (test-time augmentation).
    """
    model.eval()
    amp = use_amp and device.type == "cuda"

    total_loss = torch.zeros((), device=device)
    total_samples = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in valid_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(images)
                loss = loss_fn(logits, targets)

            probs = torch.softmax(logits.float(), dim=1)

            if tta:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    logits_flipped = model(torch.flip(images, dims=[3]))
                probs_flipped = torch.softmax(logits_flipped.float(), dim=1)
                probs = (probs + probs_flipped) / 2

            all_targets.append(targets)
            all_probs.append(probs)

            batch_size = images.shape[0]
            total_loss += loss.detach() * batch_size
            total_samples += batch_size

    valid_loss = total_loss.item() / total_samples

    final_targets = torch.cat(all_targets).cpu().numpy()
    final_probs = torch.cat(all_probs).cpu().numpy()
    valid_roc = roc_auc_score(
        final_targets, final_probs, multi_class="ovr", average="macro"
    )

    return valid_loss, valid_roc


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    use_amp: bool = True,
    patience: int = 4,
    verbose: int = 2,
    mix: Callable | None = None,
) -> dict[str, list[float]]:
    """Train for up to `epochs` epochs with early stopping on validation ROC AUC.

    Keeps and restores the model weights from the best epoch.
    """
    model.to(device)

    results = {"train_loss": [], "valid_loss": [], "valid_roc": []}

    best_roc = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    best_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            use_amp,
            mix)

        valid_loss, valid_roc = validate(
            model,
            valid_loader,
            loss_fn,
            device,
            use_amp
        )

        if valid_roc > best_roc:
            best_roc = valid_roc
            best_epoch = epoch
            epochs_without_improvement = 0

            best_state = copy.deepcopy(model.state_dict())

        else:
            epochs_without_improvement += 1

        results["train_loss"].append(train_loss)
        results["valid_loss"].append(valid_loss)
        results["valid_roc"].append(valid_roc)

        if verbose == 2:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"valid loss: {valid_loss:.4f} | "
                f"valid ROC AUC: {valid_roc:.4f}"
            )

        if epochs_without_improvement >= patience:
            if verbose == 2:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)

    results["best_roc"] = best_roc
    results["best_epoch"] = best_epoch

    return results


def run_experiment(
    name: str,
    model_config: dict,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    patience: int = 4,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    verbose: int = 2,
    use_amp: bool = True,
    mix: Callable | None = None,
) -> tuple[dict, dict, nn.Module]:
    """Build a my_LeNet from `model_config`, train it, and summarize the result."""
    if verbose == 2 or verbose == 1:
        print(f"\n{'=' * 60}")
        print(f"Experiment: {name}")
        print(f"{'=' * 60}")

    model = my_LeNet(**model_config)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history = fit(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        patience=patience,
        verbose=verbose,
        use_amp=use_amp,
        mix=mix
    )

    summary = {
        "experiment": name,
        **model_config,
        "lr" : lr,
        "weight_decay": weight_decay,
        "best_roc": history["best_roc"],
        "best_epoch": history["best_epoch"] + 1,
        "epochs_completed": len(history["valid_roc"]),
        "final_train_loss": history["train_loss"][-1],
        "final_valid_loss": history["valid_loss"][-1],
    }

    return summary, history, model
