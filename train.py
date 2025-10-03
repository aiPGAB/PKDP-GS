#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from model import create_model, create_optimizer, create_loss_function
from log import log, INFO, WARNING, ERROR
from config import parse_args
from scipy.stats import pearsonr
import optuna
import time
import functools
from utils import get_output_prefix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from optuna.samplers import TPESampler
import re
import sys
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_training_data(args):
    """Load training and testing data from files"""
    try:
        train_phe = pd.read_csv(args.train_phe, header=0, index_col=0)

        if args.pnum is None:
            phenotype_name = train_phe.columns[0]
            log(INFO, f"No phenotype specified, using first column: {phenotype_name}")
        elif isinstance(args.pnum, int):
            phenotype_name = train_phe.columns[args.pnum]
        else:
            phenotype_name = args.pnum

        log(INFO, f"Working with phenotype: {phenotype_name}")

        geno = pd.read_csv(args.geno, header=0, index_col=0)
        geno.columns = [re.sub(r'[^0-9a-zA-Z_]', '_', col) for col in geno.columns]
        feature_names = geno.columns.tolist()

        train_samples = train_phe.index[~train_phe[phenotype_name].isna()].tolist()
        log(INFO, f"Found {len(train_samples)} non-NA training samples")

        train_y = train_phe.loc[train_samples, phenotype_name]
        train_X = geno.loc[train_samples]

        if len(train_X) != len(train_samples):
            log(WARNING, f"Only {len(train_X)} of {len(train_samples)} training samples found in genotype data")

        X_train_data = train_X.reset_index(drop=True)
        y_train_data = train_y.reset_index(drop=True)

        log(INFO, f"Training set: {X_train_data.shape[0]} samples, {X_train_data.shape[1]} features")

        X_train = X_train_data.values.reshape(-1, 1, X_train_data.shape[1])
        y_train = y_train_data.values
        
        if hasattr(args, 'adjust_encoding') and args.adjust_encoding:
            log(INFO, "Adjusting genotype encoding from {0,1,2} to {-1,0,1}")
            X_train = X_train - 1
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        return X_train_tensor, y_train_tensor, feature_names, phenotype_name

    except Exception as e:
        log(ERROR, f"Error loading training data: {str(e)}")
        raise

def train_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_epoch(model, test_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def visualize_training_progress(train_losses, test_losses, output_path, prefix):
    plt.figure(figsize=(12, 7), dpi=300)
    epochs = range(1, len(train_losses) + 1)
    
    cm = plt.cm.get_cmap('viridis')
    
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    smooth_window = min(5, len(train_losses)//10) if len(train_losses) > 20 else 1
    
    if smooth_window > 1:
        train_smooth = smooth(train_losses, smooth_window)
        test_smooth = smooth(test_losses, smooth_window)
        
        plt.plot(epochs, train_smooth, '-', color='#66C2A5', linewidth=2.5, label='Train Loss (Smoothed)')
        plt.plot(epochs, test_smooth, '-', color='#FC8D62', linewidth=2.5, label='Test Loss (Smoothed)')
        
        plt.plot(epochs, train_losses, 'o', color='#66C2A5', alpha=0.4, markersize=4)
        plt.plot(epochs, test_losses, 'o', color='#FC8D62', alpha=0.4, markersize=4)
    else:
        plt.plot(epochs, train_losses, 'o-', color='#66C2A5', linewidth=2, label='Train Loss', markersize=5)
        plt.plot(epochs, test_losses, 'o-', color='#FC8D62', linewidth=2, label='Test Loss', markersize=5)
    
    min_train_idx = np.argmin(train_losses)
    min_test_idx = np.argmin(test_losses)
    
    plt.plot(min_train_idx+1, train_losses[min_train_idx], 'o', color='#66C2A5', 
             markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    plt.plot(min_test_idx+1, test_losses[min_test_idx], 'o', color='#FC8D62', 
             markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    
    plt.annotate(f'Min: {train_losses[min_train_idx]:.4f}', 
                 xy=(min_train_idx+1, train_losses[min_train_idx]),
                 xytext=(min_train_idx+1+2, train_losses[min_train_idx]),
                 fontsize=9, color='#66C2A5')
    
    plt.annotate(f'Min: {test_losses[min_test_idx]:.4f}', 
                 xy=(min_test_idx+1, test_losses[min_test_idx]),
                 xytext=(min_test_idx+1+2, test_losses[min_test_idx]),
                 fontsize=9, color='#FC8D62')
    
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Testing Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='upper right')
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{prefix}_loss_curve.png"), bbox_inches='tight')
    plt.close()


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def compute_pearson_correlation(y_true, y_pred):
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return float('nan')
    return float(pearsonr(y_true.flatten(), y_pred.flatten())[0])

def create_model_with_config(input_length, opts, feature_names, device):
    return create_model(
        input_length=input_length, 
        out_channels1=opts.main_channels[0] if hasattr(opts, 'main_channels') and opts.main_channels else None,
        out_channels2=opts.main_channels[1] if hasattr(opts, 'main_channels') and len(opts.main_channels) > 1 else None,
        out_channels3=opts.main_channels[2] if hasattr(opts, 'main_channels') and len(opts.main_channels) > 2 else None,
        prior_channels1=opts.prior_channels[0] if hasattr(opts, 'prior_channels') and opts.prior_channels else None,
        prior_channels2=opts.prior_channels[1] if hasattr(opts, 'prior_channels') and len(opts.prior_channels) > 1 else None,
        prior_channels3=opts.prior_channels[2] if hasattr(opts, 'prior_channels') and len(opts.prior_channels) > 2 else None,
        fc_layers=len(opts.fc_units) if hasattr(opts, 'fc_units') and opts.fc_units else None,
        fc_units=opts.fc_units if hasattr(opts, 'fc_units') else None,
        kernel_size=opts.conv_kernel_size if hasattr(opts, 'conv_kernel_size') else None,
        prior_kernel_size=opts.prior_kernel_size if hasattr(opts, 'prior_kernel_size') else None,
        dropout_prob=opts.dropout if hasattr(opts, 'dropout') else None,
        prior_features=opts.prior_features, 
        feature_names=feature_names
    ).to(device)


def optimize_hyperparameters(X_train, y_train, opts, device, feature_names, n_folds=5):
    
    sampler = TPESampler(seed=opts.seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    def tune_hyperparameters(trial):
        learning_rate = trial.suggest_float('lr', 1e-5, 1e-2, log=True)

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=opts.seed)
        fold_val_losses = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            X_train_fold = X_train[train_idx]
            X_val_fold = X_train[val_idx]
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]

            model = create_model_with_config(X_train.shape[2], opts, feature_names, device)
            
            optimizer = create_optimizer(model, learning_rate, opts.optimizer)
            loss_fn = create_loss_function()
            
            train_loader = DataLoader(TensorDataset(X_train_fold, y_train_fold), batch_size=opts.batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val_fold, y_val_fold), batch_size=opts.batch_size, shuffle=False)
            
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
            early_stopping = EarlyStopping(patience=10)

            best_fold_val_loss = float('inf')
            for epoch in range(opts.epochs):
                train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
                val_loss = evaluate_epoch(model, val_loader, loss_fn, device)
                
                scheduler.step(val_loss)
                early_stopping(val_loss)
                
                if val_loss < best_fold_val_loss:
                    best_fold_val_loss = val_loss
                    
                if early_stopping.stop:
                    break
            
            fold_val_losses.append(best_fold_val_loss)
            
            del model, optimizer, train_loader, val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        mean_val_loss = np.mean(fold_val_losses)
        return mean_val_loss

    study.optimize(tune_hyperparameters, n_trials=opts.optuna_trials, show_progress_bar=True)
    best_params = study.best_params
    
    return best_params

def run_train(opts, X_train, y_train, device, feature_names, cv_folds=10):
    cv_folds = max(0, min(10, cv_folds))
    time_start = time.time()
    prefix = get_output_prefix(opts)

    best_params = optimize_hyperparameters(X_train, y_train, opts, device, feature_names, n_folds=5)
    
    log(INFO, f"Best hyperparameters: {best_params}")

    if cv_folds == 0:
        model = create_model_with_config(X_train.shape[2], opts, feature_names, device)
        log(INFO, f"Model architecture:\n{model}")

        optimizer = create_optimizer(model, best_params['lr'], opts.optimizer)
        loss_fn = create_loss_function()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=opts.batch_size, shuffle=True)

        early_stopping = EarlyStopping(patience=10) if opts.early_stop else None
        best_train_loss = float('inf')
        best_train_corr = -float('inf')
        best_state_dict = None
        train_losses = []

        for epoch in range(opts.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
            with torch.no_grad():
                y_train_pred = model(X_train.to(device)).cpu().numpy()
                train_corr = compute_pearson_correlation(y_train.numpy(), y_train_pred)

            train_losses.append(train_loss)

            if train_corr > best_train_corr:
                best_train_corr = train_corr
                best_train_loss = train_loss
                best_state_dict = model.state_dict()

            scheduler.step(train_loss)
            if early_stopping:
                early_stopping(train_loss)
                if early_stopping.stop:
                    log(INFO, f"Early stopping at epoch {epoch + 1}")
                    break

            log(INFO, f"Epoch {epoch + 1}: Train Corr={train_corr:.4f}, Train Loss={train_loss:.4f}, "
                      f"Best Corr={best_train_corr:.4f}")

        model.load_state_dict(best_state_dict)
        
    else:
        log(INFO, f"Using {cv_folds}-fold cross-validation for model training")
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=opts.seed)
        best_fold_corr = -float('inf')
        best_model_state = None
        best_train_corr = -float('inf')
        best_train_loss = float('inf')
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            X_train_fold = X_train[train_idx]
            X_val_fold = X_train[val_idx]
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]

            model = create_model_with_config(X_train.shape[2], opts, feature_names, device)
            optimizer = create_optimizer(model, best_params['lr'], opts.optimizer)
            loss_fn = create_loss_function()
            
            train_loader = DataLoader(TensorDataset(X_train_fold, y_train_fold), batch_size=opts.batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val_fold, y_val_fold), batch_size=opts.batch_size, shuffle=False)
            
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
            early_stopping = EarlyStopping(patience=10)

            fold_best_val_corr = -float('inf')
            fold_best_train_corr = -float('inf')
            fold_best_train_loss = float('inf')
            fold_best_val_loss = float('inf')
            fold_best_state = None
            
            for epoch in range(opts.epochs):
                train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
                val_loss = evaluate_epoch(model, val_loader, loss_fn, device)
                
                with torch.no_grad():
                    val_pred = model(X_val_fold.to(device)).cpu().numpy()
                    val_corr = compute_pearson_correlation(y_val_fold.numpy(), val_pred)
                    
                    train_pred = model(X_train_fold.to(device)).cpu().numpy()
                    train_corr = compute_pearson_correlation(y_train_fold.numpy(), train_pred)
                
                # Log metrics for each epoch
                log(INFO, f"Fold {fold + 1}, Epoch {epoch + 1}: Train Corr={train_corr:.4f}, Train Loss={train_loss:.4f}, Val Corr={val_corr:.4f}, Val Loss={val_loss:.4f}")
                
                if val_corr > fold_best_val_corr:
                    fold_best_val_corr = val_corr
                    fold_best_train_corr = train_corr
                    fold_best_train_loss = train_loss
                    fold_best_val_loss = val_loss
                    fold_best_state = model.state_dict().copy()
                
                scheduler.step(val_loss)
                early_stopping(val_loss)
                if early_stopping.stop:
                    log(INFO, f"Fold {fold + 1}: Early stopping at epoch {epoch + 1}")
                    break
            
            if fold_best_val_corr > best_fold_corr:
                best_fold_corr = fold_best_val_corr
                best_model_state = fold_best_state
                best_train_corr = fold_best_train_corr
                best_train_loss = fold_best_train_loss
            
            log(INFO, f"Fold {fold + 1} Best Train Corr={fold_best_train_corr:.4f}, Best Train Loss={fold_best_train_loss:.4f}, Best Val Corr={fold_best_val_corr:.4f}, Best Val Loss={fold_best_val_loss:.4f}")
            
            del model, optimizer, train_loader, val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        model = create_model_with_config(X_train.shape[2], opts, feature_names, device)
        model.load_state_dict(best_model_state)
        log(INFO, f"Best fold validation correlation: {best_fold_corr:.4f}")

    model_save_path = os.path.join(opts.output_path, f"{prefix}_best_model.pth")
    torch.save(model, model_save_path)

    with torch.no_grad():
        y_train_pred = model(X_train.to(device)).cpu().numpy()
        train_corr = compute_pearson_correlation(y_train.cpu().numpy(), y_train_pred)
        results = {'time': int(time.time() - time_start), 'train_corr': best_train_corr, 'train_loss': best_train_loss, 'prefix': prefix}
        
    #     visualize_training_results(y_train.cpu().numpy(), y_train_pred, opts.output_path, prefix)

    # pd.DataFrame([results]).to_csv(os.path.join(opts.output_path, f"{prefix}_results.csv"), index=False)

    log(INFO, f"Best training correlation: {best_train_corr:.4f}")
    log(INFO, f"Model saved to {model_save_path}")
    log(INFO, f"Training completed with prefix: {prefix}")

    # with torch.no_grad():
    #     y_train_pred = model(X_train.to(device)).cpu().numpy()
    #     train_metrics = compute_evaluation_metrics(y_train.cpu().numpy(), y_train_pred)
    #     print_evaluation_metrics(train_metrics, "Training Set")
    
    return model, results

def visualize_training_results(y_train, y_pred_train, output_path, prefix):
    plot_combined(y_pred_train.flatten(), y_train.flatten(), 
                 pearsonr(y_train.flatten(), y_pred_train.flatten())[0]**2,
                 "Training Set Predictions", 
                 os.path.join(output_path, f"{prefix}_train_only_combined.png"))

def plot_combined(predicted, observed, r2, title, output_path=None):
    """Generate and save combined plots showing model performance with improved styling."""
    fig = plt.figure(figsize=(16, 6))

    # Density comparison subplot
    ax1 = fig.add_subplot(1, 2, 1)
    kde_predicted = gaussian_kde(predicted)
    kde_true = gaussian_kde(observed)

    # Plot density with Set2 colors
    min_value = min(min(predicted), min(observed))
    max_value = max(max(predicted), max(observed))
    x = np.linspace(min_value, max_value, 1000)

    ax1.fill_between(x, kde_predicted(x), color='#66C2A5', alpha=0.3, label='Predicted values')
    ax1.fill_between(x, kde_true(x), color='#FC8D62', alpha=0.3, label='Observed values')
    ax1.plot(x, kde_predicted(x), color='#66C2A5')
    ax1.plot(x, kde_true(x), color='#FC8D62', linestyle='dashed')

    ax1.set_xlabel('Values', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Predicted vs Observed Density', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Scatter plot subplot
    ax2 = fig.add_subplot(1, 2, 2)
    rmse = np.sqrt(mean_squared_error(predicted, observed))
    pcc, _ = pearsonr(predicted, observed)

    # Plot scatter with density coloring
    xy = np.vstack([predicted, observed])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    predicted_sorted, observed_sorted, z_sorted = predicted[idx], observed[idx], z[idx]

    scatter = ax2.scatter(observed_sorted, predicted_sorted, c=z_sorted, s=12, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, ax=ax2, label='Density')

    # Add 1:1 reference line
    min_val = min(predicted.min(), observed.min())
    max_val = max(predicted.max(), observed.max())
    ax2.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', label='1:1 Line')

    ax2.text(0.05, 0.95, f'RMSE: {rmse:.3f}\nPCC: {pcc:.3f}\nR²: {r2:.3f}',
             transform=ax2.transAxes, ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    ax2.set_xlabel('Observed Values', fontsize=12)
    ax2.set_ylabel('Predicted Values', fontsize=12)
    ax2.set_title(title, fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        log(INFO, f"Plot saved to {output_path}")

    plt.close(fig)

def compute_evaluation_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    corr = pearsonr(y_true.flatten(), y_pred.flatten())[0]
    r2 = corr ** 2
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'PCC': corr}

def print_evaluation_metrics(metrics, dataset_name):
    from log import log_section_start, log_section_end, log_info
    log_section_start(f"{dataset_name} Evaluation")
    for name, value in metrics.items():
        log(INFO, f"{name}: {value:.6f}")
    log_section_end("-----------------------")

def train():
    opts = parse_args()
    if opts.seed is not None:
        set_random_seed(opts.seed)
        log(INFO, f"Using fixed random seed: {opts.seed}")
    else:
        log(INFO, "No seed provided, running with random state")

    X_train, y_train, feature_names, phenotype_name = load_training_data(opts)
    device = torch.device(opts.device if torch.cuda.is_available() else 'cpu')

    if opts.prior_features is not None:
        prior_ids = opts.prior_features.split()
        clean_prior_ids = []
        for pid in prior_ids:
            if pid.isdigit():
                clean_prior_ids.append(pid)
            else:
                clean_pid = re.sub(r'[^0-9a-zA-Z_]', '_', pid)
                clean_prior_ids.append(clean_pid)
        opts.prior_features = ' '.join(clean_prior_ids)
        
        # Validate prior features and log results
        name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        matched_indices = []
        missing_features = []
        invalid_indices = []
        
        for pid in clean_prior_ids:
            try:
                if pid.isdigit():
                    idx = int(pid)
                    if 0 <= idx < len(feature_names):
                        matched_indices.append(idx)
                    else:
                        invalid_indices.append(pid)
                else:
                    idx = name_to_idx.get(pid, -1)
                    if idx >= 0:
                        matched_indices.append(idx)
                    else:
                        missing_features.append(pid)
            except ValueError:
                missing_features.append(pid)
        
        if missing_features:
            log(WARNING, f"The following prior features were not found in feature names: {missing_features}")
        if invalid_indices:
            log(WARNING, f"The following prior feature indices are out of range [0, {len(feature_names)-1}]: {invalid_indices}")
        
        if len(matched_indices) == 0 and len(prior_ids) > 0:
            log(WARNING, "None of the provided prior features matched!")
        else:
            log(INFO, f"Successfully matched {len(matched_indices)} out of {len(prior_ids)} prior features")
            
            if matched_indices:
                
                all_indices = set(range(len(feature_names)))
                prior_indices_set = set(matched_indices)
                non_prior_indices = sorted(list(all_indices - prior_indices_set))
                
                new_order = non_prior_indices + sorted(matched_indices)
                
                X_train_reordered = X_train[:, :, new_order]
                X_train = X_train_reordered
                
                feature_names_reordered = [feature_names[i] for i in new_order]
                feature_names = feature_names_reordered
                
                new_prior_indices = list(range(len(non_prior_indices), len(feature_names)))
                
                updated_prior_features = [str(i) for i in new_prior_indices]
                opts.prior_features = ' '.join(updated_prior_features)
                
                log(INFO, f"Total features: {len(feature_names)}, Prior features: {len(new_prior_indices)}")
            
        log(INFO, f"Enabled double-channel mode with {len(prior_ids)} prior features")
    else:
        log(INFO, "No prior features specified, running in single-channel mode")
    
    cv_folds = getattr(opts, 'cv_folds', 10)
    cv_folds = max(0, min(10, cv_folds))

    return run_train(opts, X_train, y_train, device, feature_names, cv_folds)

if __name__ == "__main__":
    train()