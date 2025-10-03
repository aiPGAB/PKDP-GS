#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from model import create_model
from config import parse_args
from log import log, INFO, WARNING, ERROR
from utils import get_output_prefix
import os
import re
import sys
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error

def load_prediction_data(args):
    """Load data for prediction"""
    try:
        geno_data = pd.read_csv(args.geno, header=0, index_col=0)
        geno_data.columns = [re.sub(r'[^0-9a-zA-Z_]', '_', col) for col in geno_data.columns]
        feature_names = geno_data.columns.tolist()
        
        y_data = None
        y_tensor = None
        phenotype_name = None
        valid_samples = geno_data.index.tolist()
        
        if args.test_phe:
            phe_data = pd.read_csv(args.test_phe, header=0, index_col=0)
            
            if isinstance(args.pnum, int):
                phenotype_name = phe_data.columns[args.pnum]
            elif args.pnum is not None:
                phenotype_name = args.pnum
            else:
                phenotype_name = phe_data.columns[0]
                
            log(INFO, f"Working with phenotype: {phenotype_name}")

            valid_samples = phe_data.index[~phe_data[phenotype_name].isna()].tolist()
            log(INFO, f"Found {len(valid_samples)} non-NA samples for prediction")

            y = phe_data.loc[valid_samples, phenotype_name]
            y_data = y.values
            y_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)
        else:
            log(INFO, "No phenotype file provided, will only generate predictions")

        X = geno_data.loc[valid_samples]
        
        if y_data is not None and len(X) != len(valid_samples):
            log(WARNING, f"Only {len(X)} of {len(valid_samples)} samples found in genotype data")
            
        X_data = X.values.reshape(-1, 1, X.shape[1])
        
        if hasattr(args, 'adjust_encoding') and args.adjust_encoding:
            log(INFO, "Adjusting genotype encoding from {0,1,2} to {-1,0,1}")
            X_data = X_data - 1
        
        X_tensor = torch.tensor(X_data, dtype=torch.float32)
        
        sample_ids = X.index.tolist()
        
        return X_tensor, y_tensor, sample_ids, phenotype_name, feature_names

    except Exception as e:
        log(ERROR, f"Error loading prediction data: {str(e)}")
        raise

def check_model_compatibility(model, input_data, feature_names):
    """Check model compatibility with input data"""
    expected_length = None
    
    if hasattr(model, 'seq_length'):
        expected_length = model.seq_length
        log(INFO, f"Model expects input sequence length: {expected_length}")
    
    actual_length = input_data.shape[2]
    if expected_length is not None and expected_length != actual_length:
        log(ERROR, f"Input feature count mismatch: Model expects {expected_length} features, but input has {actual_length} features")
        raise ValueError(f"Feature count mismatch: expected {expected_length}, got {actual_length}")
    
    if hasattr(model, 'prior_indices') and hasattr(model, 'feature_names') and model.prior_indices:
        log(INFO, f"Model uses prior features at indices: {model.prior_indices}")
        
        if len(model.feature_names) != len(feature_names):
            log(WARNING, f"Feature name count mismatch: Model has {len(model.feature_names)} feature names, but input has {len(feature_names)}")
        
        model.feature_names = feature_names
        
        missing_prior_indices = [idx for idx in model.prior_indices if idx >= actual_length]
        if missing_prior_indices:
            log(WARNING, f"Some prior feature indices are out of range for the input data: {missing_prior_indices}")

    return True

def predict():
    args = parse_args()
    try:
        X_test, y_test, sample_ids, phenotype_name, feature_names = load_prediction_data(args)
        log(INFO, f"Loaded test data - X shape: {X_test.shape}, features: {len(feature_names)}")
        if y_test is not None:
            log(INFO, f"Loaded phenotype data - y shape: {y_test.shape}")

        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        model_path = args.model_path
        log(INFO, f"Loading model from {model_path}")
        model = torch.load(model_path, map_location=device)

        if hasattr(model, 'prior_indices') and model.prior_indices:
            # log(INFO, f"Model uses prior features at indices: {model.prior_indices}")

            model_feature_names = getattr(model, 'feature_names', feature_names)

            if hasattr(model, 'feature_names') and model.feature_names:
                
                pred_name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

                new_order = []
                missing_features = []
                
                for model_feature in model.feature_names:
                    if model_feature in pred_name_to_idx:
                        new_order.append(pred_name_to_idx[model_feature])
                    else:
                        missing_features.append(model_feature)
                
                if missing_features:
                    log(WARNING, f"Some model features not found in prediction data: {missing_features}")
                
                if len(new_order) == len(feature_names):
                    X_test_reordered = X_test[:, :, new_order]
                    X_test = X_test_reordered
                    
                    feature_names_reordered = [feature_names[i] for i in new_order]
                    feature_names = feature_names_reordered
                    
        
        check_model_compatibility(model, X_test, feature_names)
        
        model.eval()

        with torch.no_grad():
            y_pred = model(X_test.to(device)).cpu().numpy()

        os.makedirs(args.output_path, exist_ok=True)
        prefix = get_output_prefix(args)
        
        if y_test is not None:
            metrics = compute_prediction_metrics(y_test.numpy(), y_pred)
            
            from log import log_section_start, log_section_end
            log_section_start("Prediction Evaluation")
            for name, value in metrics.items():
                log(INFO, f"{name}: {value:.6f}")
            log_section_end("-----------------------")
            
            results_df = pd.DataFrame({
                'Sample_ID': sample_ids,
                'True_Values': y_test.numpy().flatten(),
                'Predicted_Values': y_pred.flatten()
            })

            pd.DataFrame([metrics]).to_csv(os.path.join(args.output_path, f"{prefix}_metrics.csv"), index=False)

            visualize_predictions(y_test.numpy(), y_pred, args.output_path, prefix)
        else:
            results_df = pd.DataFrame({
                'Sample_ID': sample_ids,
                'Predicted_Values': y_pred.flatten()
            })
            log(INFO, "No phenotype data provided, only predictions are saved")
        
        results_df.to_csv(os.path.join(args.output_path, f"{prefix}_predictions.csv"), index=False)
        log(INFO, f"Results saved with prefix: {prefix}")
        
        return results_df, y_pred

    except Exception as e:
        log(ERROR, f"Prediction failed: {str(e)}")
        raise

def compute_prediction_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = pearsonr(y_true.flatten(), y_pred.flatten())[0] ** 2
    corr = pearsonr(y_true.flatten(), y_pred.flatten())[0]
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Correlation': corr}

def visualize_predictions(y_true, y_pred, output_path, prefix):
    plot_combined(y_pred.flatten(), y_true.flatten(),
                pearsonr(y_true.flatten(), y_pred.flatten())[0]**2,
                "Prediction Results",
                os.path.join(output_path, f"{prefix}_combined_results.png"))
    
    plt.figure(figsize=(10, 6), dpi=300)
    residuals = y_true.flatten() - y_pred.flatten()
    
    xy = np.vstack([y_pred.flatten(), residuals])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    pred_sorted, res_sorted, z_sorted = y_pred.flatten()[idx], residuals[idx], z[idx]
    
    scatter = plt.scatter(pred_sorted, res_sorted, c=z_sorted, s=12, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Density')
    
    plt.axhline(0, color='k', linestyle='--', alpha=0.7)
    
    mean_err = np.mean(residuals)
    std_err = np.std(residuals)
    
    plt.text(0.05, 0.95, f'Mean error: {mean_err:.3f}\nSTD error: {std_err:.3f}',
             transform=plt.gca().transAxes, ha='left', va='top', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Residual Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{prefix}_residuals.png"), bbox_inches='tight')
    plt.close()

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

if __name__ == "__main__":
    predict()