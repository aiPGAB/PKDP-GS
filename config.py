#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

def _add_train_arguments(parser):
    """Add training arguments to parser"""
    required = parser.add_argument_group('Required Arguments')
    required.add_argument('--train_phe', type=str, required=True, help='Path to training phenotype CSV file')
    required.add_argument('--geno', type=str, required=True, help='Path to genotype CSV file')
    required.add_argument('--output_path', type=str, required=True, help='Directory to save model and results')
    required.add_argument('--prefix', type=str, default=None, help='Prefix for output files')
    
    optional = parser.add_argument_group('Optional Arguments')
    optional.add_argument('--pnum', type=str, default=None, help='Name of phenotype to predict (default: first column)')
    optional.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (default: None, i.e., random)')
    optional.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    optional.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    optional.add_argument('--optuna_trials', type=int, default=50, help='Number of Optuna trials for hyperparameter tuning')
    optional.add_argument('--device', type=str, default='cuda', help='Device for training (e.g., "cpu", "cuda")')
    optional.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD', 'AdamW'], help='Optimizer type')
    optional.add_argument('--early_stop', action='store_true', help='Enable early stopping')
    optional.add_argument('--cv_folds', type=int, default=10, 
                          help='Number of cross-validation folds for model training (0-10, 0=disable CV, default=10)')
    optional.add_argument('--prior_features', type=str, nargs='*', default=None, 
                          help='Prior feature IDs (space-separated indices or names, e.g., "10 20" or "SNP1 SNP2")')
    optional.add_argument('--prior_features_file', type=str, default=None, 
                          help='Path to a text file with one prior feature ID per line (overrides --prior_features if provided)')
    
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--conv_kernel_size', type=int, nargs='+', default=[11, 11, 11], 
                            help='Kernel sizes for main convolution path (space-separated, e.g., 11 11 11)')
    model_group.add_argument('--prior_kernel_size', type=int, nargs='+', default=[3, 3, 3], 
                            help='Kernel sizes for prior path convolution (space-separated, e.g., 3 3 3)')
    model_group.add_argument('--main_channels', type=int, nargs='+', default=[64, 32, 32], 
                            help='Number of channels in the main convolution path (space-separated, e.g., 64 32 32)')
    model_group.add_argument('--prior_channels', type=int, nargs='+', default=[16, 32, 32], 
                            help='Number of channels in the prior knowledge path (space-separated, e.g., 16 32 32)')
    model_group.add_argument('--fc_units', type=int, nargs='+', default=[128, 64], 
                            help='Number of units in fully connected layers (space-separated, e.g., 128 64)')
    model_group.add_argument('--dropout', type=float, default=0, 
                            help='Dropout probability for fully connected layers')
    optional.add_argument('--adjust_encoding', action='store_true', 
                          help='Adjust genotype encoding from {0,1,2} to {-1,0,1}')

def _add_predict_arguments(parser):
    """Add prediction arguments to parser"""
    required = parser.add_argument_group('Required Arguments')
    required.add_argument('--geno', type=str, required=True, help='Path to genotype CSV file')
    required.add_argument('--output_path', type=str, required=True, help='Directory to save prediction results')
    required.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    required.add_argument('--prefix', type=str, default=None, help='Prefix for output files')
    
    optional = parser.add_argument_group('Optional Arguments')
    optional.add_argument('--test_phe', type=str, default=None, help='Path to phenotype CSV file for prediction')
    optional.add_argument('--pnum', type=str, default=None, help='Name of phenotype to predict (default: first column)')
    optional.add_argument('--device', type=str, default='cuda', help='Device for prediction (cpu/cuda)')
    optional.add_argument('--adjust_encoding', action='store_true', 
                          help='Adjust genotype encoding from {0,1,2} to {-1,0,1}')
    optional.add_argument('--prior_features', type=str, nargs='*', default=None, 
                          help='Prior feature index or IDs ("10 20" or "SNP1 SNP2")')
    optional.add_argument('--prior_features_file', type=str, default=None, 
                          help='Path to a text file with one prior feature ID per line (overrides --prior_features if provided)')

def get_train_parser():
    parser = argparse.ArgumentParser(
        description="Train a deep learning regression model for genomic selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _add_train_arguments(parser)
    return parser

def get_predict_parser():
    parser = argparse.ArgumentParser(
        description="Predict phenotypes using a trained deep learning regression model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _add_predict_arguments(parser)
    return parser

ART = (
    r"################################################  " + '\n' +
    r"#                                              #  " + '\n' +
    r"#    Genomic selection: PKDP                   #  " + '\n' +
    r"#                                              #  " + '\n' +
    r"#    Contributor: aiPGAB                       #  " + '\n' +
    r"#    Version    : 0.1.0                        #  " + '\n' +
    r"#                                              #  " + '\n' +
    r"#    https://github.com/yunmika/PKDP           #  " + '\n' +
    r"#                                              #  " + '\n' +
    r"################################################  " + '\n'
)

def print_art():
    """Print the ASCII art for the program."""
    print(ART)

def parse_args():
    main_parser = argparse.ArgumentParser(
        description="PKDP: Deep learning for genomic selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = main_parser.add_subparsers(
        dest='mode',
        help='Available commands',
        metavar='{train,predict}'
    )
    subparsers.required = True
    
    train_parser = subparsers.add_parser(
        'train',
        help='Train a deep learning regression model for genomic selection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _add_train_arguments(train_parser)
    
    predict_parser = subparsers.add_parser(
        'predict',
        help='Predict phenotypes using a trained deep learning regression model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _add_predict_arguments(predict_parser)
    
    args = main_parser.parse_args()
    
    if hasattr(args, 'prior_features_file') and args.prior_features_file:
        if not os.path.exists(args.prior_features_file):
            raise FileNotFoundError(f"Prior features file '{args.prior_features_file}' not found")
        with open(args.prior_features_file, 'r') as f:
            args.prior_features = ' '.join(line.strip() for line in f if line.strip())
    elif hasattr(args, 'prior_features') and args.prior_features is not None:
        args.prior_features = ' '.join(map(str, args.prior_features))
    
    return args

def parse_args_legacy():
    """Legacy argument parser for backward compatibility with --mode flag"""
    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True, help="Mode: 'train' or 'predict'")
    args, remaining_argv = mode_parser.parse_known_args()
    parser = get_train_parser() if args.mode == 'train' else get_predict_parser()
    final_args = parser.parse_args(remaining_argv)
    final_args.mode = args.mode
    
    if hasattr(final_args, 'prior_features_file') and final_args.prior_features_file:
        if not os.path.exists(final_args.prior_features_file):
            raise FileNotFoundError(f"Prior features file '{final_args.prior_features_file}' not found")
        with open(final_args.prior_features_file, 'r') as f:
            final_args.prior_features = ' '.join(line.strip() for line in f if line.strip())
    elif hasattr(final_args, 'prior_features') and final_args.prior_features is not None:
        final_args.prior_features = ' '.join(map(str, final_args.prior_features))
    
    return final_args

def print_help():
    print("Usage:")
    print(" python PKDP.py {train,predict} [arguments]")

if __name__ == "__main__":
    print_help()