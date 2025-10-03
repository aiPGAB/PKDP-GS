#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from config import parse_args, print_help, print_art
from train import train
from predict import predict
from log import log, log_info, INFO, WARNING, ERROR

def main():
    args = None
    try:
        args = parse_args()
    except SystemExit:
        if len(sys.argv) == 1:
            print_art()
            print_help()
            sys.exit(0)
        else:
            raise
    
    print_art()

    if args.mode == 'train':
        log(INFO, 'Starting model training...')
        train()
    elif args.mode == 'predict':
        log(INFO, 'Starting prediction...')
        predict()
    else:
        log(ERROR, "Please specify mode: 'train' or 'predict'")
        sys.exit(1)

if __name__ == "__main__":
    main()
