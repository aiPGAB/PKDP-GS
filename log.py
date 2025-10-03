#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from datetime import datetime

INFO = 0
WARNING = 1
ERROR = 2

log_output_stream = sys.stdout
log_level_strings = ["INFO", "WARNING", "ERROR"]

def set_log_output(output_stream):
    global log_output_stream
    log_output_stream = output_stream

def log_section_start(message):
    print(f"** {message}")

def log_section_end(message):
    print(f"** {message}")

def log_info(message):
    print(f"** {message}")

def log(level, message, *args):
    if level not in [INFO, WARNING, ERROR]:
        print("Invalid log level")
        return

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = message % args if args else message
    log_level = log_level_strings[level]
    log_entry = f"[{current_time}] {log_level}: {formatted_message}"

    if log_output_stream == sys.stdout:
        print(log_entry)
    else:
        print(log_entry, file=log_output_stream)