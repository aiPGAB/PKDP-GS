import random
import string
import time

def generate_prefix():
    """Generate a random prefix using timestamp and random characters"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
    return f"{timestamp}_{random_str}"

def get_output_prefix(args):
    """Get output prefix from args or generate a random one"""
    return args.prefix if args.prefix else generate_prefix()
