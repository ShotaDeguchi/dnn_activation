"""
********************************************************************************
inference
********************************************************************************
"""

import os
import time
import argparse

from dnn import *

def __main__():
    # argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--in-dir", 
        help = "input directory"
    )
    parser.add_argument(
        "-o", "--out-dir", 
        help = "output directory"
    )
    parser.add_argument(
        "-e", 
        type = int, 
        default = int(3e2), 
        help = "epochs"
    )
    parser.add_argument(
        "-b", 
        type = int, 
        default = int(2 ** 10), 
        help = "batch size"
    )

    # infer

if __name__ == "__main__":
    __main__()
