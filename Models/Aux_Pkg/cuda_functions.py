"""
cuda_functions.py

This module contains functions related to CUDA and GPU usage.

Author: Miguel Marques
Date: 20-03-2025
"""

import torch


def get_device(logger):
    """
    Checks if CUDA is available and returns the appropriate device

    Params:
        logger: logger object
    """

    logger.info("===================== CUDA =======================")
    if torch.cuda.is_available():
        logger.info("CUDA is available. Using GPU.")
        logger.info(torch.cuda.get_device_name(0))
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info("==================================================\n")
        return torch.device('cuda')
    else:
        logger.info("CUDA is not available. Using CPU.")
        return torch.device('cpu')


def report_gpu(logger):
    """
    Reports the GPU memory usage
    """

    logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024*1024):.2f} MB")
    logger.info(f"Used GPU Memory: {torch.cuda.memory_allocated(0) / (1024*1024):.2f} MB")
    logger.info(f"Cached GPU Memory: {torch.cuda.memory_reserved(0) / (1024*1024):.2f} MB")


