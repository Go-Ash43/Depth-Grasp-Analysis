"""DataLoader worker initialisation: one OpenCV and one torch thread per worker (Telea runs in the
workers; nested thread pools oversubscribe the CPUs)."""
import cv2
import torch


def worker_init(_worker_id):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
