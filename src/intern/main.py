import os.path as osp

import torch

import train


def main():
    # CONSTS
    BATCH_SIZE = 32

    # Configure GPU
    gpu_id = 1
    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")

    # Load Dataset
    DATA_ROOT = '/mnt/hinoki/yoshikoshi/sotsuron/dataset/pairpro'

    train_path = osp.join(DATA_ROOT, 'train.txt')
    dev_path = osp.join(DATA_ROOT, 'dev.txt')
    test_path = osp.join(DATA_ROOT, 'test.txt')

    train.run_bert(
            train_path,
            dev_path,
            BATCH_SIZE,
            device
            )


if __name__ == '__main__':
    main()
