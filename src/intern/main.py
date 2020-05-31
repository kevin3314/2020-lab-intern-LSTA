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
    DATA_ROOT = 'data/original'

    train_path = osp.join(DATA_ROOT, 'Train_Data_F.pickle')
    dev_path = osp.join(DATA_ROOT, 'Valid_Data_F.pickle')
    test_path = osp.join(DATA_ROOT, 'Test_Data_F.pickle')

    train.run_bert(
            train_path,
            dev_path,
            BATCH_SIZE,
            device
            )


if __name__ == '__main__':
    main()
