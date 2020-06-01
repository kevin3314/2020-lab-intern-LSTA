import argparse
import os.path as osp

import torch

import train
import test


def main(args):
    # CONSTS
    BATCH_SIZE = 32

    # Configure GPU
    gpu_id = 1
    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")

    # Load Dataset
    DATA_ROOT = 'data/original'

    if args.mode == 'train':
        train_path = osp.join(DATA_ROOT, 'Train_Data_F.pickle')
        dev_path = osp.join(DATA_ROOT, 'Valid_Data_F.pickle')

        train.run(
                train_path,
                dev_path,
                BATCH_SIZE,
                device
                )

    elif args.mode == 'test':
        test_path = osp.join(DATA_ROOT, 'Test_Data_F.pickle')

        test.run(
                test_path,
                BATCH_SIZE,
                device
                )

    else:
        raise AttributeError('args.mode should be train or test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Division word for Japanese')
    parser.add_argument('-m', '--mode', required=True)

    args = parser.parse_args()
    main(args)
