import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import consts
import utils
from model import LSTM_divider


def train_fn(model, data_loader, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    total_loss = 0.0
    tk0 = tqdm(data_loader, total=len(data_loader), desc="Training")

    for batch_idx, (idseq, length_list, target) in enumerate(tk0):
        idseq = idseq.to(device)
        length_list = length_list.to(device)
        target = target.to(device)

        model.zero_grad()
        output = model(idseq, length_list)

        criterion = nn.BCELoss()
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        tk0.set_postfix(loss=loss.item())

    print(f'total loss {total_loss}')


def eval_fn(model, data_loader, device):
    model.eval()

    with torch.no_grad():
        total_len = 0
        total_loss = 0
        num_correct = 0

        tk0 = tqdm(data_loader, total=len(data_loader), desc="Training")

        for batch_idx, (idseq, length_list, target) in enumerate(tk0):
            idseq = idseq.to(device)
            length_list = length_list.to(device)
            target = target.to(device)

            total_len += target.shape[0] * target.shape[1]

            output = model(idseq, length_list)
            criterion = nn.BCELoss()
            loss = criterion(output, target)

            preds = (output > 0.5).type(torch.LongTensor).cpu()
            target = target.cpu()

            corrects = torch.sum(preds == target)
            num_correct += corrects.item()

            total_loss += loss.item()
            tk0.set_postfix(loss=loss.item())
        print(f'precision -> {num_correct / total_len}')
        print(f'total loss -> {total_loss}')
        return num_correct / total_len


def run(
        train_path,
        dev_path,
        batch_size,
        device,
        epochs=50,
        path="weights/model.bin"
        ):

    # Build model
    print('Building model ...')
    net = LSTM_divider(consts.voc_size)
    net.to(device)
    print('Done!')

    print('Building dataset ...')
    train_dataset = utils.Dataset(
        train_path, consts.CHAR2IDX)
    val_dataset = utils.Dataset(
        dev_path, consts.CHAR2IDX)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=utils.make_batch
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=utils.make_batch
    )
    print('Done!')

    # Class for Early Stopping
    es = utils.EarlyStopping()

    for epoch in range(1, epochs + 1):
        print(f'epoch {epoch}')
        train_fn(net, train_data_loader, device)
        precision = eval_fn(net, val_data_loader, device)

        # If score is not improved during certain term,
        # stop running
        if es.update(precision):
            print(f'Score has not been improved for {es.max_patient} epochs')
            print(f'Best precision -> {es.best}')
            torch.save(net.state_dict(), path)
            return
