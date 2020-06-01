import torch
import torch.nn as nn
from tqdm import tqdm

import consts
import utils
from model import LSTM_divider


def eval_fn(
        model,
        data_loader,
        device
        ):

    model.eval()

    with torch.no_grad():
        total_len = 0
        total_loss = 0
        total_F = 0

        tk0 = tqdm(data_loader, total=len(data_loader), desc="Evaluationg")

        for batch_idx, (idseq, length_list, target) in enumerate(tk0):
            idseq = idseq.to(device)
            length_list = length_list.to(device)
            target = target.to(device)

            total_len += target.shape[0]

            output = model(idseq, length_list)
            criterion = nn.BCELoss()
            loss = criterion(output, target)

            preds = (output > 0.5).type(torch.LongTensor).cpu()
            target = target.cpu()

            true_positive = (preds * target).sum(axis=1)
            true_negative = ((1-preds) * (1-target)).sum(axis=1)
            false_positive = ((preds) * (1-target)).sum(axis=1)
            false_negative = ((1-preds) * target).sum(axis=1)

            precision = 1.0 * true_positive / (true_positive + false_positive)
            recall = 1.0 * true_positive / (true_positive + false_negative)

            F = 2 * precision * recall / (precision + recall)
            F = nan_to_num(F)

            total_F += float(F.sum().item())
            total_loss += loss.item()
            tk0.set_postfix(loss=loss.item())

        print(f'Average F1 -> {total_F / total_len}')
        print(f'total loss -> {total_loss}')
        return total_F / total_len


def run(
        test_path,
        batch_size,
        device,
        epochs=50,
        path="weights/model.bin"
        ):

    # Build model
    print('Building model ...')
    net = LSTM_divider(consts.voc_size)
    net.load_state_dict(torch.load(path))
    net.to(device)
    print('Done!')

    print('Building dataset ...')
    test_dataset = utils.Dataset(
        test_path, consts.CHAR2IDX)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=utils.make_batch
    )
    print('Done!')

    eval_fn(net, test_data_loader, device)


def nan_to_num(t, mynan=0.):
    if torch.all(torch.isfinite(t)):
        return t
    if len(t.size()) == 0:
        return torch.tensor(mynan)
    return torch.cat([nan_to_num(l).unsqueeze(0) for l in t],0)
