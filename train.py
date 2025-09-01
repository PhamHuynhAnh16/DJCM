import os
import sys
import torch

import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

now_dir = os.getcwd()
sys.path.append(now_dir)

from src.loss import FL
from src.model import DJCM
from src.dataset import MIR1K
from evaluate import evaluate
from src.utils import summary, cycle

def train():
    alpha = 10
    gamma = 0

    weight_pe = 2
    in_channels = 1
    n_blocks = 1
    latent_layers = 1

    seq_l = 2.56
    hop_length = 320
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logdir = 'runs/MIR1K_Cascade'
    dataset_path = "/kaggle/working/dataset"

    pitch_th = 0.08
    learning_rate = 5e-4
    batch_size = 16
    clip_grad_norm = 3

    iterations = 100000
    validation_interval = 200
    learning_rate_decay_rate = 0.98
    learning_rate_decay_steps = 1000

    save_every_checkpoint = False

    train_dataset = MIR1K(path=dataset_path, hop_length=hop_length, groups=['train'], sequence_length=seq_l, spec=True)
    print('train nums:', len(train_dataset))

    valid_dataset = MIR1K(path=dataset_path, hop_length=hop_length, groups=['test'], sequence_length=None)
    print('valid nums:', len(valid_dataset))

    data_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True, num_workers=2)
    epoch_nums = len(data_loader)
    print('epoch_nums:', epoch_nums)

    resume_iteration = None
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    if resume_iteration is None:
        model = DJCM(in_channels or 1, n_blocks or 1, latent_layers or 1)
        model = nn.DataParallel(model).to(device)

        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model = torch.load(os.path.join(logdir, f'model-{resume_iteration}.pt'), weights_only=False)
        model.train()
    
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt'), weights_only=False))

    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)
    summary(model)

    RPA, RCA = 0, 0
    loop = tqdm(range(resume_iteration + 1, iterations + 1))

    for i, data in zip(loop, cycle(data_loader)):
        spec = data['spec'].to(device)
        pitch_label = data['pitch'].to(device)

        out_pitch = model(spec) # [:, :-1, :]
        loss_pitch = FL(out_pitch, pitch_label, alpha, gamma)
        loss_total = weight_pe * loss_pitch

        optimizer.zero_grad()
        loss_total.backward()

        if clip_grad_norm:
            clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()
        scheduler.step()

        print(i, end='\t')
        print('loss_total:{:.6f}'.format(loss_total.item()), end='\t')
        print('loss_pe:{:.6f}'.format(loss_pitch.item()))

        writer.add_scalar('loss/loss_total', loss_total.item(), global_step=i)
        writer.add_scalar('loss/loss_pe', loss_pitch.item(), global_step=i)

        # if i % epoch_nums == 0:
        if i % validation_interval == 0:
            # print('*' * 50)
            # print(i, '\t', epoch_nums)

            model.eval()
            with torch.no_grad():
                metrics = evaluate(valid_dataset, model, batch_size, hop_length, seq_l, device, pitch_th)
                for key, value in metrics.items():
                    writer.add_scalar('validation/' + key, np.mean(value), global_step=i)

                rpa = np.round(np.mean(metrics['RPA']) * 100, 2)
                rca = np.round(np.mean(metrics['RCA']) * 100, 2)
                oa = np.round(np.mean(metrics['OA']) * 100, 2)

                print(f"RPA: {rpa} RCA: {rca} OA: {oa}")

                if rpa >= RPA or save_every_checkpoint:
                    RPA, RCA = rpa, rca

                    with open(os.path.join(logdir, 'result.txt'), 'a') as f:
                        f.write(str(i) + '\t')
                        f.write(str(RPA) + '±' + str(np.round(np.std(metrics['RPA']) * 100, 2)) + '\t')
                        f.write(str(RCA) + '±' + str(np.round(np.std(metrics['RCA']) * 100, 2)) + '\t')
                        f.write(str(oa) + '±' + str(np.round(np.std(metrics['OA']) * 100, 2)) + '\n')

                    torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
                    torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))

            model.train()

if __name__ == "__main__":
    train()