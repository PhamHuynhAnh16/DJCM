import os

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# import sys
# sys.path.append(os.getcwd())

from src import MIR1K, cycle, summary, FL # , mae
from inference import DJCM
from evaluate import evaluate


def train():
    weight_svs = 0
    alpha = 10
    gamma = 0
    weight_pe = 2 - weight_svs
    in_channels = 1
    n_blocks = 1
    latent_layers = 1
    seq_l = 2.56
    hop_length = 20
    seq_frames = int(seq_l * 1000 / hop_length)
    logdir = 'runs/MIR1K_Cascade/' + 'nblocks' + str(n_blocks) + '_latent' + str(latent_layers) + '_frames' + str(seq_frames) \
             + '_alpha' + str(alpha) + '_gamma' + str(gamma) + '_svs' + str(weight_svs) + '_pe' + str(weight_pe) + \
             '_gateT'

    pitch_th = 0.5
    learning_rate = 5e-4
    batch_size = 7
    clip_grad_norm = 3
    learning_rate_decay_rate = 0.95
    learning_rate_decay_epochs = 5
    train_epochs = 300
    early_stop_epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # path, hop_length, sequence_length = None, groups = None
    train_dataset = MIR1K(path=r"F:\dataset\dataset", hop_length=hop_length, groups=['train'], sequence_length=seq_l)
    print('train nums:', len(train_dataset))
    valid_dataset = MIR1K(path=r"F:\dataset\dataset", hop_length=hop_length, groups=['test'], sequence_length=None)
    print('valid nums:', len(valid_dataset))
    data_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True, num_workers=2)
    epoch_nums = len(data_loader)
    print('epoch_nums:', epoch_nums)
    learning_rate_decay_steps = len(data_loader) * learning_rate_decay_epochs
    iterations = epoch_nums * train_epochs

    resume_iteration = None
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    if resume_iteration is None:
        # in_channels, n_blocks, hop_length, latent_layers, seq_frames, seq='gru', seq_layers=1
        model = DJCM(in_channels, n_blocks, latent_layers)
        model = nn.DataParallel(model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path, weights_only=False)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt'), weights_only=False))

    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)
    summary(model)
    SDR, RPA, GNSDR, RCA, it = 0, 0, 0, 0, 0
    loop = tqdm(range(resume_iteration + 1, iterations + 1))

    try:
        for i, data in zip(loop, cycle(data_loader)):
            audio_m = data['audio_m'].to(device)
            # audio_v = data['audio_v'].to(device)
            pitch_label = data['pitch'].to(device)
            out_pitch = model(audio_m)

            # loss_svs = mae(out_audio, audio_v)
            loss_pitch = FL(out_pitch, pitch_label, alpha, gamma)
            # weight_pe = loss_svs.item() / loss_pitch.item()
            loss_total = weight_pe * loss_pitch

            optimizer.zero_grad()
            loss_total.backward()
            if clip_grad_norm:
                clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            scheduler.step()

            print(i, end='\t')
            print('loss_total:{:.6f}'.format(loss_total.item()), end='\t')
            # print('loss_svs:{:.6f}'.format(loss_svs.item()), end='\t')
            print('loss_pe:{:.6f}'.format(loss_pitch.item()))

            writer.add_scalar('loss/loss_total', loss_total.item(), global_step=i)
            # writer.add_scalar('loss/loss_svs', loss_svs.item(), global_step=i)
            writer.add_scalar('loss/loss_pe', loss_pitch.item(), global_step=i)

            if i % epoch_nums == 0:
                print('*' * 50)
                print(i, '\t', epoch_nums)
                model.eval()
                with torch.no_grad():
                    metrics = evaluate(valid_dataset, model, batch_size, hop_length, seq_l, device, None, pitch_th)
                    for key, value in metrics.items():
                        writer.add_scalar('validation/' + key, np.mean(value), global_step=i)
                    # gnsdr = np.round((np.sum(metrics["NSDR_W"]) / np.sum(metrics["LENGTH"])), 2)
                    # writer.add_scalar('validation/GNSDR', gnsdr, global_step=i)
                    # sdr = np.round(np.mean(metrics['SDR']), 2)
                    rpa = np.round(np.mean(metrics['RPA']) * 100, 2)
                    rca = np.round(np.mean(metrics['RCA']) * 100, 2)
                    oa = np.round(np.mean(metrics['OA']) * 100, 2)
                    print(f"RPA: {rpa} RCA: {rca} OA: {oa}")
                    if rpa >= RPA:
                        RPA, RCA, it = rpa, rca, i
                        with open(os.path.join(logdir, 'result.txt'), 'a') as f:
                            f.write(str(i) + '\t')
                            # f.write(str(SDR) + '±' + str(np.round(np.std(metrics['SDR']), 2)) + '\t')
                            # f.write(str(GNSDR) + '\t')
                            f.write(str(RPA) + '±' + str(np.round(np.std(metrics['RPA']) * 100, 2)) + '\t')
                            f.write(str(RCA) + '±' + str(np.round(np.std(metrics['RCA']) * 100, 2)) + '\t')
                            f.write(str(oa) + '±' + str(np.round(np.std(metrics['OA']) * 100, 2)) + '\n')
                        torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
                        torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
                model.train()

            if i - it >= epoch_nums * early_stop_epochs:
                break
    except KeyboardInterrupt:
        model.eval()

        RPA, RCA, it = rpa, rca, i
        with open(os.path.join(logdir, 'result.txt'), 'a') as f:
            f.write(str(i) + '\t')
            # f.write(str(SDR) + '±' + str(np.round(np.std(metrics['SDR']), 2)) + '\t')
            # f.write(str(GNSDR) + '\t')
            f.write(str(RPA) + '±' + str(np.round(np.std(metrics['RPA']) * 100, 2)) + '\t')
            f.write(str(RCA) + '±' + str(np.round(np.std(metrics['RCA']) * 100, 2)) + '\t')
            f.write(str(oa) + '±' + str(np.round(np.std(metrics['OA']) * 100, 2)) + '\n')

        torch.save(model, os.path.join(logdir, f'model-{i}-interrupted.pt'))
        torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state-interrupted.pt'))

if __name__ == "__main__":
    train()