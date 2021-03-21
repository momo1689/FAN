import torch
import shutil
import time
from pathlib import Path
from torch import nn, optim
from torch.utils import data
from options import get_args
from networks.FAN import FAN
from loss import Loss
from utils import weight_init_kaiming, PSNR_batch
from Dataset.data_loader import DataFromH5File, DataFromH5AWGN
from tensorboardX import SummaryWriter


def main():
    args = get_args()
    args.train_dir = 'F:/data/train_withAug_new_map.hdf5'
    args.val_dir = 'F:/data/val_withAug_new_map.hdf5'
    args.batch_size = 4
    # model initial
    model = FAN(depth_S=args.depth_S, depth_U=args.depth_U, feature_dims=args.feature_dims,
                wave_pattern=args.wave, level=args.level)
    model = nn.DataParallel(model).cuda()
    if args.weight is not None:
        model.load_state_dict(args.weight)
    else:
        model = weight_init_kaiming(model)
        if Path(args.output_path).exists():
            shutil.rmtree(args.output_path)
        Path(args.output_path).mkdir(parents=True)

    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr_initial)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cycle_epochs, eta_min=5e-10)
    loss_fn = Loss(args.loss_mode)

    # dataset
    train_set = DataFromH5File(args.train_dir)
    train_loader = data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True)
    val_set = DataFromH5File(args.val_dir)
    val_loader = data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=True)

    # train model
    min_loss = float('inf')
    max_psnr = 0
    iteration = train_set.__len__() // args.batch_size
    val_iter = val_set.__len__() // args.batch_size
    writer = SummaryWriter(args.log_dir)
    print('\nBegin training with GPU')
    for i in range(args.epochs):
        tic = time.time()
        eval_loss = 0.0

        # train stage
        model.train()
        for num, (batch_noisy, batch_gt, batch_gtmap) in enumerate(train_loader):
            optimizer.zero_grad()
            pred_map, pred_img = model(batch_noisy.cuda())
            loss = loss_fn([pred_img, pred_map], [batch_gt.cuda(), batch_gtmap.cuda()])
            loss.backward()
            optimizer.step()
            eval_loss = eval_loss + loss.item()

            # print log
            if (num+1) % args.print_freq == 0:
                log_str = '[Epoch: {:2d}/{:2d}] iteration: {:4d}/{:4d} Loss = {:+4.6f}'
                print(log_str.format(i+1, args.epochs, num+1, iteration, loss.item()))
                writer.add_scalar('Train Loss iter', loss.item(), num+1)
        log_str = 'Train loss of epoch {:2d}/{:2d}: {:+.10e}'
        eval_loss = eval_loss / iteration
        print(log_str.format(i+1, args.epochs, eval_loss))
        writer.add_scalar('Loss_epoch', eval_loss, i)

        # test stage
        print('Test Stage Begin')
        model.eval()
        eval_psnr = 0.0
        for num, (batch_noisy, batch_gt, batch_gtmap) in enumerate(val_loader):
            with torch.set_grad_enabled(False):
                pred_map, pred_img = model(batch_noisy.cuda())
            batch_gt = batch_gt.cuda()
            psnr = PSNR_batch(batch_gt, pred_img)
            eval_psnr += psnr.item()
        eval_psnr = eval_psnr / val_iter
        log_str = '[Test for Epoch: {:d}/{:d} psnr per epoch = {:f}'
        print(log_str.format(i+1, args.epochs, eval_psnr))

        scheduler.step()
        # save model
        if (i+1) % args.save_model_freq == 0 or i+1 == args.epochs or \
                (min_loss > eval_loss) or (eval_psnr > max_psnr):
            model_prefix = 'model_'
            save_path_model = Path(args.output_path).joinpath(model_prefix+str(i+1)+'_psnr_'+str(eval_psnr)+'dB')
            torch.save({'epoch': i+1, 'optimizer_state_dict': optimizer.state_dict()}, save_path_model)
            model_state_prefix = 'model_state_'
            save_path_model_sate = Path(args.output_path).joinpath(model_state_prefix+str(i+1))
            torch.save(model.state_dict(), save_path_model_sate)

        min_loss = eval_loss if eval_loss < min_loss else min_loss
        max_psnr = eval_psnr if max_psnr < eval_psnr else max_psnr
        writer.add_scalar('PSNR_epoch_test', eval_psnr, i)
        toc = time.time()
        print('This epoch takes time {:2f}\n'.format(toc - tic))

    writer.close()
    print('Training is over')


if __name__ == '__main__':
    main()
