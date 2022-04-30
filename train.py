import argparse
import numpy as np
from scipy.fftpack import ss_diff
from tqdm import trange
from tensorboardX import SummaryWriter
# 定义Summary_Writer

import torch
import torch.nn.functional as F
import torch.optim as optim
from data import get_loader
from model import DANN, BaselineModel
from utils import make_dirs, setup_seed


parser = argparse.ArgumentParser(description='PyTorch Training')

# data generation
parser.add_argument('--mode', type=int, default=4, help='which to be target domain')
parser.add_argument('--seed', type=int, default=233, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')

# network arch
parser.add_argument('--alpha', type=float, default=0.9, help='bp alpha')
parser.add_argument('--lamda', type=float, default=1., help='loss weight')
parser.add_argument('--src_only_flag', action='store_true', default=False,
                    help='baseline')

# training setting
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

# show and save
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test-interval', type=int, default=5, metavar='N',
                    help='how many epoch to wait before testing')
parser.add_argument('--save-freq', '-s', default=100, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--prefix', type=str, default='mode4-baseline',
                    help='special name')


def EVAL(model, data_loader, device, writer, flag, epoch, len_data, alpha):
    # 这里会导致bug，原因不明
    # model.eval()

    acc_total = {
        'src_class_acc' : 0, 
        'tgt_class_acc' : 0, 
        'src_domain_acc' : 0,
        'tgt_domain_acc' : 0,
    }

    for batch_idx, example in enumerate(data_loader):
        for key, v in example.items():
            if 'src' in key:
                example[key] = example[key].reshape(-1, example[key].shape[-1])
            if 'label' in key:
                example[key] = example[key].squeeze(-1)

        with torch.no_grad():
            src_class_output, src_domain_output = model(example['src_data'].to(device), alpha)
            tgt_class_output, tgt_domain_output = model(example['tgt_data'].to(device), alpha)

        acc_total['src_class_acc'] += ((src_class_output.data.max(1)[1] == example['src_label'].to(device).data).float().sum())
        acc_total['src_domain_acc'] += ((src_domain_output.data.max(1)[1] == example['src_domain_label'].to(device).data).float().sum())
        acc_total['tgt_class_acc'] += ((tgt_class_output.data.max(1)[1] == example['tgt_label'].to(device).data).float().sum())
        acc_total['tgt_domain_acc'] += ((tgt_domain_output.data.max(1)[1] == example['tgt_domain_label'].to(device).data).float().sum())
    
    for key, v in acc_total.items():
        if 'src' in key:
            acc_total[key] /= (len_data*4)
        else:
            acc_total[key] /= len_data

    writer.add_scalar(f'{flag}_src_class_acc', acc_total['src_class_acc'], epoch)
    writer.add_scalar(f'{flag}_src_domain_acc', acc_total['src_domain_acc'], epoch)
    writer.add_scalar(f'{flag}_tgt_class_acc', acc_total['tgt_class_acc'], epoch)
    writer.add_scalar(f'{flag}_tgt_domain_acc', acc_total['tgt_domain_acc'], epoch)

    return acc_total

def run_epoch(train_loader, test_loader, model, optimizer, args, device):
    
    FolderName, model_dir, log_dir = make_dirs(args.prefix)

    global_step = 0
    writer = SummaryWriter(log_dir)   # 数据存放在这个文件夹
    criterion = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)

    print('Began training ')
    with trange(1, args.epochs + 1) as t:
    # for epoch in tqdm(range(1, args.epochs + 1)):
        for epoch in t:
            model.train()
            for batch_idx, example in enumerate(train_loader):
            
                for key, v in example.items():
                    if 'src' in key:
                        example[key] = example[key].reshape(-1, example[key].shape[-1])
                    if 'label' in key:
                        example[key] = example[key].squeeze(-1)

                # train on source domain
                src_class_output, src_domain_output = model(example['src_data'].to(device), args.alpha)
                src_loss_class = criterion(src_class_output, example['src_label'].to(device))
                
                # train on target domain
                _, tgt_domain_output = model(example['tgt_data'].to(device), args.alpha)

                if args.src_only_flag:
                    loss = src_loss_class
                else:
                    src_loss_domain = criterion(src_domain_output, example['src_domain_label'].to(device))
                    tgt_loss_domain = criterion(tgt_domain_output, example['tgt_domain_label'].to(device))
                    loss = src_loss_class + args.lamda * (src_loss_domain + tgt_loss_domain)
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # log
                global_step +=1
                if global_step % args.log_interval == 0:
                    writer.add_scalar('loss', loss, global_step)
                    writer.add_scalar('class_loss', src_loss_class, global_step)
                    if not args.src_only_flag:
                        writer.add_scalar('domain_loss', src_loss_domain + tgt_loss_domain, global_step)
            t.set_description("EPOCH %i"%epoch)
            if args.src_only_flag:
                t.set_postfix(loss=loss.item(), class_loss = src_loss_class.item(), str="lyh")
            else:
                t.set_postfix(loss=loss.item(), class_loss = src_loss_class.item(), domain_loss = (src_loss_domain + tgt_loss_domain).item(), str="lyh")

            if epoch % args.test_interval == 0:
                EVAL(model, train_loader, device, writer, 'train', epoch, args.len_train, args.alpha)
                EVAL(model, test_loader, device, writer, 'test', epoch, args.len_test, args.alpha)

    state = {'net':model.state_dict()}
    torch.save(state, model_dir)
    print('Model is saved in ' + model_dir)


if __name__ == '__main__':

    args = parser.parse_args()

    # settings
    setup_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': False} if use_cuda else {}
    print('It is running on device')

    # setup data loader
    train_loader,  test_loader, args.len_train, args.len_test = get_loader(args)
    print(f'Dataset {args.mode} is the target domain')

    # setup model and optimizer
    model = DANN().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    run_epoch(train_loader, test_loader, model, optimizer, args, device)

    


    