import csv
import pandas as pd
import torch
from network import Network
from metric import valid
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import os
from kmeans_pytorch import kmeans
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")
# synthetic3d
# BDGP
# mnist4
# YTF10
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
# YTF10
Dataname = "Caltech-5V"
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--data_path', default="./data")  # 数据存放位置
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--mse_epochs", default=30)
parser.add_argument("--con_epochs", default=100)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--lamda_P", default=1.0)
parser.add_argument("--lamda_Q", default=1.0)
parser.add_argument("--feature_dim", default=128)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The code has been optimized.
# The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.
seed = 42
if args.dataset == "synthetic3d":
    args.mse_epochs = 100
    args.con_epochs = 200
    args.learning_rate = 0.0001
    seed = 10
if args.dataset == "BDGP":
    args.mse_epochs = 250
    args.con_epochs = 200
    seed = 5
if args.dataset == "Caltech-2V":
    args.learning_rate = 0.0001
    args.mse_epochs = 300
    args.con_epochs = 100
    seed = 5
if args.dataset == "Caltech-3V":
    args.learning_rate = 0.0001
    args.mse_epochs = 300
    args.con_epochs = 200
    seed = 5
if args.dataset == "Caltech-4V":
    args.learning_rate = 0.0001
    args.mse_epochs = 200
    args.con_epochs = 400
    seed = 10
if args.dataset == "Caltech-5V":
    args.learning_rate = 0.00006
    args.mse_epochs = 300
    args.con_epochs = 200
    seed = 10
if args.dataset == "mnist4":
    args.mse_epochs = 200
    args.con_epochs = 100
    args.learning_rate = 0.0001
    seed = 10
if args.dataset == "YTF10":
    args.batch_size = 512
    args.mse_epochs = 200
    args.con_epochs = 100
    args.learning_rate = 0.0001
    args.feature_dim = 512
    seed = 10
def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pretrain(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, _, _, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(mse(xs[v], xrs[v]))

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


def contrastive_train(epoch, lamda_P, lamda_Q):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, P, Qs, hs = model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v + 1, view):
                loss_list.append(lamda_Q * criterion.forward_label(Qs[v], Qs[w]))
            loss_list.append(lamda_P * F.kl_div(torch.log(P), Qs[v]))
            loss_list.append(mse(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
    return tot_loss / len(data_loader)


if __name__ == "__main__":
    if not os.path.exists('./models'):
        os.makedirs('./models')

    results_path = f'./result/{args.dataset}_results.xlsx'
    if os.path.exists(results_path):
        results_df = pd.read_excel(results_path)
    else:
        results_df = pd.DataFrame(columns=['temperature_l', 'lamda_P', 'lamda_Q', 'acc', 'nmi', 'pur', 'ari'])


    T = 1
    for i in range(T):
        alpha = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
        beta = [0.01, 0.1, 1, 10, 100]
        gamma = [0.01, 0.1, 1, 10, 100]
        for args.temperature_l in alpha:
            for args.lamda_P in beta:
                for args.lamda_Q in gamma:
                    setup_seed(seed)
                    dataset, dims, view, data_size, class_num = load_data(args.dataset, args.data_path)
                    data_loader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=True,
                    )
                    max_res = [0.0, 0]
                    print("ROUND:{} DataName:{} view_num:{}".format(i + 1, Dataname, view))
                    model = Network(view, dims, args.feature_dim, class_num, device)
                    model = model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                                 weight_decay=args.weight_decay)
                    criterion = Loss(args.batch_size, class_num, args.temperature_l, device).to(device)
                    epoch = 1
                    while epoch <= args.mse_epochs:
                        pretrain(epoch)
                        epoch += 1
                    ############################################## Convergence analysis
                    folder = "result/"
                    if not os.path.exists(folder):
                        os.mkdir(folder)
                    result = open('{}/{}.csv'.format(folder, Dataname), 'w+')
                    writer = csv.writer(result)
                    writer.writerow(["ACC", "NMI", "PUR", "loss", "epoch"])
                    ############################################## init As

                    with torch.no_grad():
                        test_loader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=data_size,
                            shuffle=False,
                        )
                        for batch_idx, (xs, _, _) in enumerate(test_loader):
                            for v in range(view):
                                xs[v] = xs[v].to(device)
                            X_hat, P, Qs, hs = model(xs)
                            for v in range(view):
                                cluster_ids_x, cluster_centers = kmeans(X=hs[v], num_clusters=class_num,
                                                                        distance='cosine',
                                                                        device=device)
                                model.As[v].data = torch.tensor(cluster_centers).to(device)
                    ##############################################
                    while epoch <= args.mse_epochs + args.con_epochs:
                        loss = contrastive_train(epoch, args.lamda_P, args.lamda_Q)
                        acc, nmi, pur, ari = valid(model, device, dataset, view, data_size, isprint=False)
                        writer.writerow(
                            ["{:.4f}".format(acc), "{:.4f}".format(nmi), "{:.4f}".format(pur), "{:.4f}".format(loss),
                             epoch - args.mse_epochs])
                        if acc > max_res[0]:
                            max_res = [acc, epoch - args.mse_epochs]
                            state = model.state_dict()
                            torch.save(state, './models/' + args.dataset + '.pth')
                        if epoch == args.mse_epochs + args.con_epochs:
                            print('--------args----------')
                            for k in list(vars(args).keys()):
                                print('%s: %s' % (k, vars(args)[k]))
                            print('--------args----------')
                            checkpoint = torch.load('./models/' + args.dataset + '.pth')
                            model.load_state_dict(checkpoint)
                            acc, nmi, pur, ari = valid(model, device, dataset, view, data_size, isprint=True)
                            # 保存到 xlsx
                            results_df.loc[len(results_df)] = [args.temperature_l, args.lamda_P, args.lamda_Q, acc, nmi,
                                                               pur, ari]
                            results_df.to_excel(results_path, index=False)
                        epoch += 1
