import torch
import torch.nn.functional as F
import numpy as np
import copy
import math
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
import utils as utils
import model4 as models
import os
import argparse
import logging
import time
from termcolor import colored
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(22)

writer = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_logger(args):
    # create logger
    os.makedirs(args.output_log_dir, exist_ok=True)
    time_str = time.strftime('%m-%d-%H-%M')
    log_file = args.dataset + '_lr_' + str(args.lr) + '_norm_type_' + args.norm_type + \
               '_batch_size_' + str(args.batch_size)  +  '_{}.log'.format(time_str)
    final_log_file = os.path.join(args.output_log_dir, log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #
    fmt = '[%(asctime)s] %(message)s'
    color_fmt = colored('[%(asctime)s]', 'green') + ' %(message)s'

    file = logging.FileHandler(filename=final_log_file, mode='a')
    file.setLevel(logging.INFO)
    file.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console)

    return logger


class VAGMSDA():
    def __init__(self, model=models.VAGMSDANet(), source_loaders=0, target_loader=0, batch_size=16, iteration=2000,
                 lr=0.001, momentum=0.9, log_interval=10, id = 1,save_model=None):
        self.model = model
        self.model.to(device)
        self.source_loaders = source_loaders
        self.target_loader = target_loader
        self.batch_size = batch_size
        self.iteration = iteration
        self.lr = lr
        self.momentum = momentum
        self.log_interval = log_interval
        self.id = id
        self.save_model = save_model

    def __getModel__(self):
        return self.model


    def train(self):
        # best_model_wts = copy.deepcopy(model.state_dict())
        LEARNING_RATE = self.lr
        correct = 0
        confusion = 0
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=5e-3)
        for i in range(1, self.iteration + 1):
            self.model.train()


            source_iters = [iter(self.source_loaders[i]) for i in range(len(self.source_loaders))]
            #print(len(source_iters))
            data_source = [next(source_iters[i]) for i in range(len(self.source_loaders))]
            source_data = torch.concatenate([data_source[i][0] for i in range(len(self.source_loaders))], axis=0)
            source_label = torch.concatenate([data_source[i][1] for i in range(len(self.source_loaders))], axis=0) #torch.Size([224, 1])
            #print(source_label.shape)
            s_batch_size = len(source_data)
            s_domain_label = torch.zeros(s_batch_size).long()

            target_iter = iter(self.target_loader)
            target_data, target_label = next(target_iter)
            target_labels = []
            for k in range(14):
                target_labels.append(target_label)
            target_labels = torch.concatenate([target_labels[i] for i in range(14)], axis=0)
            t_batch_size = len(target_data)
            t_domain_label = torch.ones(t_batch_size).long()


            source_data, source_label = source_data.to(
                device), source_label.to(device)
            #print(source_data.shape)


            target_data = target_data.to(device)


            domain_label = torch.cat([s_domain_label,t_domain_label],dim=0)
            domain_label = domain_label.to(device)


            optimizer.zero_grad()

            # get loss
            cls_loss, mmd_loss, att_loss, l1_loss = self.model(source_data, number_of_source=len(self.source_loaders), data_tgt=target_data, label_src=source_label, mark=0)

            # data_src_DSFE = torch.concatenate([data_src_DSFE[i] for i in range(14)], axis=0)
            # data_tgt_DSFE = torch.concatenate([data_tgt_DSFE[i] for i in range(14)], axis=0)
            # #print(data_src_DSFE.shape)
            # source_X = data_src_DSFE.detach().cpu()
            # #print(source_X.size)
            # source_y = source_label.detach().cpu()
            #
            # target_X = data_tgt_DSFE.detach().cpu()
            # #print(target_X.size)
            # target_y = target_labels.detach().cpu()
            #
            # X = torch.cat([source_X, target_X], dim=0)  # 合并特征数据
            # y = torch.cat([source_y, target_y], dim=0)  # 合并标签
            # domains = torch.cat([torch.zeros(source_X.size(0), 1), torch.ones(target_X.size(0), 1)],
            #                     dim=0)  # 0表示源域，1表示目标域
            #
            # X_np = X.numpy()
            # y_np = y.numpy().flatten()
            # domains_np = domains.numpy().flatten()
            #
            # # 使用t-SNE降维到2D，调整perplexity为小于样本数
            # tsne = TSNE(n_components=2, perplexity=5, random_state=42)
            # X_tsne = tsne.fit_transform(X_np)
            #
            # # 定义不同标签对应的标记形状
            # markers = ['o', 's', 'p', 'X']  # 'o'是圆圈，'s'是方形
            # #markers = ['o', 's', '^', 'D']  # 不同标签对应的形状
            # colors = ['blue', 'red']  # 源域为蓝色，目标域为红色
            #
            # # 可视化降维后的数据
            # plt.figure(figsize=(10, 8))
            #
            # # 绘制源域数据
            # for m in range(4):  # 假设有4类标签
            #     plt.scatter(X_tsne[(y_np == m) & (domains_np == 0), 0],
            #                 X_tsne[(y_np == m) & (domains_np == 0), 1],
            #                 marker=markers[m], color=colors[0], label=f'Source Label {m}' if m == 0 else "")
            #
            # # 绘制目标域数据
            # for m in range(4):  # 假设有4类标签
            #     plt.scatter(X_tsne[(y_np == m) & (domains_np == 1), 0],
            #                 X_tsne[(y_np == m) & (domains_np == 1), 1],
            #                 marker=markers[m], color=colors[1], label=f'Target Label {m}' if m == 0 else "")
            # plt.title('t-SNE Visualization')
            # plt.xlabel('t-SNE Component 1')
            # plt.ylabel('t-SNE Component 2')
            # if i % 52 == 0:
            #     plt.savefig(f"./tsne_visualization/{i}.png")
            #plt.show()
            #plt.close()

            gamma = 1 / (1 + math.exp(-10 * (i) / (self.iteration)))
            beta = gamma / 100
            loss = cls_loss + gamma * (mmd_loss + l1_loss + att_loss)


            #print(source_label.shape)
            # loss = cls_loss
            # loss = cls_loss + mmd_loss +  l1_loss
            # loss = cls_loss + gamma * (mmd_loss)
            # writer.add_scalar('Loss/training cls loss', cls_loss, i)
            # writer.add_scalar('Loss/training mmd loss', mmd_loss, i)
            # writer.add_scalar('Loss/training l1 loss', l1_loss, i)
            # writer.add_scalar('Loss/training gamma', gamma, i)
            # writer.add_scalar('Loss/training loss', loss, i)
            loss.backward()
            optimizer.step()
            t_correct, t_confusion = self.test(i)
            if i % log_interval == 0:
                logging.info('acc: {} [({:.6f}%)] \t Loss: {:.6f} \t soft_loss: {:.6f} \t mmd_loss {:.6f} \t l1_loss: {:.6f} \t att_loss: {:.6f}'.format(
                    i, 100. * t_correct /len(self.target_loader.dataset), loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item(), att_loss.item()
                )
                                 )
                if t_correct > correct:
                    correct = t_correct
                    confusion = t_confusion
                    if not os.path.exists('./'+self.save_model+'/model_csub_{}'.format(self.id[0]+1)):
                        os.makedirs('./'+self.save_model+'/model_csub_{}'.format(self.id[0]+1))
                    torch.save(self.model, './'+self.save_model+'/model_csub_{}/{}_BEST.pth'.format(self.id[0]+1,self.id[1]))
                # logging.info('to target max correct: ', correct.item(), "\n")

        return 100. * correct / len(self.target_loader.dataset), confusion

    def test(self, i):
        self.model.eval()
        test_loss = 0
        correct = 0
        corrects = []
        confusion = torch.zeros(4, 4)
        for i in range(len(self.source_loaders)):
            corrects.append(0)
        with torch.no_grad():
            for data, target in self.target_loader:
                # data = map_matrix(data)
                data = data.to(device)
                target = target.to(device)
                preds = self.model(data, len(self.source_loaders))
                for i in range(len(preds)):
                    preds[i] = F.softmax(preds[i], dim=1)
                pred = sum(preds) / len(preds)
                test_loss += F.nll_loss(F.log_softmax(pred,
                                                      dim=1), target.squeeze()).item()
                pred = pred.data.max(1)[1]
                correct += pred.eq(target.data.squeeze()).cpu().sum()
                pred_confusion = confusion_matrix(target.data.squeeze().cpu(), pred.cpu())
                # 如果生成的混淆矩阵形状与累加的混淆矩阵不同，则进行填充
                if pred_confusion.shape != confusion.shape:
                    # 确保混淆矩阵形状相同
                    pred_confusion = np.pad(pred_confusion,
                                            [(0, confusion.shape[0] - pred_confusion.shape[0]),
                                             (0, confusion.shape[1] - pred_confusion.shape[1])],
                                            mode='constant')
                confusion += pred_confusion
                for j in range(len(self.source_loaders)):
                    pred = preds[j].data.max(1)[1]
                    corrects[j] += pred.eq(target.data.squeeze()).cpu().sum()
                # sorted_list = sorted(corrects)
                #
                # # 2. 取前十个值
                # top_ten = sorted_list[:10]
                #
                # # 3. 计算平均值
                # correct1 = sum(top_ten) / len(top_ten)

            # print("correct=", correct)

            sorted_list = sorted(corrects)
            # 取排序后的前 10 项
            top_ten = sorted_list[-10:]
            # 计算均值
            top_ten_mean = sum(top_ten) / len(top_ten)
            #first_ten_mean = sum(corrects[:10]) / 10
            #mean_value = np.mean(corrects)
            # print(len(top_ten))
            # print(top_ten_mean)


            test_loss /= len(self.target_loader.dataset)

            #writer.add_scalar("Test/Test loss", test_loss, i)

            # logging.info('\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            #     test_loss, correct, len(self.target_loader.dataset),
            #     100. * correct / len(self.target_loader.dataset)
            # ))
            # for n in range(len(corrects)):
            #     logging.info('Source' + str(n) + 'accnum {}'.format(corrects[n]/ len(self.target_loader.dataset)))
        return top_ten_mean, confusion


def cross_subject(data, label, session_id, subject_id, category_number, batch_size, iteration, lr, momentum,
                  log_interval):
    one_session_data, one_session_label = np.array(copy.deepcopy(data[session_id])), np.array(copy.deepcopy(label[session_id]))
    train_idxs = list(range(15))
    del train_idxs[subject_id]
    test_idx = subject_id
    target_data, target_label = copy.deepcopy(one_session_data[test_idx]), copy.deepcopy(one_session_label[test_idx])
    source_data, source_label = copy.deepcopy(one_session_data[train_idxs]), copy.deepcopy(
        one_session_label[train_idxs])
    # logging.info('Target_subject_id: ', test_idx)
    # logging.info('Source_subject_id: ', train_idxs)

    del one_session_label
    del one_session_data

    source_loaders = []
    for j in range(len(source_data)):
        source_loaders.append(torch.utils.data.DataLoader(dataset=utils.CustomDataset(source_data[j], source_label[j]),
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          drop_last=True))
    target_loader = torch.utils.data.DataLoader(dataset=utils.CustomDataset(target_data, target_label),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)
    model = VAGMSDA(model=models.MSMDAERNet(pretrained=False, number_of_source=len(source_loaders),
                                            number_of_category=category_number),
                    source_loaders=source_loaders,
                    target_loader=target_loader,
                    batch_size=batch_size,
                    iteration=iteration,
                    lr=lr,
                    momentum=momentum,
                    log_interval=log_interval,
                    id = [session_id, subject_id],
                    save_model='model_aamsda_'+dataset_name+'_cross_subject' )
    # logging.info(model.__getModel__())
    acc, confusion = model.train()
    logging.info('Target_subject_id: {}, current_session_id: {}, acc: {}'.format(test_idx, session_id, acc))
    logging.info('Target_subject_id: {}, current_session_id: {}, confusion: {}'.format(subject_id,session_id, confusion))
    return acc.item(), confusion


def cross_session(data, label, session_id, subject_id, category_number, batch_size, iteration, lr, momentum,
                  log_interval):
    ## LOSO
    train_idxs = list(range(3))
    del train_idxs[session_id]
    test_idx = session_id

    target_data, target_label = copy.deepcopy(np.array(data[test_idx][subject_id])), copy.deepcopy(
        np.array(label[test_idx][subject_id]))

    source_loaders = []
    for j in train_idxs:
        source_loaders.append(torch.utils.data.DataLoader(dataset=utils.CustomDataset(data[j][subject_id], label[j][subject_id]),
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          drop_last=True))
    target_loader = torch.utils.data.DataLoader(dataset=utils.CustomDataset(target_data, target_label),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)
    model = VAGMSDA(model=models.MSMDAERNet(pretrained=False, number_of_source=len(source_loaders),
                                            number_of_category=category_number),
                    source_loaders=source_loaders,
                    target_loader=target_loader,
                    batch_size=batch_size,
                    iteration=iteration,
                    lr=lr,
                    momentum=momentum,
                    log_interval=log_interval,
                    id = [session_id, subject_id],
                    save_model='model_aamsda_'+dataset_name+'_cross_session')
    # logging.info(model.__getModel__())
    acc, confusion = model.train()
    logging.info('Target_session_id: {}, current_subject_id: {}, acc: {}'.format(test_idx, subject_id, acc))
    logging.info('Target_session_id: {}, current_subject_id: {}, confusion: {}'.format(session_id, subject_id, confusion))
    return acc.item(), confusion

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MS-MDAER parameters')
    parser.add_argument('--dataset', type=str, default='seed4',
                        help='the dataset used for MS-MDAER, "seed3" or "seed4"')
    parser.add_argument('--norm_type', type=str, default='ele',
                        help='the normalization type used for data, "ele", "sample", "global" or "none"')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='size for one batch, integer')
    parser.add_argument('--epoch', type=int, default=200,
                        help='training epoch, integer')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--output_log_dir', default='./train/train_log1', type=str,
                        help='output path, subdir under output_root')
    args = parser.parse_args()
    dataset_name = args.dataset
    bn = args.norm_type
    logger = create_logger(args)
    # data preparation
    logging.info(f'Model name: MS-MDAER. Dataset name: {dataset_name}')
    data, label = utils.load_data(dataset_name)
    logging.info(f'Normalization type: {bn}')
    if bn == 'ele':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.norminy(data_tmp[i][j])
    elif bn == 'sample':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.norminx(data_tmp[i][j])
    elif bn == 'global':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.normalization(data_tmp[i][j])
    elif bn == 'none':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
    else:
        pass
    trial_total, category_number, _ = utils.get_number_of_label_n_trial(
        dataset_name)

    # training settings
    batch_size = args.batch_size
    epoch = args.epoch
    lr = args.lr
    logging.info('BS: {}, epoch: {}'.format(batch_size, epoch))
    momentum = 0.9
    log_interval = 10
    iteration = 0
    if dataset_name == 'seed3':
        iteration = math.ceil(epoch * 3394 / batch_size)
    elif dataset_name == 'seed4':
        iteration = math.ceil(epoch * 820 / batch_size)
    else:
        iteration = 5000
    logging.info('Iteration: {}'.format(iteration))
    # store the results
    csub = []
    csesn = []
    cfm = []
     # iteration = 100
    # cross-validation, LOSO
    for session_id_main in range(1,2):
        for subject_id_main in range(15):
            #print(subject_id_main)
            temp_csub, temp_cfm = cross_subject(data_tmp, label_tmp, session_id_main, subject_id_main, category_number,
                                      batch_size, iteration, lr, momentum, log_interval)
            csub.append(temp_csub)
            cfm.append(temp_cfm)
    csub = np.reshape(np.array(csub),[1,15])

    # for subject_id_main in range(2, 15):
    #     for session_id_main in range(3):
    #         temp_csesn, temp_cfm = cross_session(data_tmp, label_tmp, session_id_main, subject_id_main, category_number,
    #                                    batch_size, iteration, lr, momentum, log_interval)
    #         csesn.append(temp_csesn)
    #         cfm.append(temp_cfm)
    # csesn = np.reshape(np.array(csesn), [3, 15])
    # 计算每一行的均值
    row_means = np.mean(csub, axis=1)

    # 计算每一行的标准差
    row_std = np.std(csub, axis=1)

    # 输出结果
    print("每一行的均值：", row_means)
    print("每一行的标准差：", row_std)

#    logging.info(f"Cross-session: {csesn}")
    logging.info(f"Cross-subject: {csub}")
#    logging.info(f"Cross-session mean:{np.mean(csesn)} std: {np.std(csesn)}, confusion: {np.sum(cfm, axis=0)}")
    logging.info(f"Cross-subject mean: {np.mean(csub)} std: {np.std(csub)}, confusion: {np.sum(cfm, axis=0)}")

