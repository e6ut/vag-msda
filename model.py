import torch.nn.functional as F
import torch
import utils as utils
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from einops import rearrange, repeat
from node_location import convert_dis_m, get_ini_dis_m, return_coordinates
import graphpool


class ChannelAttentionMPL(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionMPL, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, 32, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(32, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        # self.sigmoid = nn.Softmax(dim=1)

    def forward(self, x):
        avg = self.avg_pool(x)
        max = self.max_pool(x)
        avg_out = self.fc2(self.relu1(self.fc1(avg)))
        max_out = self.fc2(self.relu1(self.fc1(max)))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        out = x + attention * x
        return out, attention


class LocalLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LocalLayer, self).__init__()

        self.in_features = 5
        self.out_features = 5
        self.lrelu = nn.LeakyReLU(0.001)
        self.bias = bias
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input, lap, is_weight=True):
        if is_weight:
            weighted_feature = torch.einsum('b i j, j d -> b i d', input, self.weight)
            output = torch.einsum('i j, b j d -> b i d', lap, weighted_feature) + self.bias
        else:
            output = torch.einsum('i j, b j d -> b i d', lap, input)
        return output  # (batch_size, 62, out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        return f"{self.__class__.__name__}: {str(self.in_features)} -> {str(self.out_features)}"


class MesoLayer(nn.Module):
    def __init__(self, subgraph_num, num_heads, coordinate, trainable_vector):
        super(MesoLayer, self).__init__()
        self.subgraph_num = subgraph_num
        self.coordinate = coordinate

        self.lrelu = nn.LeakyReLU(0.001)
        self.graph_list = self.sort_subgraph(subgraph_num)
        self.emb_size = 5
        # self.num_heads = num_heads

        self.softmax = nn.Softmax(dim=0)
        self.att_softmax = nn.Softmax(dim=1)

        # 用于meso区域的节点自适应权重
        self.trainable_vec = Parameter(torch.FloatTensor(trainable_vector))
        # self.trainable_coor_vec = Parameter(torch.FloatTensor(trainable_vector))
        self.weight = Parameter(torch.FloatTensor(self.emb_size, 10))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.trainable_vec.size(0))
        self.trainable_vec.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        # self.trainable_coor_vec.data.uniform_(-stdv, stdv)

    def forward(self, x):
        coarsen_x = self.att_coarsen(x)
        return coarsen_x

    """使用att进行聚合"""

    def att_coarsen(self, features):
        # 获取当前设备
        device = features.device  # 假设 features 已经在正确的设备上
        features = graphpool.feature_trans(self.subgraph_num, features).to(device)
        coordinates = graphpool.location_trans(self.subgraph_num, self.coordinate).to(device)

        coarsen_feature, coarsen_coordinate = [], []

        idx_head = 0
        for index_length in self.graph_list:
            idx_tail = idx_head + index_length
            sub_feature = features[:, idx_head:idx_tail]
            sub_coordinate = coordinates[idx_head:idx_tail]

            # 计算注意力权重
            feature_with_weight = torch.einsum('b j g, g h -> b j h', sub_feature, self.weight)
            feature_T = rearrange(feature_with_weight, 'b j h -> b h j')
            att_weight_matrix = torch.einsum('b j h, b h i -> b j i', feature_with_weight, feature_T)

            att_weight_vector = torch.sum(att_weight_matrix, dim=2)

            att_vec = self.att_softmax(att_weight_vector)

            sub_feature_ = torch.einsum('b j, b j g -> b g', att_vec, sub_feature)
            sub_coordinate_ = torch.einsum('b j, j g -> b g', att_vec, sub_coordinate)
            sub_coordinate_ = torch.mean(sub_coordinate_, dim=0)

            coarsen_feature.append(rearrange(sub_feature_, "b g -> b 1 g"))
            coarsen_coordinate.append(rearrange(sub_coordinate_, "g -> 1 g"))
            idx_head = idx_tail

        coarsen_features = torch.cat(tuple(coarsen_feature), 1)
        coarsen_coordinates = torch.cat(tuple(coarsen_coordinate), 0)
        return coarsen_features

    def sort_subgraph(self, subgraph_num):
        """
        根据子图数量确定子图的划分细节
        :param subgraph_num:
        :return:
        """
        subgraph_16 = [5, 3, 3, 3, 3, 3, 5, 5, 5, 11, 3, 3, 3, 3, 3, 12]
        subgraph_7 = [5, 9, 9, 25, 9, 9, 12]


        graph_list = None
        if subgraph_num == 16:
            graph_list = subgraph_16
        elif subgraph_num == 7:
            graph_list = subgraph_7


        return graph_list


class EmotionAttention(nn.Module):
    def __init__(self, in_planes=[5, 62], ratio=16):
        super(EmotionAttention, self).__init__()

        self.frequency_mix = nn.Sequential(
            # FeedForward(num_patch, token_dim, dropout),
            # nn.LayerNorm(62),
            # nn.BatchNorm1d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ChannelAttentionMPL(in_planes=in_planes[0])
        )
        self.adj = Parameter(torch.FloatTensor(convert_dis_m(get_ini_dis_m(), 9)))
        self.coordinate_matrix = torch.FloatTensor(return_coordinates())

        self.channel_mix = nn.Sequential(
            # nn.LayerNorm(5),
            ChannelAttentionMPL(in_planes=in_planes[1]),
        )
        self.local_gcn_1 = LocalLayer(5, 10, True)
        self.meso_layer_1 = MesoLayer(subgraph_num=16, num_heads=6, coordinate=self.coordinate_matrix,
                                      trainable_vector=78)
        self.meso_layer_2 = MesoLayer(subgraph_num=7, num_heads=6, coordinate=self.coordinate_matrix,
                                      trainable_vector=78)
        self.lrelu = nn.LeakyReLU(0.001)
        self.module = nn.Sequential(
            nn.Linear(735, 256),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x.view(x.size(0), 5, 62)
        # print(x.shape)  torch.Size([224, 5, 62])

        x = x.transpose(1, 2)
        x = self.local_gcn_1(x, self.adj, True)
        residual = x
        # x, attn = self.global_layer_1(x)
        # print(attn.shape) #torch.Size([3584, 62, 62])
        x = x.transpose(1, 2)
        x1 = x
        x = x.contiguous().view(x.size(0), -1)
        # x1 = x.contiguous().view(x.size(0), -1)
        # print(x1.shape)
        # out, att_weight2 = self.channel_mix(x.transpose(1, 2))
        out, att_weight1 = self.frequency_mix(x1)
        # print(att_weight1.shape) #torch.Size([3584, 5, 1])
        out, att_weight2 = self.channel_mix(out.transpose(1, 2))
        x1 = self.meso_layer_1(residual)
        x1 = x1.transpose(1, 2)
        x1 = x1.contiguous().view(x1.size(0), -1)

        x3 = self.meso_layer_2(residual)  # torch.Size([256, 7, 5])\
        x3 = x3.transpose(1, 2)
        x3 = x3.contiguous().view(x3.size(0), -1)
        # print(x1.shape)
        out = out.transpose(1, 2)
        out = out.contiguous().view(out.size(0), -1)
        # print(out.shape) #torch.Size([256, 310])
        # residual = residual.view(residual.size(0), -1)
        # print(residual.shape)
        out = torch.cat((x, out, x1, x3), 1)
        # out = out + residual.view(residual.size(0),-1)
        # print(out.shape)
        out = self.module(out)
        return out, [att_weight1, att_weight2]
        # return out, [attn, att_weight1, att_weight2]


def pretrained_CFE(pretrained=False):
    model = EmotionAttention()
    if pretrained:
        pass
    return model


class DSFE(nn.Module):
    def __init__(self):
        super(DSFE, self).__init__()
        self.module = nn.Sequential(

            nn.Linear(64, 32),
            nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

        )

    def forward(self, x):
        x = self.module(x)
        return x


def domain_discrepancy(out1, out2, loss_type):
    def huber_loss(e, d=1):
        t = torch.abs(e)
        ret = torch.where(t < d, 0.5 * t ** 2, d * (t - 0.5 * d))
        return torch.mean(ret)

    diff = out1 - out2
    if loss_type == 'L1':
        loss = torch.mean(torch.abs(diff))
    elif loss_type == 'Huber':
        loss = huber_loss(diff)
    else:
        loss = torch.mean(diff * diff)
    return loss


def process_domain_data(features, labels):
    """处理特征和标签，返回排序后的特征和各类别样本数"""
    features = [features] if not isinstance(features, list) else features
    sorted_idx = torch.argsort(labels)
    sorted_features = [f[sorted_idx] for f in features]
    unique_labels = torch.unique(labels[sorted_idx])
    num_samples = [(labels == lbl).sum().item() for lbl in unique_labels]
    return sorted_features, num_samples


class MSMDAERNet(nn.Module):
    def __init__(self, pretrained=False, number_of_source=15, number_of_category=4):
        super(MSMDAERNet, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)

        for i in range(number_of_source):
            exec('self.DSFE' + str(i) + '=DSFE()')
            exec('self.cls_fc_DSC' + str(i) +
                 '=nn.Linear(32,' + str(number_of_category) + ')')

        self.weight_d = 0.3
        self.src_ca_last1 = [1. for _ in range(number_of_source)]
        self.tar_ca_last1 = [1.]
        self.src_ca_last2 = [1. for _ in range(number_of_source)]
        self.tar_ca_last2 = [1.]
        # self.src_ca_last3 = [1. for _ in range(number_of_source)]
        # self.tar_ca_last3 = [1.]
        self.domain_loss_type = 'L1'
        self.number_of_source = number_of_source
        self.number_of_category = number_of_category

        self.id = id

    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0):

        mmd_loss = 0

        data_src_DSFE = []
        data_tgt_DSFE = []
        att_loss = 0
        cls_loss = 0
        l1_loss = 0
        tcls_loss = 0
        if self.training == True:
            # common feature extractor
            data_src_CFE, att_src_CFE = self.sharedNet(data_src)
            data_tgt_CFE, att_tgt_CFE = self.sharedNet(data_tgt)

            data_src_CFE = torch.chunk(data_src_CFE, number_of_source, 0)
            label_src = torch.chunk(label_src, number_of_source, 0)
            att_src_CFE_last = torch.chunk(att_src_CFE[-1], number_of_source, 0)
            att_src_CFE_last2 = torch.chunk(att_src_CFE[-2], number_of_source, 0)
            # att_src_CFE_last3 = torch.chunk(att_src_CFE[-3], number_of_source, 0)

            # print(label_src[1].shape)  #torch.Size([16, 1])
            pred_tgt = []
            with torch.no_grad():
                for i in range(number_of_source):
                    DSFE_name = 'self.DSFE' + str(i)
                    data_tgt_DSFE_i = eval(DSFE_name)(data_tgt_CFE)  # torch.Size([16, 32])
                    DSC_name = 'self.cls_fc_DSC' + str(i)
                    pred_tgt_i = eval(DSC_name)(data_tgt_DSFE_i)
                    # print(pred_tgt_i.shape)
                    pred_tgt_i = F.softmax(pred_tgt_i, dim=1)

                    pred_tgt.append(pred_tgt_i.unsqueeze(1))
                pred_tgt = torch.cat(pred_tgt, dim=1)
                # print(pred_tgt[1].shape)#torch.Size([14, 4])
                pred_tgt_w = pred_tgt.mean(1)
                max_prob, label_tgt = pred_tgt_w.max(1)  # (B)
                label_tgt_mask = (max_prob >= 0.95).float()

            pred_tgt_all = []
            mmd_loss_sub = []  # 用于存储每个源域和目标域的 MMD 损失
            # mmd_loss = 0  # 总的 MMD 损失

            for i in range(number_of_source):
                # Each domain-specific feature extractor
                # to extract the domain-specific feature of target data

                DSFE_name = 'self.DSFE' + str(i)

                # 提取当前源域的特征
                data_src_DSFE_i = eval(DSFE_name)(data_src_CFE[i])
                data_tgt_DSFE_i = eval(DSFE_name)(data_tgt_CFE)

                # 将当前特征分别存储到列表中
                data_src_DSFE.append(data_src_DSFE_i)
                data_tgt_DSFE.append(data_tgt_DSFE_i)



                # 计算当前源域与目标域的 MMD 损失
                mmd_loss_i = utils.mmd_linear(data_src_DSFE_i, data_tgt_DSFE_i)

                # 将损失存储到列表中
                mmd_loss_sub.append(mmd_loss_i)
                # mmd_loss_sub 是计算后的损失列表

                # 累加到总损失
                mmd_loss += mmd_loss_i

                # Each domian specific classifier

                DSC_name = 'self.cls_fc_DSC' + str(i)
                pred_src_i = eval(DSC_name)(data_src_DSFE_i)
                cls_loss += F.nll_loss(F.log_softmax(
                    pred_src_i, dim=1), label_src[i].squeeze())

                pred_tgt_i = eval(DSC_name)(data_tgt_DSFE_i)
                # print(pred_tgt_i.shape) #torch.Size([16, 4])
                tcls_loss_i = F.nll_loss(F.log_softmax(
                    pred_tgt_i, dim=1), label_tgt, reduction='none')
                tcls_loss += (tcls_loss_i * label_tgt_mask).mean()
                pred_tgt_all.append(pred_tgt_i)
                # print(label_src[i].squeeze())
                # print(pred_tgt_i.shape)

                # print(label_tgt)
                tgt_features_sorted, num_t = process_domain_data(pred_tgt_i, label_tgt)
                # print(len(num_s))
                # print(len(num_t))

                ema_alpha = 0.8

                mean_tar_ca1 = self.tar_ca_last1[0] * ema_alpha + (1. - ema_alpha) * torch.mean(att_tgt_CFE[-1], 0)
                self.tar_ca_last1[0] = mean_tar_ca1.detach()

                mean_src_ca1 = self.src_ca_last1[i] * ema_alpha + (1. - ema_alpha) * torch.mean(att_src_CFE_last[i], 0)
                att_loss += self.weight_d / self.number_of_source * domain_discrepancy(mean_src_ca1, mean_tar_ca1,
                                                                                       self.domain_loss_type)

                mean_tar_ca2 = self.tar_ca_last2[0] * ema_alpha + (1. - ema_alpha) * torch.mean(att_tgt_CFE[-2], 0)
                self.tar_ca_last2[0] = mean_tar_ca2.detach()

                mean_src_ca2 = self.src_ca_last2[i] * ema_alpha + (1. - ema_alpha) * torch.mean(att_src_CFE_last2[i], 0)
                att_loss += self.weight_d / self.number_of_source * domain_discrepancy(mean_src_ca2, mean_tar_ca2,
                                                                                       self.domain_loss_type)

                self.src_ca_last1[i] = mean_src_ca1.detach()
                self.src_ca_last2[i] = mean_src_ca2.detach()
                # self.src_ca_last3[i] = mean_tar_ca3.detach()



            for i in range(self.number_of_source):
                    for j in range(i + 1, self.number_of_source):
                        pred_tgt_i = pred_tgt_all[i]
                        pred_tgt_j = pred_tgt_all[j]
                        l1_loss += torch.mean(torch.abs(
                            F.softmax(pred_tgt_i, dim=1) - F.softmax(pred_tgt_j, dim=1)
                        ))

                # 归一化损失
            l1_loss *= 2 / (self.number_of_source * (self.number_of_source - 1))


            return cls_loss + 0.2 * tcls_loss, mmd_loss, att_loss, l1_loss




        else:
            data_CFE, _ = self.sharedNet(data_src)
            pred = []
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                DSC_name = 'self.cls_fc_DSC' + str(i)
                feature_DSFE_i = eval(DSFE_name)(data_CFE)
                pred.append(eval(DSC_name)(feature_DSFE_i))

            return pred



