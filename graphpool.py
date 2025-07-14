import torch
import numpy as np



def feature_trans(subgraph_num, feature):

    if subgraph_num==16:  # 脑区分割
        return feature_trans_16(feature)
    elif subgraph_num==7:                  # 脑区分割
        return feature_trans_7(feature)
    
    pass


def location_trans(subgraph_num, location):

    if subgraph_num==16:                  # 脑区分割
        return location_trans_16(location)
    elif subgraph_num==7:                  # 脑区分割
        return location_trans_7(location)
   
    pass


##############################################################
########################### 脑区分割
##############################################################

def feature_trans_7(feature):
    """
    对于原始的特征进行变换，使其符合att_pooling的形状
    :param feature:
    :return:
    """
    reassigned_feature = torch.cat((
        feature[:, 0:5],

        feature[:, 5:8], feature[:, 14:17], feature[:, 23:26],

        feature[:, 23:26], feature[:, 32:35], feature[:, 41:44],

        feature[:, 7:12], feature[:, 16:21],feature[:, 25:30],
        feature[:, 34:39], feature[:, 43:48],

        feature[:, 11:14],feature[:, 20:23],feature[:, 29:32],

        feature[:, 29:32],feature[:, 38:41],feature[:, 47:50],

        feature[:, 50:62]), dim=1)

    return reassigned_feature




def location_trans_7(location):
    """
    对于原始的坐标进行变换，使其符合att_pooling的形状
    :param feature:
    :return:
    """
    reassigned_location = torch.cat((
        location[0:5],

        location[5:8], location[14:17], location[23:26],

        location[23:26], location[32:35], location[41:44],

        location[7:12], location[16:21],location[25:30],
        location[34:39], location[43:48],

        location[11:14],location[20:23],location[29:32],

        location[29:32],location[38:41],location[47:50],

        location[50:62]), dim=0)

    return reassigned_location

def feature_trans_16(feature):
    """
    对于原始的特征进行变换，使其符合att_pooling的形状
    :param feature:
    :return:
    """
    reassigned_feature = torch.cat((
        feature[:, 0:5],

        feature[:, 5:8],

        feature[:, 14:17],

        feature[:, 23:26],

        feature[:, 32:35],

        feature[:, 41:44],

        feature[:, 7:12],

        feature[:, 16:21],

        feature[:, 25:30],

        feature[:, 34:39], feature[:, 43:48],

        feature[:, 11:14],

        feature[:, 20:23],

        feature[:, 29:32],

        feature[:, 38:41],

        feature[:, 47:50],

        feature[:, 50:62]), dim=1)

    return reassigned_feature


def location_trans_16(location):
    """
    对于原始的坐标进行变换，使其符合att_pooling的形状
    :param feature:
    :return:
    """
    reassigned_location = torch.cat((
        location[0:5],

        location[5:8],

        location[14:17],

        location[23:26],

        location[32:35],

        location[41:44],

        location[7:12],

        location[16:21],

        location[25:30],

        location[34:39], location[43:48],

        location[11:14],

        location[20:23],

        location[29:32],

        location[38:41],

        location[47:50],

        location[50:62]), dim=0)

    return reassigned_location

