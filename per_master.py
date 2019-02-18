# -*- encoding=utf-8 -*-
"""
Created on Mon Jan 25 14:53:39 CST 2019
last modify on Mon Jan 28 14:53:39 CST 2019
@author: Ead.Y 
"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np


def get_score(wi_list,data):
    """
    :param wi_list: 权重系数列表
    :param data：评价指标数据框
    :return:返回得分
    """

    #  将权重转换为矩阵
    cof_var = np.mat(wi_list)

    #  将数据框转换为矩阵
    context_train_data = np.mat(data)

    #  权重跟自变量相乘
    last_hot_matrix = context_train_data * cof_var.T
    last_hot_score = pd.DataFrame(last_hot_matrix)

    return last_hot_score

def get_entropy_weight(data):
    """
    :param data: 评价指标数据框
    :return: 各指标权重列表
    """
    # 数据标准化
    data = (data - data.min())/(data.max() - data.min())
    m,n=data.shape
    #将dataframe格式转化为matrix格式
    data=data.as_matrix(columns=None)
    k=1/np.log(m)
    yij=data.sum(axis=0)
    #第二步，计算pij
    pij=data/yij
    test=pij*np.log(pij)
    test=np.nan_to_num(test)

    #计算每种指标的信息熵
    ej=-k*(test.sum(axis=0))
    #计算每种指标的权重
    wi=(1-ej)/np.sum(1-ej)
    wi_list=list(wi)

    return  wi_list

def get_little_Ydelay(logx):
    sig = logx.std()
    Xmax = logx.mean()
    a = 15 / logx.std()
    Ymax = 70
    Ydelay = Ymax + a * (Xmax - logx)
    Ydelay = Ydelay.apply(lambda Ydelay: 0 if Ydelay<=0 else 100 if Ydelay>=100 else Ydelay)
    return Ydelay

def get_big_Ydelay(logx):
    sig = logx.std()
    Xmax = logx.mean()
    a = 15 / logx.std()
    Ymax = 70
    Ydelay = Ymax + a * (logx - Xmax)
    Ydelay = Ydelay.apply(lambda Ydelay: 0 if Ydelay<=0 else 100 if Ydelay>=100 else Ydelay)
    return Ydelay
	

if __name__ == '__main__':

	column_names=['user','big','litit','action']
	data0 = pd.read_csv('data.csv',names=column_names,encoding='utf8')
	
	big_logx = np.log(1 + data0['big'])
	litit_logx = np.log(1 + data0['litit'])
	data0['big'] = data0['big'].replace(0,np.e**big_logx.mean()) 
	data0['litit'] = data0['litit'].replace(0,np.e**litit_logx.mean())
	big_logx = np.log(1 + data0['big'])
	litit_logx = np.log(1 + data0['litit'])
	data0["big_score"] = get_big_Ydelay(big_logx)
	data0["litit_score"] = get_little_Ydelay(litit_logx)
		
	data = data0.iloc[:,3:6]
	wi_list=get_entropy_weight(data)
	print wi_list
	score_list=get_score(wi_list,data)
	data0['score']=score_list
	
    # 然后对数据框按得分从大到小排序
    #result = mm.sort_values(by='score', axis=0, ascending=False)
    #result['rank'] = range(1,len(result) + 1)
    #print(result)
	
    # 写出csv数据
	data0.to_csv('perception_v2.csv',float_format='%.4f',index=0,header=0)
	print "----------------end---------------"
