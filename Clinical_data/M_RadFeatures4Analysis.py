'''
[ 구현 내용 ]
.mat 형태로 되어있는 radiomic feature (scaling 전)를
clinical feature와 결합 할 수 있도록 전처리 해주는 코드
[ 수행 과정 ]
▶ rawFeatures : .mat 파일에서 hand-crafted features 불러옴
▶ SplitandScale : train/test set으로 나누고 각각 scaling
▶ featureRank : .mat 파일에서 NCA로 계산한 feature weight 기준으로 significant feature 산정
▶ selection
▶ sequence
'''

import pandas as pd
import numpy as np
import matlab as ml

from scipy import io
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import NeighborhoodComponentsAnalysis

class Feature:
    m_meta_loop = ['intra', 'peri', 'comb']
    Folds = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}  # [1,2,3,4,5]
    feature_num = 0
    type = 'n'
    train_norm = pd.DataFrame()
    test_norm = pd.DataFrame()
    train_feature_selected = pd.DataFrame()
    test_feature_selected = pd.DataFrame()

    def __init__(self):
        print("Feature class generated")

    def rawFeatures(self, _type):
        self.type = _type
        if 'intra' in _type:  # False 이면 -1반환
            self.feature_num = 69
        elif 'peri' in _type:
            self.feature_num = 58
        elif 'comb' in _type:
            self.feature_num = 127

        hcf_g1_raw = pd.DataFrame(io.loadmat('./csv/Clinical_radiomic/mat/{}_hcf_g1.mat'.format(_type))['hcf']).iloc[:,0:self.feature_num]
        hcf_g2_raw = pd.DataFrame(io.loadmat('./csv/Clinical_radiomic/mat/{}_hcf_g2.mat'.format(_type))['hcf']).iloc[:,0:self.feature_num]

        print("[{}] rawFeature function done!".format(_type))
        return hcf_g1_raw, hcf_g2_raw

    def SplitandScale(self, _hcf_g1, _hcf_g2, _test_fold):
        DW_Patients = pd.read_csv(".\csv\Clinical_radiomic\i_ref_B.csv", index_col='p_num')
        DW_Radiomic = pd.read_csv(".\csv\Clinical_radiomic\i_ref_C.csv")

        is_g1 = DW_Patients['2RFS'] == 0; patient_g1 = DW_Patients[is_g1]
        is_g2 = DW_Patients['2RFS'] == 1; patient_g2 = DW_Patients[is_g2]
        idx_patient_g1 = list(patient_g1.index)
        idx_patient_g2 = list(patient_g2.index)

        col_radiomic = DW_Radiomic.loc[0:self.feature_num-1,['sub_category']]

        self.col_radiomic = col_radiomic
        col_radiomic = col_radiomic.values.transpose().tolist()


        _hcf_g1.index=idx_patient_g1; _hcf_g1.columns=col_radiomic
        _hcf_g2.index=idx_patient_g2; _hcf_g2.columns=col_radiomic

        hcf = pd.concat([_hcf_g1, _hcf_g2], axis=0).sort_index()
        self.hcf = hcf
        hcf_f = pd.concat([hcf, DW_Patients['fold']], axis=1)
        hcf = pd.concat([hcf, DW_Patients['2RFS']], axis=1)

        is_train = hcf_f['fold'] != self.Folds[_test_fold]
        is_test = hcf_f['fold'] == self.Folds[_test_fold]

        train = hcf[is_train]
        test = hcf[is_test]

        scaler = MinMaxScaler()
        train_norm = scaler.fit_transform(train)
        test_norm = scaler.transform(test)
        self.train_norm = pd.DataFrame(train_norm, index=train.index, columns=hcf.columns)
        self.test_norm = pd.DataFrame(test_norm, index=test.index, columns=hcf.columns)

        tmp =  self.train_norm.sort_values(by="2RFS")

        print("\t<test fold {}> SplitandScale function done!".format(_test_fold))

        return train_norm, test_norm

    def featureRank(self, _test_fold):

        """
        idx_col_by_fold
        :feature selection은 train set으로 진행하기 때문에
        해당하는 test fold가 test로 사용될 때의 순서를 가르키는 인덱스
        """
        idx_col_by_fold = 5 - self.Folds[_test_fold]

        csv_file = './csv/Clinical_radiomic/mat/classify_HCF_{}(0412)_total.mat'.format(self.type)
        feature_weight = pd.DataFrame(io.loadmat(csv_file)['feat_weights_nca'])
        #feature_weight.to_csv('./gg.csv',index=True)
        feature_tmp = pd.concat([feature_weight.iloc[:,idx_col_by_fold], self.col_radiomic], axis=1)
        feature_tmp.index = feature_tmp.index; feature_tmp.columns = ['weight', 'name']
        feature_ranking = feature_tmp.sort_values("weight", ascending=False)

        print("\t<test fold {}> featureRank function done!".format(_test_fold))
        return feature_ranking

    def Selection(self, rank, num):
        idx_selected = rank.iloc[0:num]['name']
        name_selected = idx_selected.values
        idx_num_selected = list(idx_selected.index); idx_num_selected.append(-1)

        train = self.train_norm.iloc[:,idx_num_selected]
        self.train_feature_selected = train
        test = self.test_norm.iloc[:,idx_num_selected]
        self.test_feature_selected = test

        print("\t\t - selection function done! ({}features)".format(num))
        return train, test


    def sequence(self, _type, _test_fold, _feature_num):
        hcf_g1_raw, hcf_g2_raw = self.rawFeatures(_type)
        train_norm, test_norm = self.SplitandScale(hcf_g1_raw, hcf_g2_raw, _test_fold)
        feature_ranking = self.featureRank(_test_fold)
        train_radiomic, test_radiomic = self.Selection(feature_ranking, _feature_num)

        return train_radiomic, test_radiomic
        return train_radiomic, test_radiomic

