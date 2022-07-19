'''
[ 구현 내용 ]
clinical feature들을 fold별로 training/test set으로 나누어줌
'''

import pandas as pd

class ClnFeature:

    date = '000000'

    def __init__(self, _date):
        print("ClnFeature class generated")
        self.date = _date

    def road_clnFeature(self, _test_fold):
        Significant_features = ['T-stage', 'N-stage', 'Pathology-visceral-pleural','Hg']
        csv_path = './csv/Clinical_radiomic/Clinical_features.csv'.format(self.date)
        cln_total = pd.read_csv(csv_path, index_col=0)
        DW_Patients = pd.read_csv("./csv/Clinical_radiomic/i_ref_B.csv", index_col='p_num')
        del cln_total['2yRFS']
        Folds = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}  # [1,2,3,4,5]

        is_test = DW_Patients['fold'] == Folds[_test_fold]
        test = cln_total[is_test]
        train = cln_total[~is_test]

        print("\t<test fold {}> clinical data split!".format(_test_fold))
        return train, test
