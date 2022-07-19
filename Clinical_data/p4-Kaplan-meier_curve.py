from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib import rcParams
from scipy import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import io

'''
file 형태가 mat인지, csv인지 선택해줘야함!
- ifmat 함수
'''

date = "210521"

def draw_KM_curve(event,time, clf):
    #### matplotlib 설정
    rcParams.update({'font.size': 16})
    rcParams.update({'font.family':'Times New Roman'})
    fig, ax = plt.subplots()
    styles = ['-', '--']
    # group_c = ['r', 'g']
    lw = 2
    labels = ['non-recurrence group',
              'recurrence group']

    #### Kaplan-Meier 그리기
    kmf = KaplanMeierFitter()
    for j, label in enumerate(labels):
        ix = np.array(event) == (j+1)
        # TT = T[ix];
        # EE = E[ix];
        # ll = labels[j]
        timeline_arr = np.linspace(0, 24, 25)
        kmf.fit(time[ix], event_observed=event[ix], label=labels[j], timeline=timeline_arr)
        kmf.plot(ax=ax, ci_show=False, linewidth=lw, style=styles[j])

    #### Logrank 검정
    ix = np.array(event) == 1
    result = logrank_test(time[ix], time[~ix], event[ix], event[~ix], alpha=.99)
    pvalue = result.p_value
    ax.text(1,0.2, 'P-value=%.3f' % pvalue, fontsize=20)  # 위치(3.4,0.75) 수동으로 지정필요
    ax.set_ylim([0,1.05])


    #### 그래프 세부설정
    # font_setting0 = plt.font_manager.FontProperties()
    # font_setting0.set_family('Times')  # 'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
    # font_setting0.set_style('normal')  # 'normal', 'oblique', 'italic'
    # font_setting0.set_weight('bold')

    ax.set_xlabel('Time(month)')
    ax.set_ylabel('2-year recurrence free survival')
    #ax.legend(loc='upper right')

    #### 그래프 저장하고 출력하기
    plt.tight_layout()


    plt.savefig(dst_path + 'KM_plot__{}.png'.format(clf), format='png', dpi=300)
    # plt.show()


if __name__ == '__main__':
    #### Kaplan-Meier curve로 나타낼 데이터 입력
    ### csv 인지 mat인지 선택
    ifmat = True
    '''
    Reference: Rich, Jason T., et al. "A practical guide to understanding Kaplan-Meier curves."
               Otolaryngology—Head and Neck Surgery 143.3 (2010): 331-336.
    '''

    if ifmat:
        # mat file 형태로 되어 있을 떄
        ## clinical
        # src_path = "./Kaplan-meier/Clinical_feature/mat({})/".format(date)
        # dst_path = "./Kaplan-meier/Clinical_feature/KM_plot({})/".format(date)
        ## radiomic
        src_path = "./Kaplan-meier/Clinico-radiomic_feature/mat({})/".format(date)
        dst_path = "./Kaplan-meier/Clinico-radiomic_feature/mat({})/".format(date)

        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        root = pd.read_csv('./Kaplan-meier/Clinico-radiomic_feature/csv/GT.csv', index_col='data_num')
        T = root['month']
        # folds = root['fold']
        #datanum = root.index

        clfs = ['cln'] # 'cln', 'intra', 'peri', 'comb'

        ## using radiomic
        mms = ["0mm"]# ["3mm","6mm","9mm","12mm","15mm", "18mm","21mm","24mm","27mm","30mm"]

        from itertools import product

        clf_mms = [clfs, mms]
        clf_mms = list(product(*clf_mms))
        gt_mat = pd.DataFrame()
        for i, clf_mm in enumerate(clf_mms):
            clf_mm = clf_mm[0] + '_' + clf_mm[1]
            file_list = os.listdir(src_path+'{}/'.format(clf_mm))
            label_217 = pd.DataFrame()

            for j, fold in enumerate(file_list):
                splt = fold.split('_')
                splt = splt[2]
                splt = splt.split('.')[0]
                matfile_name = '{}{}/{}'.format(src_path, clf_mm, fold)
                matcontent_name = 'test_label_{}'.format(splt)

                df = pd.DataFrame(io.loadmat(matfile_name)[matcontent_name])
                gt_mat_tmp = pd.DataFrame(io.loadmat(matfile_name)['test_gt'])

                df.index = root[root['fold']==(j+1)].index
                label_217 = pd.concat([label_217, df], axis=0)
                gt_mat = pd.concat([gt_mat, gt_mat_tmp], axis=0)

            label_217.columns = ['predicted']
            E = label_217['predicted']

            tmp = pd.concat([label_217, T], axis = 1)
            tmp.to_csv('{}{}_label_217.csv'.format(dst_path, clf_mm))
            draw_KM_curve(E, T, clf_mm)

    if not ifmat:
        # mat file 형태로 되어있지 않을 때 (csv)
        # 컬럼은 data_num, month, predicted 로 구성되어있어야 함
        # file이름 수동 지정
        ### <REAL CURVE> file_name = 'GT'

        file_name = 'GT'
        src_path = "./Kaplan-meier/Clinical_feature/csv/"
        dst_path = "./Kaplan-meier/Clinical_feature/KM_plot({})/".format(date)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        root = pd.read_csv('./Kaplan-meier/Clinical_feature/csv/{}.csv'.format(file_name), index_col='data_num')
        T = root['month']
        E = root['2yRFS']
        # datanum = root['data_num']

        draw_KM_curve(E, T, file_name)



