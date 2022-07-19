from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib import rcParams

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


if __name__ == '__main__':
    #### 1. Kaplan-Meier curve로 나타낼 데이터 입력
    '''
    Reference: Rich, Jason T., et al. "A practical guide to understanding Kaplan-Meier curves."
               Otolaryngology—Head and Neck Surgery 143.3 (2010): 331-336.
    '''
    src_path = "./Kaplan-meier/Clinical_feature/"
    dst_path = src_path+'KM_plot/'
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)


    # groups = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    # events = [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    # times = [1, 2, 3, 4, 4.5, 5, 0.5, 0.75, 1, 1.5, 2, 3.5]
    df = pd.read_csv(src_path+'Clinical_feature_from_Uni.csv')

    # stratification criteria for binary features
    stratification_criterion_b = ['pathology-lymphovascularinvasion#-',
                                'pathology-visceralpleural#-',
                                'tumor_stage#StageIIB',
                                'tumor_stage#StageIIIA']
    stratification_criterion_o = {'primary_tumor_pathologic_spread':3,
                                'lymphnode_pathologic_spread':3}

    groups = []
    events = np.array(df['2RFS'])
    times = np.array(df['month'])

    for i, cri in enumerate(stratification_criterion_b):
        groups = np.array(df[cri])
        labels = [cri+"-NO", cri+"-YES"]

        #### 2. 데이터 전처리
        E = np.array(events, dtype=np.int32)
        T = np.array(times, dtype=np.float32)

        #### 3. matplotlib 설정
        rcParams.update({'font.size': 12})
        fig, ax = plt.subplots()
        styles = ['-', '--']
        #group_c = ['r', 'g']
        lw = 3

        #### 4. Kaplan-Meier 그리기
        kmf = KaplanMeierFitter()
        for j, label in enumerate(labels):
            ix = np.array(groups) == (j)
            TT = T[ix]; EE = E[ix]; ll = labels[j]
            timeline_arr = np.linspace(0,24,25)
            kmf.fit(T[ix], event_observed=E[ix], label=labels[j], timeline=timeline_arr)
            kmf.plot(ax=ax, ci_show=False, linewidth=lw, style=styles[j])

        #### 5. Logrank 검정
        ix = np.array(groups) == 1
        result = logrank_test(T[ix], T[~ix], E[ix], E[~ix], alpha=.99)
        pvalue = result.p_value
        ax.text(1.0, 0.4, 'P-value=%.3f' % pvalue)  # 위치(3.4,0.75) 수동으로 지정필요

        #### 6. 그래프 세부설정
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('Survival', fontsize=14)
        ax.legend(loc='upper right')

        #### 7. 그래프 저장하고 출력하기
        plt.tight_layout()

        plt.savefig(dst_path+'KM_plot__{}.png'.format(cri), format='png', dpi=300)
        #plt.show()


    for i, cri in enumerate(list(stratification_criterion_o.keys())):
        groups = np.array(df[cri])
        labels = []
        for j in range(stratification_criterion_o[cri]):
            labels.append(cri+'-'+str(j+1))
            labels

        #### 2. 데이터 전처리
        E = np.array(events, dtype=np.int32)
        T = np.array(times, dtype=np.float32)

        #### 3. matplotlib 설정
        rcParams.update({'font.size': 12})
        fig, ax = plt.subplots()
        styles = ['-', '--', '-']
        # group_c = ['r', 'g']
        lw = 3
        timeline_arr = np.linspace(0, 24, 25),


        #### 4. Kaplan-Meier 그리기
        kmf = KaplanMeierFitter()
        for k, label in enumerate(labels):
            ix = np.array(groups) == (k+1)
            TT = T[ix];
            EE = E[ix];
            ll = labels[k]
            kmf.fit(T[ix], event_observed=E[ix], label=labels[k], timeline=timeline_arr)
            kmf.plot(ax=ax, ci_show=False, linewidth=lw, style=styles[k])

        #### 5. Logrank 검정

        ix = np.array(groups) == 1
        result = logrank_test(T[ix], T[~ix], E[ix], E[~ix], alpha=.99)
        pvalue = result.p_value
        ax.text(1.0, 0.4, 'P-value=%.3f' % pvalue)  # 위치(3.4,0.75) 수동으로 지정필요

        # pvalue = []
        # ix = np.array(groups) == 1
        # result = logrank_test(T[ix], T[~ix], E[ix], E[~ix], alpha=.99)
        # pvalue.append(result.p_value)
        #
        # ix = np.array(groups) == 2
        # result = logrank_test(T[ix], T[~ix], E[ix], E[~ix], alpha=.99)
        # pvalue.append(result.p_value)
        #
        # ix = np.array(groups) == 3
        # result = logrank_test(T[ix], T[~ix], E[ix], E[~ix], alpha=.99)
        # pvalue.append(result.p_value)
        #
        # pvalue_avg = sum(pvalue)/len(pvalue)
        # ax.text(0.5, 0.15, 'P-value=%.3f' % pvalue_avg)  # 위치(3.4,0.75) 수동으로 지정필요

        #### 6. 그래프 세부설정
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('Survival', fontsize=14)
        ax.legend(loc='upper right')

        #### 7. 그래프 저장하고 출력하기
        plt.tight_layout()

        plt.savefig(dst_path + 'KM_plot__{}.png'.format(cri), format='png', dpi=300)
        #plt.show()