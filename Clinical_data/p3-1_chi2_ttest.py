'''
[ 구현 내용 ]
Chi-squared test
two-sampled t-test
[ 수행 내용 ]
▶ load 한 변수들에 대해서 모두 검정 후 결과 저장
▶ 검정 결과 중 pvalue가 threshold값 이하인 feature들만 selection 후 결과 저장
'''

import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import numpy as np

date = '210322_3'

path = './csv/o_preprocessed_clinical_information({})_dummy_corr.csv'.format(date)
source_data_X = pd.read_csv(path, index_col='data_num')
path = './csv/i_PatientsList_217.csv'
source_data_Y  = pd.read_csv(path, index_col=0)


categorical = source_data_X.iloc[:, 0:26]
numerical = source_data_X.iloc[:, 26:]
threshold = 0.1 # p-value threshold
y = source_data_Y.loc[:,'RFS']

tool = 'Logistic-regression' # 'chi-and-T-test', 'Logistic-regression'

"""
단변량 분석 : Chi-squared test, T-test, Logistic Regression 
1. 범주형 독립변수와 범주형 종속 변수의 독립성 검정 (Chi-squared test) (사용안함)
2. recurrence group과 non-recurrence group간의 수치형 변수에 대한 동질성 검정 (T-test) (사용안함)
3. 단변량 logistic regression 분석 
"""
if tool == 'Logistic-regression':
    pval_list_tmp = pd.DataFrame(columns=['col','odds_r','CI_025','CI_975','pval'])
    pval_list_tmp_rr = pd.DataFrame(columns=['col','odds_r','CI_025','CI_975','pval'])

    X = source_data_X.drop(['fold','2yRFS'], axis=1)
    for feature_name, feature in X.iteritems():
        LR_feat = sm.add_constant(feature)
        LR_model = sm.Logit(source_data_Y, LR_feat)

        LR_results = LR_model.fit(disp=False)
        #L_w = LR_results.llf
        print(LR_results.summary())
        Odds_ratio = round(np.exp(LR_results.params)[1],2)
        pvalue = round(LR_results.pvalues[1],3)
        std_err = LR_results.bse[1]

        CI_975 = LR_results.params[1]+std_err*1.96
        CI_025 = LR_results.params[1]-std_err*1.96

        pval_list_tmp = pval_list_tmp.append({'col': feature_name, 'odds_r': Odds_ratio, 'CI_025':CI_025, 'CI_975':CI_975, 'pval': pvalue}, ignore_index=True)
        pval_list_tmp_rr = pval_list_tmp_rr.append({'col': feature_name, 'odds_r': Odds_ratio, 'CI_025':CI_025, 'CI_975':CI_975, 'pval': pvalue}, ignore_index=True)
        LR_results.summary()
        values = feature.unique()
        if len(values) < 6:
            for i, value in enumerate(values):
                categori_odds = round(np.exp(LR_results.params[1]*(value-values[0])),2)
                categori_CI_975 = round(np.exp(CI_975*(value-values[0])),2)
                categori_CI_025 = round(np.exp(CI_025*(value-values[0])),2)
                categori_CI = '('+str(categori_CI_025)+' - '+str(categori_CI_975)+')'
                featuren_name_val = ' - #'+str(i)
                pval_list_tmp_rr = pval_list_tmp_rr.append({'col': featuren_name_val, 'odds_r': categori_odds, 'CI_025':categori_CI_025, 'CI_975':categori_CI_975, 'pval': pvalue}, ignore_index=True)

        # pval_list_tmp = pval_list_tmp.append({'col': feature_name, 'odds_r': Odds_ratio ,'pval': pvalue}, ignore_index=True)

    pval_list = pval_list_tmp_rr[pval_list_tmp_rr.loc[:,'pval']<threshold]
    print(pval_list_tmp)
    print('-- under 0.1')
    print(pval_list)
    Univariate_list = pval_list_tmp[pval_list_tmp.loc[:,'pval']<threshold].loc[:,'col']

    pval_list_tmp.to_csv('./csv/o_Univariate_analysis_LR_result({}).csv'.format(date))
    pval_list_tmp_rr.to_csv('./csv/o_Univariate_analysis_LR_result_including_category({}).csv'.format(date))
    Univariate_feature_list = source_data_X.loc[:, Univariate_list]
    Univariate_feature_list = pd.concat([Univariate_feature_list, source_data_X.loc[:, 'fold'], source_data_X.loc[:, '2yRFS']], axis=1)
    Univariate_feature_list.to_csv('./csv/o_Univariate_feature_selection_LR({}).csv'.format(date))




if tool == 'chi-and-T-test':
    '''
    The significance of categorical variables between two groups
    were tested by Chi-squared test
    '''
    pval_list_ct = pd.DataFrame(columns=['col','pval'])

    for i in categorical:
        result_ctbl = pd.crosstab(categorical.loc[:,i], y)
        result_chi = stats.chi2_contingency(observed=result_ctbl)

        pval_list_ct = pval_list_ct.append({'col': i, 'pval': result_chi[1]}, ignore_index=True)

    print(pval_list_ct)
    print(pval_list_ct[pval_list_ct.loc[:,'pval']<threshold])



    '''
    The significance of continuous variables between two groups
    were tested by two-smapled t-test
    
    독립표본 t 검정 
    : 두 집단의 등분산 여부에 따라 검정 방법이 조금 달라지기 때문에 등분산 검정을 수행해야함
     - levene's test 기각 : 이분산 가정 t-test
     - levene's test 기각X : 등분산 가정 t-test
    '''

    numerical = pd.concat([numerical, source_data_Y], axis=1)
    numerical_recur = numerical.groupby('RFS').get_group(1)
    numerical_nonrc = numerical.groupby('RFS').get_group(0)
    numerical = numerical.drop('RFS',axis='columns')

    pval_list_nu = pd.DataFrame(columns=['col','pval'])

    for i in numerical:
        group_recur = numerical_recur.loc[:, i]
        group_nonrc = numerical_nonrc.loc[:, i]

        # 등분산 검정
        '''
         등분산 검정: 대상 집단의 분산이 같은지 다른지 통계적으로 검정하는 방법
            - 귀무가설(H0): 모든 집단의 분산은 차이가 없다.
            - 대립가설(H1): 적어도 하나 이상의 집단의 분산에 차이가 있다.
            검정 결과 유의값(p-value)이 0.05 미만인 경우 대립가설을 지지
            F값이 클수록 두 표본의 분산이 동일하지 않다는 의미
        '''
        lresult = stats.levene(group_recur, group_nonrc)

        hyper_leven = False
        if lresult[1] < 0.05:
            hyper_leven = True
        #print('{} \n -LeveneResult (F): {} / -p-value : {} \n {}'.format(i,lresult[0], lresult[1], hyper_leven))

        result = stats.ttest_ind(group_recur, group_nonrc, equal_var = hyper_leven)
        pval_list_nu = pval_list_nu.append({'col':i, 'pval':result[1]}, ignore_index=True)

    print(pval_list_nu)
    print(pval_list_nu[pval_list_nu.loc[:,'pval']<threshold])

    chi2_ttest_result = pd.concat([pval_list_ct, pval_list_nu], axis=0)
    chi2_ttest_result.to_csv('./csv/o_chi2_ttest_result({}).csv'.format(date))

    Univariate_list = pd.concat([pval_list_ct[pval_list_ct.loc[:,'pval']<threshold].col,pval_list_nu[pval_list_nu.loc[:,'pval']<0.1].col])
    print(Univariate_list)

    Univariate_feature_list = source_data_X.loc[:,Univariate_list]
    Univariate_feature_list = pd.concat([Univariate_feature_list,source_data_X.loc[:,'fold']], axis=1)
    Univariate_feature_list.to_csv('./csv/o_Univariate_feature_selection({}).csv'.format(date))




'''
------------------------------------------------------------------------------------------------------------------------------------------------------------------
○ chi-squared test
 - https://junsik-hwang.tistory.com/23
 - https://alex-blog.tistory.com/entry/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%B6%84%EC%84%9D%EC%9D%84-%EC%9C%84%ED%95%9C-%ED%86%B5%EA%B3%84%EB%B6%84%EC%84%9D%
   EC%B9%B4%EC%9D%B4%EC%8A%A4%ED%80%98%EC%96%B4-%EA%B2%80%EC%A0%95-feat-python
 - https://freedata.tistory.com/60
○ T-test
 - https://statools.tistory.com/153
 - https://no17.tistory.com/189
------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''