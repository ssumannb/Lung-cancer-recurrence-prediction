__author__='soominlee@swu.ac.kr'

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats.distributions import chi2
import matplotlib.pyplot as plt


date = '210726'

path = './csv/o_Univariate_feature_selection_LR({}).csv'.format(date)
# path = './csv/o_preprocessed_clinical_information({})_dummy_corr.csv'.format(date)
source_data_X = pd.read_csv(path, index_col='data_num')
path = './csv/i_PatientsList_217.csv'
source_data_Y  = pd.read_csv(path, index_col=0)
X = pd.DataFrame(source_data_X)
Y = pd.DataFrame(source_data_Y)


fold = X['fold']; fold.index = X.index
RFS_2y = X['2yRFS']
del X['fold']; del X['2yRFS']


def likelihood_ratio(llf_with, llf_without):
    return (2*(llf_with-llf_without))  # -2ln(llf_with/llf_without) = 2[log(llf_without) - log(llf_with)]


## Pearson correlation coefficient 검정
import seaborn as sns

X_corr = X.corr()
cmap = sns.light_palette("darkgray", as_cmap=True)
sns.heatmap(X_corr, annot=True, cmap='Blues')
# plt.show()
# X = X.drop(['lymphnode_pathologic_spread'], axis='columns') # 상관관계 높은 feature 중 하나 제외

### VIF coefficient 검정
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print(vif)

X_vif = X.loc[:, ['tumor_stage', 'lymphnode_pathologic_spread']]
X_vif = sm.add_constant(X_vif)

OLS_total = sm.OLS(RFS_2y, X_vif)

# res_total = OLS_total.fit()
# print(res_total.summary())


X_vif = pd.concat([X, RFS_2y], axis='columns')
X_corr = X_vif.corr()
cmap = sns.light_palette("darkgray", as_cmap=True)
sns.heatmap(X_corr, annot=True, cmap=cmap)
plt.show()


# Logistic regression model fitting
improvement = 0
X_best_subset = X
X_temp = sm.add_constant(X_best_subset.drop(['lymphnode_pathologic_spread'],axis='columns'))     # 절편 추가
prev_model = sm.Logit(RFS_2y,X_temp)
prev_result = prev_model.fit_regularized(disp=True)

result_save = pd.DataFrame()
result_save['col'] = X_temp.columns
result_save['odds_r'] = np.exp(prev_result.params).tolist()
result_save['CI_025'] = np.exp(prev_result.params-prev_result.bse*1.96).tolist()
result_save['CI_975'] = np.exp(prev_result.params+prev_result.bse*1.96).tolist()
result_save['pval'] = (prev_result.pvalues).tolist()
# prev_AIC = prev_result.aic
result_save.to_csv('./csv/o_Multiavariate_analysis_results ({}).csv'.format(date))

print(prev_result.summary())
print(np.exp(prev_result.params))
print(prev_result.pvalues)


'''
Logistic regression for Multivariate analysis

in summary table()
* coef : the coefficients of the independent variables in the regression equation.
* Log-Likelihood : the natural logarithm of the Maximum Likelihood Estimation(MLE) function. MLE is the optimisation process of finding the set of parameters which result in best fit.
* LL-Null : the value of log-likelihood of the model when no independent variable is included (only an intercept is included)
* Pseudo R-squ : a substitute for the R-square value in Least Squares linear regression. It is the ratio of the log-likelihood of the null model to that of the full model.
'''
while improvement > 0:# or improvement == 0 : #improvement >= 0:
    pval_list = pd.DataFrame(columns=['col','LR','pval'])
    print('------------------------------------------')
    print('- feature numbers of best subset : {} '.format(len(X_best_subset.columns)))
    X_best_subset_prev = X_best_subset

    '''
    Likelihood rate test
    : 우도비 검정은 두 모델의 우도 비를 이용한 검정 방법이다.
    : 이때 매개변수를 더 가진 모델을 full model이라고 하며, 매개변수가 더 적은 쪽을 reduced model이라고 한다.
    '''
    for i in X_best_subset:
        X_with = sm.add_constant(X_best_subset)
        X_without = sm.add_constant(X_best_subset.drop(i, axis='columns'))

        model_with = sm.Logit(Y, X_with)
        result_with = model_with.fit(disp=False)
        L_w = result_with.llf

        model_without = sm.Logit(Y, X_without)
        result_without = model_without.fit(disp=False)
        L_wo = result_without.llf

        LR = likelihood_ratio(L_w, L_wo)

        '''
         모형에 대한 검정- 카이제곱검정
        '''
        p = chi2.sf(LR, 1)       # chi2.sf(chi_squred, degrees_of_reedom)
        pval_list = pval_list.append({'col':i, 'LR':LR, 'pval':p}, ignore_index=True)

    p_max = pval_list.iloc[pval_list['pval'].idxmax()]
    print(p_max)

    # 가장 쓸모없는 risk factor가 제거된 모델과 포함된 모델의 AIC 값 차이 계산
    '''
    AIC : 로지스틱 회귀 모델을 사용할 때 모형의 설명력을 측정해주는 척도 (작을수록 좋다)
     - X_best_subset_prev : 제거되지 않은 모델
     - X_best_subset_post : 제거된 모델
    '''
    X_best_subset_post = X_best_subset

    if p_max.pval > 0.05:
        X_best_subset_post = X_best_subset_post.drop(p_max.col, axis='columns')
        X_best_subset = X_best_subset_post
    else:
        print('**********************************')
        print('p-val is under 0.05!')
        print(p_max.col)
        print('**********************************')
        break


    X_temp = sm.add_constant(X_best_subset_post)
    model_subset = sm.Logit(Y, X_temp)
    result = model_subset.fit(disp=False)
    print('$$$ removed model - {} $$$'.format(p_max.col))
    print(result.summary())

    # improvement = prev_AIC - result.aic
    # X_best_subset = X_best_subset_post
    # if improvement < 0 :
    #     print("모델 설명력 개선 없음 | {}".format(improvement))
    #     X_best_subset = X_best_subset_prev
    # else:
    #     print("모델 설명력 개선 있음 | {}".format(improvement))
    #     X_best_subset = X_best_subset_post
    #
    # print('{} - {} = {}'.format(prev_AIC, result.aic, improvement))
    # prev_AIC = result.aic

print('Features({}) {} \n'.format(len(X_best_subset.columns),X_best_subset.columns))

final_featureset = sm.add_constant(X_best_subset)
final_model = sm.Logit(Y,final_featureset)
final_result = final_model.fit()

print(final_result.summary())
#ODDS_str = '{}(95% CI, {}-{}'.format(z)
print(np.exp(final_result.params))

final_featureset = pd.concat([final_featureset, RFS_2y], axis = 1)
del final_featureset['const']
final_featureset_fold = pd.concat([final_featureset, fold], axis = 1)
final_featureset.to_csv('./csv/o_feature_selection_LR({}).csv'.format(date))

fold = ['A','B','C','D','E']

for i in range(5):
    save_by_fold_test = final_featureset[final_featureset_fold['fold']==(i+1)]
    save_by_fold_train = final_featureset[final_featureset_fold['fold']!=(i+1)]

    save_by_fold_train.to_csv('./csv/fold({})/cln_testfold({})_train.csv'.format(date, date, fold[i]))
    save_by_fold_test.to_csv('./csv/fold({})/cln_testfold({})_test.csv'.format(date, date, fold[i]))


'''
------------------------------------------------------------------------------------------------------------------------------------------------------------------
○ Theory of logistic regression feature selection using backward elimination
 - https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faqhow-are-the-likelihood-ratio-wald-and-lagrange-multiplier-score-tests-different-andor-similar/
 - https://link.springer.com/article/10.1186/1751-0473-3-17
○ Statsmodel : https://tedboy.github.io/statsmodels_doc/generated/generated/statsmodels.api.Logit.html#
    examples 
     - https://zetawiki.com/wiki/Statsmodels_%EB%A1%9C%EC%A7%80%EC%8A%A4%ED%8B%B1%ED%9A%8C%EA%B7%80%EB%B6%84%EC%84%9D
○ Loglikelihood 
 - https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.LogitResults.html
 - https://shlee1990.tistory.com/766
○ Calculate P-value : https://stackoverflow.com/questions/38248595/likelihood-ratio-test-in-python
------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
