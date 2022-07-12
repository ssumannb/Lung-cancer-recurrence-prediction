cd('E:/research/Prediction/NSCLC/Codes/')
dir_base = ['./Log/',date,'/'];
Experiment_setting = strcat('/classifiy_HCF_',flg,'-',mm);
Experiment_setting = convertCharsToStrings(Experiment_setting)

% 5-fold cross validation
K_cv=5;

% Random-forest 
num_of_rf_trees=60;
% Relief-F%
k_of_relieff=35;
% Nca
num_of_all_feat = 69;

str = [dir_base,'/mat_p_hcf/',flg,'_', mm,'/hcf_g1.mat'];
load(str);

if strcmp(flg,'intra')
    num_g1 = size(hcf, 1); 
    hcf_g1 = hcf(1:num_g1, :);
elseif strcmp(flg, 'peri') 
    num_of_all_feat = 58;
    num_g1 = size(hcf_peri, 1);
    hcf_g1 = hcf_peri(1:num_g1, :);
elseif strcmp(flg, 'comb') 
    num_of_all_feat = 127;
    num_g1 = size(hcf_comb, 1);
    hcf_g1 = hcf_comb(1:num_g1, :);
end

feat_weights_nca=zeros(num_of_all_feat,K_cv);

%str = [dir_base, 'Codes/mat_p_hcf/hcf_g2.mat'];
str = [dir_base,'/mat_p_hcf/',flg,'_', mm,'/hcf_g2.mat'];
load(str);

if strcmp(flg,'intra')
    num_g2 = size(hcf, 1); 
    hcf_g2 = hcf(1:num_g2, :);
elseif strcmp(flg, 'peri') 
    num_g2 = size(hcf_peri, 1);
    hcf_g2 = hcf_peri(1:num_g2, :);
elseif strcmp(flg, 'comb') 
    num_g2 = size(hcf_comb, 1);
    hcf_g2 = hcf_comb(1:num_g2, :);
end

clear 'str'

%% To train a classifier for each fold
disp('Training classifiers...');
fold_num_g1 = 1;
fold_num_g2 = 1;
cv_g1 = fix(num_g1/K_cv)+1;
cv_g2 = fix(num_g2/K_cv)+1;

A_fold = [hcf_g1(1:cv_g1,:); hcf_g2(1:cv_g2,:)];                        A_fold_gt = [ones(size(hcf_g1(1:cv_g1,:),1),1); 2*ones(size(hcf_g2(1:cv_g2,:),1),1)];
B_fold = [hcf_g1(cv_g1+1:cv_g1*2,:); hcf_g2(cv_g2+1:cv_g2*2,:)];        B_fold_gt = [ones(size(hcf_g1(cv_g1+1:cv_g1*2,:),1),1); 2*ones(size(hcf_g2(cv_g2+1:cv_g2*2,:),1),1)];
C_fold = [hcf_g1(cv_g1*2+1:cv_g1*3,:); hcf_g2(cv_g2*2+1:cv_g2*3,:)];    C_fold_gt = [ones(size(hcf_g1(cv_g1*2+1:cv_g1*3,:),1),1); 2*ones(size(hcf_g1(cv_g2*2+1:cv_g2*3,:),1),1)];
D_fold = [hcf_g1(cv_g1*3+1:cv_g1*4,:); hcf_g2(cv_g2*3+1:cv_g2*4,:)];    D_fold_gt = [ones(size(hcf_g1(cv_g1*3+1:cv_g1*4,:),1),1); 2*ones(size(hcf_g2(cv_g2*3+1:cv_g2*4,:),1),1)];
E_fold = [hcf_g1(cv_g1*4+1:num_g1,:); hcf_g2(cv_g2*4+1:num_g2,:)];      E_fold_gt = [ones(size(hcf_g1(cv_g1*4+1:num_g1,:),1),1); 2*ones(size(hcf_g2(cv_g2*4+1:num_g2,:),1),1)];

clear 'num_g1' 'num_g2' 'cv_g1' 'cv_g2'

ABC_fold = {A_fold, B_fold, C_fold, D_fold, E_fold};                %전체 데이터를 담은 folder
ABC_gt = {A_fold_gt, B_fold_gt, C_fold_gt, D_fold_gt, E_fold_gt};   %전체 Ground truth를 담은 folder
n_fold = {}; n_gt = {};                                             %순서 바꿔주는 임시 folder
cv_result_svm = {}; cv_result_rf = {}; cv_result_esbl = {};
cv_train_feature = {}; cv_test_feature = {};
cv_train_feature_selected = {}; cv_test_feature_selected = {};
cv_feature_weight = {};

%% waitbar
mwb = MultiWaitBar(3,1,'Nested loop demo...', 'g');
% initialize wait bars
loopName = {'Folder loop progress...', 'feature number loop progress...', 'Random forest loop progress...'};
for ix = 1:3 % initialize waitbars
    mwb.Update(ix, 1, 0, loopName{ix});
end

Fold_arr = ["test-E", "test-D", "test-C", "test-B", "test-A"]
for i = 1:K_cv
    fprintf('Fold %d \n', i);
    mwb.Update(1,1,i/5,strcat('Fold loop',num2str(i)),[1,0,0]); 

    foldnum=[1,2,3,4,5];
    Fold_train_feature = [cell2mat(ABC_fold(foldnum(1))); cell2mat(ABC_fold(foldnum(2))); cell2mat(ABC_fold(foldnum(3))); cell2mat(ABC_fold(foldnum(4)))];
    Fold_train_gt = [cell2mat(ABC_gt(foldnum(1))); cell2mat(ABC_gt(foldnum(2))); cell2mat(ABC_gt(foldnum(3))); cell2mat(ABC_gt(foldnum(4)))];
    Fold_test_feature = [cell2mat(ABC_fold(foldnum(5)))];
    Fold_test_gt = [cell2mat(ABC_gt(foldnum(5)))];
   
    % Normalize features
    foffset=min(Fold_train_feature);
    fslope=max(Fold_train_feature)-foffset+eps;
    Fold_train_feature=(Fold_train_feature-repmat(foffset,size(Fold_train_feature,1),1))./repmat(fslope,size(Fold_train_feature,1),1);
    Fold_train_feature(isnan(Fold_train_feature))=eps;
    Fold_test_feature=(Fold_test_feature-repmat(foffset,size(Fold_test_feature,1),1))./repmat(fslope,size(Fold_test_feature,1),1);
    Fold_test_feature(isnan(Fold_test_feature))=eps;
    %test_size=size(Fold_test_feature(:,1)); test_size=test_size(:,1);
    clear 'foffset' 'fslope'
    
    %%%% signficant feature number select %%%%
    % Feature selection:nca # calculate the feature weight
    mdl=fscnca(Fold_train_feature, Fold_train_gt);%,'Solver','sgd','Verbose',1); %
    feat_weights_nca(:, i)=mdl.FeatureWeights;

    grid on
    [~, feat_sel_ind]=sort(mdl.FeatureWeights,'descend');
    
    sgfnum_result_svm = zeros(num_of_all_feat-1,6); sgfnum_result_rf = zeros(num_of_all_feat-1,6); 
    sgfnum_result_esbl = zeros(num_of_all_feat-1,6); %sgfnum = significant feature number 

    idx=1;
    for num_of_selected_features = 2:5%num_of_all_feat  % 
        mwb.Update(2,1,idx/num_of_all_feat,strcat('feature number',num2str(num_of_selected_features)),[0,1,0]);
        % Feature selection: nca  # seperate the selected features by ranking
        Fold_train_feature_selected=Fold_train_feature(:, feat_sel_ind(1:num_of_selected_features));
        Fold_test_feature_selected=Fold_test_feature(:, feat_sel_ind(1:num_of_selected_features));      
 
       %% Classifier 1 : SVM classifiers
        mdl_svm=fitcsvm(Fold_train_feature_selected,Fold_train_gt,'Standardize',true,'KernelFunction','RBF',...
      'KernelScale','auto');
        
        % posterior probability of svm model
        CompactSVMModel = compact(mdl_svm);
        CompactSVMModel = fitPosterior(CompactSVMModel,...
        Fold_train_feature_selected, Fold_train_gt);

%         mdl_svm =fitcsvm(Fold_train_feature_selected,Fold_train_gt, 'OptimizeHyperparameters','auto',...
%            'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%            'expected-improvement-plus', 'Showplots', false));   % 교차 검증된 모델 객체로 반환 % 베이지안 최적화 기법으로 최적화된 하이퍼파라미터를 찾음
        
        % predict SVM
        [temp,temp1]=predict(mdl_svm,Fold_test_feature_selected);
        [temp2,temp3]=predict(CompactSVMModel, Fold_test_feature_selected);
        
        test_label_svm=temp;
        test_score_svm=temp1;
        test_prob_svm=temp3;

        clear 'temp' 'temp1' 'temp2'
        test_result_svm=compute_accuracy2_180514(test_label_svm,test_score_svm,Fold_test_gt);
        sgfnum_result_svm(idx,:) = test_result_svm;

        rf_iter = 20;
        test_result_rf_all = zeros(rf_iter,6);
        test_label_rf_all = {};
        test_score_rf_all = {};

         %% Classifier 2 : Random forest classifiers
        
        parfor j = 1:rf_iter
            rf_iter = 20;
            %predict RF
            mwb.Update(3,1,j/rf_iter,strcat('Random forest iter',num2str(j)),[0,0,1]);
            if mod(j,10)==0
                fprintf("\t\t-rf_iter %d \n",j)
            end
            mdl_rf = TreeBagger(num_of_rf_trees, Fold_train_feature_selected, Fold_train_gt,'Method','classification','Surrogate','on',...
                'PredictorSelection','curvature','OOBPredictorImportance','on');
%             mdl_rf = fitctree(Fold_train_feature_selected,Fold_train_gt,'OptimizeHyperparameters','auto',...
%                'HyperparameterOptimizationOptions',struct('Showplots', false)) % 자동 하이퍼파라미터 최적화
            
            [temp,temp1]=predict(mdl_rf,Fold_test_feature_selected);
            temp_label_rf=str2double(temp);
            temp_score_rf=temp1;
            temp_eval_rf=compute_accuracy2_180514(temp_label_rf,temp_score_rf,Fold_test_gt);

            test_result_rf_all(j,:) = temp_eval_rf;
            test_label_rf_all(j,:) = {temp_label_rf};
            test_score_rf_all(j,:) = {temp_score_rf};
        end
        %%
         % 평균 result와 가까운 값을 가진 행을 찾고 인덱스 반환
        result_rf = mean(test_result_rf_all);
        scaling_comp = [1.5, 0.1, 0.1, 0.1, 0.1, 200];
        scaled_result_rf = result_rf.*scaling_comp;
        scaled_results = test_result_rf_all.*scaling_comp;
        for idx_rf_loop_norm = 1:rf_iter
            similarity(idx_rf_loop_norm,:) = norm(scaled_result_rf-scaled_results(idx_rf_loop_norm,:));
        end
           
        [M, I] = min(similarity);
           
         sgfnum_result_rf(idx,:) = result_rf; %result_rf_temp;

         % 평균 result와 가장 근사한 값을 가진 결과에 해당하는 label, score 할당
         test_label_rf = cell2mat(test_label_rf_all(I,:));
         test_prob_rf = cell2mat(test_score_rf_all(I,:));
         
         %% Classifier 3 : Ensemble of SVM and random forest classifiers
         test_prob_esbl = (test_prob_svm + test_prob_rf)/2;  
         test_label_esbl = dtmEsmblLabel(test_prob_esbl);
 
         temp_result_esbl=compute_accuracy2_180514(test_label_esbl,test_prob_esbl,Fold_test_gt);
         sgfnum_result_esbl(idx,:) = temp_result_esbl;     

%          save_path_byfeatnum = strcat(path_save_fold, num2str(num_of_selected_features), '_features used/');
%          save_byfeatnum = strcat(save_path_byfeatnum, p_regionType,'_testfold-', fold, '_svm_rf_esbl.mat');
% 
%          if(isfolder(save_path_byfeatnum)==0)
%              mkdir(save_path_byfeatnum);
%          end


        %%
         % Save results
         disp('Saving results...');
         dirc=strcat(result_save_path, Fold_arr(i),'-fold/',num2str(num_of_selected_features),' num_feature');
         str=[dirc];
         if(isfolder(str)==0)
             mkdir(str);
         end
         str1=strcat(str, Experiment_setting, '-', Fold_arr(i));
         cv_train_feature_selected = [cv_train_feature_selected; Fold_train_feature_selected];
         cv_test_feature_selected = [cv_test_feature_selected; Fold_test_feature_selected];
         save(str1,'test_label_rf','test_prob_rf','result_rf','I',...
             'test_label_rf_all','test_score_rf_all','test_result_rf_all',...
             'test_label_svm','test_score_svm','test_result_svm','test_prob_svm',...
             'temp_result_esbl','test_label_esbl','test_prob_esbl',...
             'Fold_test_gt')
         idx = idx+1;
                
        end


    cv_result_svm = [cv_result_svm; sgfnum_result_svm];
    cv_result_rf = [cv_result_rf; sgfnum_result_rf];
    cv_result_esbl = [cv_result_esbl; sgfnum_result_esbl];
    cv_train_feature = [cv_train_feature; Fold_train_feature];
    cv_test_feature = [cv_test_feature; Fold_test_feature];
    
    % Save results
    disp('Saving results...');
    dirc=strcat(result_save_path, Fold_arr(i),'-fold');
    str=[dirc];
    if(isfolder(str)==0)
        mkdir(str);
    end
    str1=strcat(str, strcat(Experiment_setting,'_total'));
    save(str1, 'feat_weights_nca','cv_train_feature','cv_test_feature','cv_train_feature_selected','cv_test_feature_selected',...,
        'cv_result_svm','cv_result_rf','cv_result_esbl')
    
   %% Fold Change
   n_fold = {cell2mat(ABC_fold(5)), cell2mat(ABC_fold(1)), cell2mat(ABC_fold(2)), cell2mat(ABC_fold(3)), cell2mat(ABC_fold(4))};
   n_gt = {cell2mat(ABC_gt(5)), cell2mat(ABC_gt(1)), cell2mat(ABC_gt(2)), cell2mat(ABC_gt(3)), cell2mat(ABC_gt(4))};
   ABC_fold = n_fold; ABC_gt = n_gt;
   mwb.Update(1,1,i/5,'ColorMap',[1,0,0]);
end


function y = dtmEsmblLabel(x)
    y = zeros(size(x,1), 1);
    x1 = x(:,1); x2 = x(:,2);
    
    for i = 1:size(x,1)
        y(i) = 1*(x1(i)>x2(i)) + 2*(x1(i)<x2(i));
        if x1(i) == x2(i)
            error('Cannat determine the ensemble lable');
        end
    end
        
end