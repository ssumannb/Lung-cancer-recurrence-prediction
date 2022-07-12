cd('E:/research/Prediction/NSCLC/Codes/')
dir_base = 'E:/research/Prediction/NSCLC/Codes';
path_dirbase_csv = './Clinical_Radiomic_combine/';


%%% loop index
loop_primary = ["cln";"intra";"peri";"comb"];%"intra";"peri";"comb"];%;"comb"];
loop_peri = ["3mm"; "12mm"];
loop_comb = ["6mm"; "9mm"];
%%% i_ref_B version
%"3mm";"6mm";"9mm";"12mm";"15mm";"18mm";"21mm";"24mm";"27mm";"30mm"
%loop_peri = ["6mm";"9mm";"15mm";"18mm";"21mm";"24mm";"27mm";"30mm"];
%loop_comb = ["3mm";"12mm";"15mm";"18mm";"21mm";"24mm";"27mm";"30mm"]; %comb3mm 까지는 돌림

%global loop_fold;
loop_fold = ["A"; "B"; "C"; "D"; "E"];

%%% clincal, intra, peri, comb classification 수행을 위한 정보
%%% fold별 classifier
%global base_info_csv; global base_info;
base_info_csv = strcat(path_dirbase_csv,'i_ref_C.csv');     % 정보 저장된 csv file
base_info = readtable(base_info_csv);                       % csv file read
clf_list = table2array(base_info(:,{'type','clf'}));        % fold별 classifier 정리된 list

%%% classification을 진행해야 하는 classfier(clinical/intra/peri/comb) 식별 후
%%% foldLoop 수행
for idx_type = 1:size(loop_primary)
    region_type = loop_primary(idx_type);
    %z_mwb.Update(1,1,idx_type/size(loop_primary),strcat('Meta loop',num2str(region_type)),[1,0,0]); 
    result_intra = zeros(5,6); %sgfnum_result_svm = zeros(15,6);
    clf = [];
    if region_type == 'intra'
        clf = clf_list(clf_list(:,1)==region_type,2);
        foldLoop(region_type, clf);
        fprintf('region %s classification end\n', region_type);
    end
    if region_type == 'peri'
        for idx_mm=1:size(loop_peri)
            region_type='peri';
            region_type = strcat(region_type, loop_peri(idx_mm,1));
            clf = clf_list(clf_list(:,1)==region_type,2);
            foldLoop(region_type, clf);
            fprintf('region %s classification end\n', region_type)
        end
    end
    if region_type == 'comb'
        for idx_mm=1:size(loop_comb)
            region_type='comb';
            region_type = strcat(region_type, loop_comb(idx_mm,1));
            clf = clf_list(clf_list(:,1)==region_type,2);
            foldLoop(region_type, clf);
            fprintf('region %s classification end\n', region_type)
        end
    end
    if region_type == 'cln'
        clf = cellstr(["rfsvm";"rfsvm";"rfsvm";"rfsvm";"rfsvm"]);
        foldLoop(region_type, clf, "_rf");
        fprintf('region %s classification end\n', region_type)
%         clf = cellstr(["svm";"svm";"svm";"svm";"svm"]);
%         foldLoop(region_type, clf, "_svm");
%         fprintf('region %s classification end\n', region_type)
    end
   
end


%%% foldLoop
%%% fold별로 (feature 2개 사용부터 all 사용까지) classification 수행
function foldLoop(p_regionType, p_clf, varargin)

    %%% 변수 공유 안돼서 다시 선언
    date = "210510";
    path_dirbase_csv = './Clinical_Radiomic_combine/';
    path_save_base = strcat(path_dirbase_csv,'output(',date,')/',p_regionType,'/');
    loop_fold = ["A"; "B"; "C"; "D"; "E"];
    %%%
    
    opt = '_';
    if nargin == 3
        opt = varargin{1};
    end
    % Random-forest 
    
    % fold별 사용한 radiomic feature 개수 불러오기
    base_info_csv = strcat(path_dirbase_csv,'i_ref_C.csv');                     % 정보 담긴 csv file
    base_info = readtable(base_info_csv);                                       % csv file read
    feat_num_list = base_info{base_info{:,'type'}==p_regionType,'feature'};     % fold별 사용 radiomic feature개수 list로 저장
    
    cln_rad_feat_weight={};
    cv_result = {}; 
    cv_train_feature = {}; cv_test_feature = {};
    cv_train_feature_selected = {}; cv_test_feature_selected = {};
    
    fprintf('region %s classification start\n', p_regionType)
    
    %% fold interation
    for idx_fold = 1:size(loop_fold)
        fold = loop_fold(idx_fold);
        fprintf(' > %s fold test start \n',fold)
        path_save_fold = strcat(path_save_base, fold, '_fold/');
        
        if(isfolder(path_save_fold)==0)
         mkdir(path_save_fold);
        end
        
       %% train/test data read
        %z_mwb.Update(2,1,cnt/15,strcat('Fold loop: test fold',fold),[0,1,0]);
        if p_regionType == 'cln'
            %%% clinical features (VPI, tumor stage)
            csv_path_train = strcat(path_dirbase_csv,'fold(',date,')/', p_regionType,'_testfold(',fold,')_train.csv');  % csv that clinical feaeture saved (train)
            csv_path_test = strcat(path_dirbase_csv,'fold(',date,')/', p_regionType,'_testfold(',fold,')_test.csv');    % csv that clinical feaeture saved (test)
            trainset = readmatrix(csv_path_train);                                      % read clinical features (train)
            train_gt = trainset(:,end)+1; trainset(:,end) = []; trainset(:,1) = [];     % split the ground truth from trainset
            testset = readmatrix(csv_path_test);                                        % read clinical features (test)
            test_gt = testset(:,end)+1; testset(:,end) = []; testset(:,1) = [];         % split the ground truth from testset
            feature_number = 2;
        else
            %%% radiomic features used in this fold
            % A:1 B:2 C:3 D:4 E:5
            csv_path_train = strcat(path_dirbase_csv,'fold(',date,')/', 'trainfold_gt_',fold,'.csv');   % csv that ground truth saved (train)
            csv_path_test = strcat(path_dirbase_csv,'fold(',date,')/', 'testfold_gt_',fold,'.csv');     % csv that ground truth saved (test)
            cln_trainset = readmatrix(csv_path_train);                                                      % read ground truth (train)
            train_gt = cln_trainset(:,end)+1; cln_trainset(:,end) = []; cln_trainset(:,1) = [];             
            cln_testset = readmatrix(csv_path_test);
            test_gt = cln_testset(:,end)+1; cln_testset(:,end) = []; cln_testset(:,1) = [];                 % read ground truth (test)
            feature_number = feat_num_list(idx_fold);                                                       % assign the number of radiomic feature used
            
            
            %%% 사용된 radiomic feature가 개수별로 저장된 file
            mat_path = strcat(path_dirbase_csv, 'fold(', date,')/','classifiy_HCF_',p_regionType,'_total.mat');
            load(mat_path)
            total_rad_feat_num = 0;
            
            % fold 구분 없이 순서대로 저장되어 있어 classifier별 total radiomic feature개수 필요
            if contains(p_regionType,'intra')
                total_rad_feat_num = 69-1;
            elseif contains(p_regionType,'peri')
                total_rad_feat_num = 58-1;
            elseif contains(p_regionType,'comb')
                total_rad_feat_num = 127-1;
            end    
            % fold 구분 없이 저장되어 있어 (전체feature개수)*(fold순서)+(fold에서 사용한 feature개수-1)하여
            % 해당 fold에서 사용된 radiomic feature에 인덱싱 함
            % A:4 B:3 C:2 D:1 E:0 이도록 tmp_idx 맞춰줌
            tmp_idx = total_rad_feat_num*(5-idx_fold)+(feature_number-1);
            rad_trainset = cv_train_feature_selected{tmp_idx};
            rad_testset = cv_test_feature_selected{tmp_idx};

            % clinical feature와 radiomic feature 연결하여 trainset, testset 만들어줌
            trainset = horzcat(cln_trainset, rad_trainset);
            testset = horzcat(cln_testset, rad_testset);
                       
            %%% (additional) radiomic feature name extract
%             csv_path_rad_name = strcat(path_dirbase_csv,'fold(',date,')/',p_regionType,'_testfold(',fold,')_test.csv');
%             tmp_rad_name = readtable(csv_path_rad_name);
%             rad_name = tmp_rad_name.Properties.VariableNames;
%             rad_name = rad_name(1,2:end-1);
        end
        
       %% feature weight 산출
%         rad_name_tmp = rad_name.';          % feature name 
%         rad_name = string(rad_name_tmp);    % feature name
%         clear rad_name_tmp
%         
        mdl=fscnca(trainset, train_gt);     % NCA 
%         feat_weight_byfold = horzcat(rad_name,mdl.FeatureWeights);  % feature weight and name matching
%         div = [fold,' fold feature weight'];                        % fold별 divide index
%         div_fold = [div; feat_weight_byfold];                       % divide index와 feature weight matching
%         cln_rad_feat_weight = [cln_rad_feat_weight;div_fold];       % fold별 feature weight을 볼 수 있도록 결합

        [~, feat_sel_ind]=sort(mdl.FeatureWeights,'descend');       % weight 별로 sort


        %%% feature weight 저장
        % save_path_whg = strcat(save_path_base, p_regionType,'_testfold-', fold, '_weight.mat');
        % save(save_path_whg, 'feat_weight_byfold', 'feat_sel_ind');
        
        if p_regionType ~= 'cln'
            feature_number_total = feature_number+2;            % 사용된 clinical+radiomic feature 개수 
        else
            feature_number_total = feature_number;
        end 
        result_byfeatnum_svm = zeros(feature_number_total-1,7);     % feature 개수 별 결과 저장하는 변수 선언
        result_byfeatnum_rf = zeros(feature_number_total-1,7);    
        result_byfeatnum_esbl = zeros(feature_number_total-1,7);    
        
       %% feature 2 to all loop 
        fprintf('\t 2 to all feature test start \n');

        for num_of_selected_features = 2:feature_number_total
            
            % selected 된 feature 개수만큼 train/test dataset 구성
            trainset_selected_features=trainset(:, feat_sel_ind(1:num_of_selected_features));
            testset_selected_features=testset(:, feat_sel_ind(1:num_of_selected_features));      
            fprintf('\t\t test %d features\n', num_of_selected_features);
           
          %% classifier 1 : SVM 
            % svm model fitting
            mdl_svm=fitcsvm(trainset_selected_features,train_gt,'Standardize',true,'KernelFunction','rbf',...
                'KernelScale','auto'); %, 'Prior', 'empirical'); %,'CategoricalPredictors', (1:3)
            
            % hyper-parameter opt. version
%             mdl_svm = fitcsvm(trainset_selected_features,train_gt, 'OptimizeHyperparameters','auto',...
%             'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%             'expected-improvement-plus', 'Showplots', false));   % 교차 검증된 모델 객체로 반환 % 베이지안 최적화 기법으로 최적화된 하이퍼파라미터를 찾음

            % posterior probability of svm model
            CompactSVMModel = compact(mdl_svm);
            CompactSVMModel = fitPosterior(CompactSVMModel,...
            trainset_selected_features, train_gt);

            [temp,temp1]=predict(mdl_svm,testset_selected_features);
            [temp2,temp3]=predict(CompactSVMModel, testset_selected_features);
            test_label_svm=temp;
            test_score_svm=temp1;
            test_prob_svm=temp3;
            
            tmp_table = table(test_gt,temp2,temp3(:,1),temp3(:,2),'VariableNames',...
                {'TrueLabels','PredictedLabels(1)','PredictedLabels(2)','PosClassPosterior'});

            temp_result_svm=compute_accuracy2_180514(test_label_svm,test_score_svm,test_gt);
            %results(idx_fold,:) = temp_result_svm;
            result_byfeatnum_svm((num_of_selected_features-1),:) = [num_of_selected_features,temp_result_svm];

            clear 'temp' 'temp1' 'temp2' 'temp3'

           %% Claasifier 2 : Random Forest
            rf_iter = 15;
            num_of_trees = 60;
            result_rf_mean = zeros(rf_iter,6);
            test_label_rf_all = {};
            test_score_rf_all = {};

            parfor idx_rf_loop = 1:rf_iter
                mdl_rf = TreeBagger(60, trainset_selected_features, train_gt,'Method','classification','Surrogate','on',...
                    'PredictorSelection','curvature','OOBPredictorImportance','on');
%                 mdl_rf = fitctree(trainset_selected_features,train_gt,'OptimizeHyperparameters','auto',...
%                    'HyperparameterOptimizationOptions',struct('Showplots', false)); % 자동 하이퍼파라미터 최적화

                [temp,temp1]=predict(mdl_rf,testset_selected_features);
                temp_label_rf=str2double(temp);
                temp_score_rf=temp1;
                temp_eval_rf=compute_accuracy2_180514(temp_label_rf,temp_score_rf,test_gt)
                result_rf_mean(idx_rf_loop,:) = temp_eval_rf;
                test_label_rf_all(idx_rf_loop,:) = {temp_label_rf};
                test_score_rf_all(idx_rf_loop,:) = {temp_score_rf};
            end
            % 평균 result와 가까운 값을 가진 행을 찾고 인덱스 반환
            result_rf = mean(result_rf_mean);
            similarity_weight = [1.5, 0.1, 0.1, 0.1, 0.1, 200];
            weighted_result_rf = result_rf.*similarity_weight;
            weighted_results = result_rf_mean.*similarity_weight;
            for idx_rf_loop_norm = 1:rf_iter
               %kk = weighted_results(idx_rf_loop_norm,:);
                similarity(idx_rf_loop_norm,:) = norm(weighted_result_rf-weighted_results(idx_rf_loop_norm,:));
            end

            [M, I] = min(similarity);

            %results(idx_fold,:) = result_rf; %result_rf_temp;
            result_byfeatnum_rf((num_of_selected_features-1),:) = [num_of_selected_features,result_rf];

           % 평균 result와 가장 근사한 값을 가진 결과에 해당하는 label, score 할당
            test_label_rf = cell2mat(test_label_rf_all(I,:));
            test_score_rf = cell2mat(test_score_rf_all(I,:));
            test_prob_rf = cell2mat(test_score_rf_all(I,:));
            
           %% Classifier 3 : Ensemble of SVM and random forest classifiers
            test_prob_esbl = (test_prob_svm + test_prob_rf)/2;  
            test_label_esbl = dtmEsmblLabel(test_prob_esbl);
            
            temp_result_esbl=compute_accuracy2_180514(test_label_esbl,test_prob_esbl,test_gt);
            result_byfeatnum_esbl((num_of_selected_features-1),:) = [num_of_selected_features,temp_result_esbl];
            
            save_path_byfeatnum = strcat(path_save_fold, num2str(num_of_selected_features), '_features used/');
            save_byfeatnum = strcat(save_path_byfeatnum, p_regionType,'_testfold-', fold, '_svm_rf_esbl.mat');

            if(isfolder(save_path_byfeatnum)==0)
                mkdir(save_path_byfeatnum);
            end
            
            save(save_byfeatnum, 'temp_result_svm','result_rf','temp_result_esbl',...
                'test_label_svm','test_label_rf','test_label_esbl',...
                'test_prob_svm','test_score_svm','test_score_rf','test_prob_esbl',...
                'trainset','testset','train_gt','test_gt','trainset_selected_features','testset_selected_features',...
                'I','result_rf_mean','test_label_rf_all','test_score_rf_all');
                
        end
    fprintf(' > %s fold test end\n', fold);

    cv_result_svm = [loop_fold(idx_fold,1), "svm","","","","","";"featnum","ACC", "SEN", "SPEC", "PPV", "NPV","AUC";result_byfeatnum_svm];
    cv_result_rf = [loop_fold(idx_fold,1), "rf","","","","","";"featnum","ACC", "SEN", "SPEC", "PPV", "NPV","AUC";result_byfeatnum_rf];
    cv_result_esbl = [loop_fold(idx_fold,1), "esbl","","","","","";"featnum","ACC", "SEN", "SPEC", "PPV", "NPV","AUC";result_byfeatnum_esbl];
    
    
    cv_result = [cv_result; cv_result_svm, cv_result_rf, cv_result_esbl];
    
  
    end
    % Feature weight
%     save_path_whg = strcat(path_save_base, p_regionType,'_weight.csv');
%     writematrix(cln_rad_feat_weight, save_path_whg);
    
    save_path = strcat(path_dirbase_csv, 'output(',date,')/', p_regionType,'_clnNrad_result',opt,'.csv'); %output
    col = ["testfold", "clf", "ACC", "SEN", "SPEC", "PPV", "NPV","AUC"];
    %result_save = [col; loop_fold(:,1), p_clf(:,1), result_byfeatnum(:,:)]; result_save = table(result_save);
    result_save = table(cv_result);
    writetable(result_save, save_path);
    
    clear 'trainset' 'train_gt' 'testset' 'testgt' 
    clearvars -except 'date' 'dir_base_csv' 'loop_fold' 'num_of_rf_trees' 'path_save_base' 'opt' 'p_regionType' 'p_clf' 'varargin'
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