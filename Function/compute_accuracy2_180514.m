function test_accuracy=compute_accuracy2_180514(labels_predict,score_predict,labels_gt)

% labels_predict=double(labels_predict==3)+1;
% labels_gt=double(labels_gt==3)+1;

% Compute accuracy, sensitivity, specificity, ppv, npv, and AUC
test_accuracy=zeros([1,6]);
test_accuracy(1)=100*sum(labels_predict==labels_gt)/length(labels_gt);
test_accuracy(2)=100*sum(labels_predict(labels_predict==2)==labels_gt(labels_predict==2))/(eps+sum(labels_gt==2));
test_accuracy(3)=100*sum(labels_predict(labels_predict==1)==labels_gt(labels_predict==1))/(eps+sum(labels_gt==1));
test_accuracy(4)=100*sum(labels_predict(labels_predict==2)==labels_gt(labels_predict==2))/(eps+sum(labels_predict==2));
test_accuracy(5)=100*sum(labels_predict(labels_predict==1)==labels_gt(labels_predict==1))/(eps+sum(labels_predict==1));
[~,~,~,temp]=perfcurve(labels_gt,score_predict(:,2),2);
test_accuracy(6)=temp;
clear 'temp'
% [~,~,~,labels_predict2]=perfcurve(labels_gt,scores_predict_svm,1);
% test_accuracy(6)=labels_predict2;
% clear 'labels_predict'