function features=save_shape_features_190130(image,mass,flag)
% addpath(genpath('GLRL'));

features=zeros(6+4+1,size(image,1));

% Store features
for data_num=1:size(image,1)
%     tumor_img=image(:,:,data_num);
    temp=mass{data_num};
    tumor_mass=(temp==255);    
    %clear 'temp'
    tumor_features=zeros(size(features,1),1);
    temp_props=regionprops(tumor_mass,'Area','Perimeter','ConvexArea','Eccentricity',...
        'EulerNumber','MajorAxisLength','MinorAxisLength','Solidity');
    
    % cuvrature feature k
    [ind_x,ind_y] = find(tumor_mass==1);
%     hold off
%     plot(ind_x, ind_y, '.');
%     hold on
    tumor_boundary = boundary(ind_x, ind_y, 1);
%     plot(ind_x(tumor_boundary), ind_y(tumor_boundary));
    boundaries = [ind_x(tumor_boundary),ind_y(tumor_boundary)];
    k=LineCurvature2D(boundaries); 
    
    % 잘 뽑히는지 plot으로 출력 (의미없음) 
    m = mean(abs(k));
%     fig = gcf;
%     if(flag==0)
%         saveas(fig,['./boundary/9mm/',int2str(m*1000),'_non_',int2str(data_num),'.png']) 
%     elseif(flag==1)
%         saveas(fig,['./boundary/9mm/',int2str(m*1000),'_rec_',int2str(data_num),'.png']) 
%     end
%     clear tumor_boundary ind_x ind_y
    
    [R cx cy]=max_inscribed_circle(temp,0);
    
    tumor_features(1)=temp_props(1).Area/temp_props(1).Perimeter; % (1) Area/Perimeter Ratio
    tumor_features(2)=temp_props(1).ConvexArea; % (2) Convex Area
    tumor_features(3)=temp_props(1).Eccentricity; % (3) Eccentricity
    tumor_features(4)=temp_props(1).EulerNumber; % (4) Euler #
    tumor_features(5)=temp_props(1).Solidity; % (5) Solidity
    tumor_features(6)=temp_props(1).MajorAxisLength/temp_props(1).MinorAxisLength; % Major-minor axis ratio
    
    tumor_features(7)=temp_props(1).MajorAxisLength; % (Size 1) Major Axis Length
    tumor_features(8)=temp_props(1).MinorAxisLength; % (Size 2) Minor Axis Length
    tumor_features(9)=temp_props(1).Area; % (Size 3) Area
    tumor_features(10)=temp_props(1).Perimeter; % (Size 4) Perimeter
    tumor_features(11)=mean(m);
%     [temp,temp1]=compute_mean_curvature(tumor_mass);
%     tumor_features(8)=temp; % (8) Min Curvature
%     tumor_features(9)=temp1;
    
    features(:,data_num)=tumor_features;
    clear 'tumor_img'
    clear 'tumor_img_region' 'tumor_img_black'
    clear 'tumor_roi'
    clear 'base_num'
    
    clear 'tumor_hist' 'tumor_features'
    clear 'data_num' 'data_id'
    clear 'img_raw' 'img_gray'
    clear 'roi_cord' 'roi' 'roi_img' 'temp_img'
    clear 'temp_tumor'
    clear 'slicenum'
    clear 'rect_tumor'
    clear 'roi_cont'
    clear 'temp_props'
end
clear 'test_num' 'tumor_slices'
clear 'tumor_num' 'num_of_data'
clear 'num_of_slice' 'num_slice'
clear 'rand_data' 'rand_flip' 'rand_rotate'
clear 'train_ind_AML' 'train_ind_RCC'

features=transpose(features);

% features_norm=zeros(size(features));
% for colnum=1:size(features,2)
%     temp=features(:,colnum);
%     features_norm(:,colnum)=(temp-min(temp))/(max(temp)-min(temp)+eps);
% end
clear 'temp' 'rownum'
clear 'rect_tumor_all' 'rsna_ind'