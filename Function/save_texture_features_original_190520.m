function features=save_texture_features_original_190520(image,image_raw,mass,air)
addpath(genpath('GLRL'));

% 58개
features=zeros(7+5+14+22+10,size(image,1));

% Store features
for data_num=1:size(image,1)
    tumor_img=image{data_num};
    tumor_mass=mass{data_num};
    tumor_img_raw=image_raw{data_num};
    tmp = double(tumor_mass);
    tmp = tmp/255.0;  % binary으로 만들어줌
    tumor_timg=double(tumor_img).*double(tmp); % mask있는 region만 살려줌
    %tumor_timg=double(tumor_img).*double(tumor_mass);
    tumor_img_raw_reg=tumor_img_raw(tumor_mass==255); % chest setting image에서 tumor 있는 region만 추출
    
%     if(air==0)      % histogram feature 추출 시 너무 낮은 intensity 값을 가진 air 영역 제외 
%         tumor_img_raw_reg=tumor_img_raw(tumor_img_raw >= -300 & tumor_mass==255)
%         x=linspace(-300,150);
%         tumor_timg(tumor_img_raw<-300)=0; %A(A>10) = 10
%     end
    
    clear 'temp'
    tumor_features=zeros(size(features,1),1);
    %tumor_features_no_air=zeros(size(features,1),1);

    disp(data_num)
    % Basic histogram properties (7개)
    base_num=0;
    %tumor_feateure_no_air(base_num)=mean(tumor_img_no_air(:));
    tumor_features(base_num+1)=mean(tumor_img_raw_reg(:));
    tumor_features(base_num+2)=std(double(tumor_img_raw_reg(:)));
    tumor_features(base_num+3)=min(tumor_img_raw_reg(:));
    tumor_features(base_num+4)=max(tumor_img_raw_reg(:));
    tumor_features(base_num+5)=skewness(double(tumor_img_raw_reg(:)));
    tumor_features(base_num+6)=kurtosis(double(tumor_img_raw_reg(:)));
    tumor_features(base_num+7)=entropy(double(tumor_img_raw_reg(:)));

    % Percentile HU (5개)
    base_num=7;
    tumor_features(base_num+1)=prctile(double(tumor_img_raw_reg(:)),5);
    tumor_features(base_num+2)=prctile(double(tumor_img_raw_reg(:)),25);
    tumor_features(base_num+3)=prctile(double(tumor_img_raw_reg(:)),50);
    tumor_features(base_num+4)=prctile(double(tumor_img_raw_reg(:)),75);
    tumor_features(base_num+5)=prctile(double(tumor_img_raw_reg(:)),95);
    
    % GLCM (14개)
    base_num=12;
    temp=double(tumor_timg);

    %{
    glcm1=graycomatrix(temp,'GrayLimits',[100 255],...
        'NumLevels',64,'Offset',[0 1]);
    glcm2=graycomatrix(temp,'GrayLimits',[100 255],...
        'NumLevels',64,'Offset',[-1 1]);
    glcm3=graycomatrix(temp,'GrayLimits',[100 255],...
        'NumLevels',64,'Offset',[-1 0]);
    glcm4=graycomatrix(temp,'GrayLimits',[100 255],...
        'NumLevels',64,'Offset',[-1 -1]);
    %}
    
    % glcm(n): 각 길이 및 방향에 해당하는 동시발생행렬
    glcm1=graycomatrix(temp,'GrayLimits',[0 255],...
        'NumLevels',64,'Offset',[0 1]);
    glcm2=graycomatrix(temp,'GrayLimits',[0 255],...
        'NumLevels',64,'Offset',[-1 1]);
    glcm3=graycomatrix(temp,'GrayLimits',[0 255],...
        'NumLevels',64,'Offset',[-1 0]);
    glcm4=graycomatrix(temp,'GrayLimits',[0 255],...
        'NumLevels',64,'Offset',[-1 -1]);
    
    [f11,f21,f61,f71,f81,f91,f111]=glcm_features(glcm1,64);
    [f12,f22,f62,f72,f82,f92,f112]=glcm_features(glcm2,64);
    [f13,f23,f63,f73,f83,f93,f113]=glcm_features(glcm3,64);
    [f14,f24,f64,f74,f84,f94,f114]=glcm_features(glcm4,64);
    
    tumor_features(base_num+1)=mean([f11,f12,f13,f14]); %13
    tumor_features(base_num+2)=std([f11,f12,f13,f14]);
    tumor_features(base_num+3)=mean([f21,f22,f23,f24]);
    tumor_features(base_num+4)=std([f21,f22,f23,f24]);
    tumor_features(base_num+5)=mean([f61,f62,f63,f64]); %17
    tumor_features(base_num+6)=std([f61,f62,f63,f64]);%18
    tumor_features(base_num+7)=mean([f71,f72,f73,f74]);
    tumor_features(base_num+8)=std([f71,f72,f73,f74]);
    tumor_features(base_num+9)=mean([f81,f82,f83,f84]); %21
    tumor_features(base_num+10)=std([f81,f82,f83,f84]); %
    tumor_features(base_num+11)=mean([f91,f92,f93,f94]); %23
    tumor_features(base_num+12)=std([f91,f92,f93,f94]);
    tumor_features(base_num+13)=mean([f111,f112,f113,f114]);
    tumor_features(base_num+14)=std([f111,f112,f113,f114]);
    
    % Run-length (22개)
    base_num=12+14; %38
    temp=double(tumor_timg);
    grlm=grayrlmatrix(temp,'GrayLimits',[0 255],'NumLevels',256);
    grstat=grayrlprops(grlm);
    
    tumor_features(base_num+1)=mean(grstat(:,1)); %27
    tumor_features(base_num+2)=std(grstat(:,1));
    tumor_features(base_num+3)=mean(grstat(:,2));
    tumor_features(base_num+4)=std(grstat(:,2));
    tumor_features(base_num+5)=mean(grstat(:,3)); %31
    tumor_features(base_num+6)=std(grstat(:,3));
    tumor_features(base_num+7)=mean(grstat(:,4)) %33
    tumor_features(base_num+8)=std(grstat(:,4));
    tumor_features(base_num+9)=mean(grstat(:,5));
    tumor_features(base_num+10)=std(grstat(:,5)); %36
    tumor_features(base_num+11)=mean(grstat(:,6));
    tumor_features(base_num+12)=std(grstat(:,6));
    tumor_features(base_num+13)=mean(grstat(:,7)); %39
    tumor_features(base_num+14)=std(grstat(:,7));
    tumor_features(base_num+15)=mean(grstat(:,8));
    tumor_features(base_num+16)=std(grstat(:,8)); %
    tumor_features(base_num+17)=mean(grstat(:,9));
    tumor_features(base_num+18)=std(grstat(:,9));%44
    tumor_features(base_num+19)=mean(grstat(:,10));
    tumor_features(base_num+20)=std(grstat(:,10));
    tumor_features(base_num+21)=mean(grstat(:,11)); %48
    tumor_features(base_num+22)=std(grstat(:,11));
    
    clear 'temp' 'glcm' 'prop'
    clear 'glcm1' 'glcm2' 'glcm3' 'glcm4'
    clear 'grlm' 'grstat'
    clear 'f11' 'f21' 'f61' 'f71' 'f81' 'f91' 'f111'
    clear 'f12' 'f22' 'f62' 'f72' 'f82' 'f92' 'f112'
    clear 'f13' 'f23' 'f63' 'f73' 'f83' 'f93' 'f113'
    clear 'f14' 'f24' 'f64' 'f74' 'f84' 'f94' 'f114'
    
    % LBP features (59개) -> (10개)
    %base_num=24+14+22; %60
    %tumor_features((base_num+1):(base_num+59))=double(extractLBPFeatures(uint8(tumor_timg),'Upright',true,'Radius',1));
    base_num=12+14+22; %60
    tumor_features((base_num+1):(base_num+10))=double(extractLBPFeatures(uint8(tumor_timg),'Upright',false,'Radius',1));

    clear 'thresh' 'temp'
  
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
end
clear 'test_num' 'tumor_slices'
clear 'tumor_num' 'num_of_data'
clear 'num_of_slice' 'num_slice'
clear 'rand_data' 'rand_flip' 'rand_rotate'
clear 'train_ind_AML' 'train_ind_RCC'

features=transpose(features);

clear 'temp' 'rownum'
clear 'rect_tumor_all' 'rsna_ind'