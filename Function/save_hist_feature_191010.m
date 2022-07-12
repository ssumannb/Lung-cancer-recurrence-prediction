function features=save_hist_feature_191010(image,image_raw,mass,color,air)

x = linspace(-1350, 150);

for data_num=1:size(image,1)
    tumor_img=image{data_num};
    tumor_mass=mass{data_num};
    tumor_img_raw=image_raw{data_num};
    tmp = double(tumor_mass);
    tmp = tmp/255.0; % binary으로 만들어줌
    tumor_timg=double(tumor_img).*double(tmp); % mask있는 region만 살려줌
    %tumor_timg=double(tumor_img).*double(tumor_mass);
    tumor_img_raw_reg=tumor_img_raw(tumor_mass==255); % chest setting에서 tumor이 있는 region만 추출..?
    if(air==0)
        tumor_img_raw_reg=tumor_img_raw(tumor_img_raw >= -300 & tumor_mass==255)
        %x=linspace(-300,150)
    end
       
    %% histogram
    %hc = histcounts(tumor_img_raw_reg, x)
    h = histogram(tumor_img_raw_reg, x)
    %h.BinCounts = linspace(0,1,4500)
    h.FaceColor = color
    h.FaceAlpha = 0.5
    h.EdgeColor = color
    h.EdgeAlpha = 0
    if(air==0)
        xlim([-300 150])
    end
    ylim([0 4500])
    %disp(h.Values)
    
    %% plot
    %interval=0:1:100
    %hc = histcounts(tumor_img_raw_reg, x)
    %plot(0:1:size(hc,1), hc, 'color' ,color)
    
   hold on
        
end

features=h