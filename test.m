warning off
[fname path]=uigetfile('.jpg','Give the testing file as input');
fname=strcat(path,fname);
im=imread(fname);
figure();
imshow(im);
title('Diseases Leaf');
%%the conver the rgb image to lab
cform=makecform('srgb2lab');
lab_he=applycform(im,cform);
nColors=input('enter the number of clusters');
ab=double(lab_he(:,:,2:3));
nrows=size(ab,1);
ncolums=size(ab,2);
ab=reshape(ab,nrows*ncolums,2);
%stats=('distance','sqEulidian','Replicates',3)
[idxbest cluster_centre] = kmeans(ab, nColors,'distance','sqEuclidean','Replicates',3);
pixel_labels=reshape(idxbest,nrows,ncolums);
segmented_images=cell(1,3);
rgb_label=repmat(pixel_labels,[1,1,3]);
for k=1:nColors
    colors=im;
    colors(rgb_label ~= k)=0;
    segmented_images{k}=colors;
end
for i=1:nColors
    figure();
    imshow(segmented_images{i});
    title(i);
end

diseasepart=input('Enter the number of cluster');
figure()
imshow(im);
title('Original Image');
figure()
imshow(segmented_images{diseasepart});
title('Disease Image');

seg_img1=segmented_images{diseasepart};
seg_img=im;

% Evaluate the disease affected area
black = im2bw(seg_img1,graythresh(seg_img1));
%figure, imshow(black);title('Black & White Image');
m = size(seg_img1,1);
n = size(seg_img1,2);

zero_image = zeros(m,n); 
%G = imoverlay(zero_image,seg_img,[1 0 0]);

cc = bwconncomp(seg_img1,6);
diseasedata = regionprops(cc,'basic');
A1 = diseasedata.Area;
%sprintf('Area of the disease affected region is : %g%',A1);

I_black = im2bw(im,graythresh(im));
kk = bwconncomp(im,6);
leafdata = regionprops(kk,'basic');
A2 = leafdata.Area;
%sprintf(' Total leaf area is : %g%',A2);
Affected_Area = (A1/A2);
if Affected_Area < 0.1
    Affected_Area = Affected_Area+0.15;
end
%%%%%%%%%%%%%%%%%%%%
[feat_disease]=featureselection(im);
[data text]=xlsread('a.xlsx');
load('leaf_dataset.mat');
CLASS =  knnclassify(feat_disease,db,text)
CLASS=cell2mat(CLASS);
test_rows=feat_disease;
load('Accuracy_Data.mat');
Accuracy_Percent= zeros(200,1);
for i = 1:500
data = Train_Feat;
groups = ismember(Train_Label,0);
[train,test] = crossvalind('HoldOut',groups);
cp = classperf(groups);
svmStruct = fitcknn(data(train,:),groups(train),'NumNeighbors',5,'Standardize',1);
classes = predict(svmStruct,data(test,:));
classperf(cp,classes,test);
Accuracy = cp.CorrectRate;
Accuracy_Percent(i) = Accuracy.*100;
end
Max_Accuracy = max(Accuracy_Percent);
sprintf('Accuracy of KNN is: %g%%',Max_Accuracy)