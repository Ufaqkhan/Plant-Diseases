[fname path]=uigetfile('.jpg','Give the testing file as input');
fname=strcat(path,fname);
im=imread(fname);
imshow(im);
[feat_disease]=featureselection(im);
%diseasename=inputdlg('Enter the disease name');
%diseasename=cell2mat(diseasename);
try
load leaf_dataset.mat;
F=[feat_disease];
db=[db;F];
save leaf_dataset.mat db
catch
F=[feat_disease];
db=F;
save leaf_dataset.mat db
end
