# YOLACT_Instance_segmentation
In this project we have implemented a fully-convolutional model for realtime  Instance Segmentation through YOLACT < Real Time Instance Segmentation > .
For backbone , Resnet50 is used alongwith Feature Pyramid Network (FPN )  and Protonet . Accomplished by breaking Instance segmentation into two parallel subtasks: (1) generating a set of prototype masks and (2) predicting per-instance mask coefficients. Then we produce instance masks by linearly combining the prototypes with the mask coefficients.

The model was trained on PASCAL VOC 2012 data which contained 2913 'train + val' images .  
The model was trained on Google Colab GPU for 10 hours and achieved fairly good results. Some results can be found in Result Folder .

Drive Link for Dataset :- 
https://drive.google.com/drive/folders/1ur6EsfH_5ZUQxZgT5Qn3ULPfnQqJc_pj?usp=sharing

Drive Link for trained weights :- 
https://drive.google.com/drive/folders/12hglkT6aRq7KG48zgTY0tXmr6XsKegvk?usp=sharing

NETWORK STRUCTURE :-
![pic](https://user-images.githubusercontent.com/61914611/108886416-f2dbdb00-762e-11eb-93e6-0fea1adc6ace.png)



