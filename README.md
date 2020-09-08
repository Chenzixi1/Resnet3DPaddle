# Resnet3DPaddle
Resnet3D by PaddlePaddle  
https://www.paddlepaddle.org.cn/

## Reference
Learning Spatio-Temporal Features with 3D Residual Networks for Action Recognition (2017ICCV)  
https://github.com/kenshohara/3D-ResNets-PyTorch

## DataSet
UCF-101 Action Recognition DataSet  
http://crcv.ucf.edu/data/UCF101.php

## Pretrain
Pretrain pytorch model in [here](https://drive.google.com/drive/folders/1xbYbZ7rpyjftI_KCk6YuL-XrfQDz7Yd4) which is    
`<r3d50_KM_200ep.pth --model resnet --model_depth 50 --n_pretrain_classes 1039>`  
transform pretrain model from torch version to paddlepaddle version
`python model2model.py`

## Pretreatment
create label list  
`python savelabel.py`  
transform from jpg to pkl  
`python jpg2pkl.py`  
package to list  
`python data_list_gener.py`  

## Training
`python train.py --resume True --use_gpu True --epoch 1 `  

## Val
`python eval.py --weights 'checkpoints_models/res503d' --use_gpu True`

## Infer
`python infer.py --weights 'checkpoints_models/res503d' --use_gpu True`
