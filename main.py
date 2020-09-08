import os

os.system('tar zxvf /home/aistudio/data/data49167/UCF-101-jpg.tgz -C data/')
os.system('python savelabel.py')
os.system('python jpg2pkl.py')
os.system('python data_list_gener.py')
os.system('python train.py --use_gpu True --epoch 100')

os.system('''python eval.py --weights 'checkpoints_models/tsn_model' --use_gpu True''')