#!/bin/bash


pip install opencv-python gdown matplotlib ftfy regex scikit-image selective_search 

cd training-free-object-counter/pretrain
wget -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
cd ../dataset





gdown 1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S

for f in *.zip; do unzip -d "${f%.zip}" "$f"; done
cd FSC147_384_V2
wget -O https://raw.githubusercontent.com/ActiveVisionLab/LearningToCountAnything/master/data/FSC-147/annotation_FSC147_384.json
wget -O https://raw.githubusercontent.com/ActiveVisionLab/LearningToCountAnything/master/data/FSC-147/Train_Test_Val_FSC_147.json
cd ..

mv FSC147_384_V2/images_384_VarV2 FSC147_384_V2/images
cd ..

python main-fsc147.py --test-split='test' --prompt-type='box' --device='cuda:0'


