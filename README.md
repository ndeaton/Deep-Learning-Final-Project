# Deep-Learning-Final-Project
Supplementary code for the CS 7643 final project.

## citation: The DenseTorch Code is a modified version of the code found at the following repository:
https://github.com/DrSleep/light-weight-refinenet}

## Download Files

Download the folders in this GitHub: https://github.com/ndeaton/Deep-Learning-Final-Project.git

If you will be using Google Collab, create zip files of newEndoViz and DenseTorch named newEndoViz.zip and DenseTorch.zip

## Dataset

The newEndoVis dataset is a rearranged version of the MICCAI'15 EndoVis dataset which can be downloaded here:

http://opencas.webarchiv.kit.edu/?q=node/30

The EndoVis dataset has been rearranged to include a validation folder

## Training a model

Open the DenseTorch.ipynb file. This can be run in Google Collab using the runtime type GPU. 

The first command is to upload the zip files to Google Collab which can be skipped if running the notebook as a jupyter notebook.
The second command will unzip newEndoViz.zip and DenseTorch.zip
The third command will move all files and folders from the DenseTorch folder created when DenseTorch.zip is unzipped to the home directory. Alternatively, the newEndoViz folder can be moved into the DenseTorch folder, and then the rest of the commands can be run from the DenseTorch directory
Install requirements with: 
```
pip install -r requirements.txt
$pip install -e .
```
Sometimes the requirements were not fully installed. This can be fixed by running: $pip install -e .
The model can be trained by using the following command (a pretrained model was not included due to file size upload restrictions.
```
python train.py
```
After training finishes, the best model will be stored in the corresponding `pth.tar` file.
The model can be tested by using the following command
```
python test.py
```

## Altering hyperparameters
hyperparameters for training the model can be altered by changing the values in the following file

config.py

hyperparameters for testing the model can be altered by changing the values in the following file

config_test.py

The format of these files is the same. Within these files the definition for the parameters are as follows:
batch_size: batch size for training
val_batch_size: batch size for validation
n_epochs: number of training epochs
val_every: number of epochs run between validation

lr_enc: encoder learning rate
optim_enc: encoder optimization method
mom_enc: encoder momentum
wd_enc: weight decay encoder
lr_dec: decoder learning rate
optim_dec: decoder optimization method
mom_dec: decoder momentum
wd_dec: weight decay decoder

## Results

| Result | Description |
| --- | --- |
| saved classified image | Image classified by the model. Saved as out{}r{}.png where the first {} represents the number of the image in the batch and the second {} is a random identification number  
| saved ground truth segmentation | Ground truth segmentation for the image classified by the model. Saved as target{}r{}.png where the first {} represents the number of the image in the batch and the second {} is the same a random identification number as the classified image
| mIoU | mean Intersection over Union. Printed automatically to terminal or default printing location of the system running test.py 
