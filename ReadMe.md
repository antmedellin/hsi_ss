# Overview 

This package is used to train models for the WHISPERS 2024 MMSeg-YREB: Multi-Modal Remote Sensing Semantic Segmentation Challenge. 




## Installation 

Build the docker image:
```
docker build -t hsi_ss .  
```

Modify the devcontainer.json file to ensure the image data is mounted into the container, according to your file structure.
  
You can then open the code in a Dev Container in VS Code. 


## Usage

As with training machine learning models, there are many parameters that can be tuned to meet your hardware setup and your improve training performance. Some of the parameters that can be tuned include: 
loss function, batch size, ignore index, number of workers, initial learning rate, stochastic weight averaging learning rate, image height, image width, max number of epochs, accumulated gradient batches, gradient clipping value, normalization parameters, and data augmentations

After the training setup is tuned to match your desired setup, run the `train.py` file. 

You can monitor training performance with Tensorboard. 