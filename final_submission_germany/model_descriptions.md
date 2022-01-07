GERMANY
------------------------------------

## Preprocessing

* Used Planet data only
* Rasterize crop fields and randomly extract 640 pixels
* Augmentation: Create 10 samples of dimension (TIME, BANDS, 64) out of each crop field
* --> 21520 samples for training / validation
* Include NDVI as 5th band

## Model

* PseLTae: Pixel Set Encoder with Lightweight Temporal Attention
    * Adapted from https://github.com/VSainteuf/pytorch-psetae
    * Spatial Encoder: Random pixels instead of convolutional layer
    * Temporal Encoder: Attention mechanism
    * Decoder: Predict class logits
* Reduced model capacity by 90% with respect to default configuration
* Hyperparameters: See Notebook

## Training

* Focal loss
* 5-fold cross validation
* Adam optimizer, lr=1e-3, weight decay = 1e-6
* Early stopping, monitors validation accuracy
* Explored combination with Sentinel-1 and Sentinel-2 data, but yielded worse accuracy

## Inference

* Save average predicted crop probability across k-folds


SOUTH AFRICA
------------------------------------


## Preprocessing

* Used Planet and Sentinel-1 data
* Rasterize crop fields and randomly extract 640 pixels
* Augmentation: Create 10 samples of dimension (TIME, BANDS, 64) out of each crop field
* --> 41430 samples for training / validation
* Include NDVI as 5th band (Planet), RVI as 3rd band (Sentinel-1)

## Model

* Combined PseLTae: Pixel Set Encoder with Lightweight Temporal Attention
    * Adapted from https://github.com/VSainteuf/pytorch-psetae
    * Spatial Encoder: Random pixels instead of convolutional layer
    * Temporal Encoder: Attention mechanism
    * Decoder: Predict class logits
* One PseLTae for each dataset
* Combine in Decoder module
* Hyperparameters: See Notebook

## Training

* Focal loss
* 10-fold cross validation
* Adam optimizer, lr=1e-3, weight decay = 1e-6
* Early stopping, monitors validation accuracy
* Explored combination with Sentinel-2 data, but yielded worse accuracy


## Inference

* Save average predicted crop probability across k-folds