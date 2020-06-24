# Hand-Gesture-Recognition-System-for-Sign-Language
## Architecture
This System implements a CNN created using TensorFlow and uses OpenCV for utilities and Computer Vision and Image Processing tasks.

## Inputs:
Variable size of images to be transformed into a uniform size through preprocessing utility-"cvt_dataset_into_bin" by changing sizes.
64 is preferable for quicker training.

## How to use:
* Clone the repository
* Make a dataset by capturing images of each sign and placing in Dataset/{letter of the sign}/{image_name.jpg}
* run train.py
* run GUI.py

## Future work:
* Ensemble of models for better prediction
* Using a RCNN or better models to look at "flowing" gesture indicating words or sentences in certain languages
