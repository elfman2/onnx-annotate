# The Training Framework Models
The following scripts specify the CNN model. 
For the purpose of this experiment, we are working with untrained model parameters
(we assume that the models are trained by some other script).

## [tfm-keras.py](tfm-keras.py) 
is the keras script to design the CNN, and export it to ONNX

## [tfm-torch.py](tfm-torch.py) 
is the torch script to design the CNN, and export it to ONNX using 2 methods (legacy export and dynamo export)
