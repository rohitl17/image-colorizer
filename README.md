An autoencoder for colorizing natural landscape grayscale images to colored images. 


The repository has 5 different python files:
- train.py : Training script
- inference.py : Colorizer prediction
- model.py : Variational autoencoder model
- colorize.py : Prediction modules and batch generator
- color_temperature_adjustment.py : Cooling and warming filters for adjusting color temperature of the image


Training the model:
> Pass model save path and image directory using arguments
> From command line run the code, "python train.py images_path model_save_path"

Inference:
> Pass model save path, input image path and final image save path using arguments
> From command line run the code, "python inference.py image_path saved_model_path"
