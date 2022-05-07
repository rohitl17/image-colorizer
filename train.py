from model import model
from colorize import dataloader

import sys

def train_model(batch_size, train_files, validation_files, model_save_path):
    file_path=model_save_path

    # reduce learning rate by sqrt(0.1) if the loss does not improve in 5 epochs
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   verbose=1,
                                   min_lr=0.5e-6)

    # Save weights
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True, save_weights_only=True)

    
    autoencoder=model()
    # Mean Square Error (MSE) loss function, Adam optimizer
    autoencoder.compile(loss='mse', optimizer='adam')

    # called every epoch
    callbacks = [lr_reducer, checkpoint]
    
    autoencoder.fit_generator(dataloader(train_files, batch_size), validation_data=dataloader(validation_files, batch_size),
                    steps_per_epoch=len(train_files)//batch_size, validation_steps=len(validation_files)/batch_size, epochs=20, verbose=1, callbacks=callbacks)
    
    
def prepare_data_and_start_training(images_path, model_save_path):
    
    all_files=glob.glob(images_path+'*.jpg')

    random.shuffle(all_files)

    train_files=all_files[:int(len(all_files)*0.8)]
    validation_files=all_files[int(len(all_files)*0.8):]
    
    train_model(32, train_files, validation_files)
    

images_path=sys.argv[0]
model_save_path=sys.argv[1]
prepare_data_and_start_training(images_path, model_save_path)

 
