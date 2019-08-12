#!/usr/bin/env python

"""
    convnet/main.py
"""
#Convolutional Neural Network on MNIST handwritten digit dataset
import sys
import json
import argparse
import keras
from time import time
from keras.layers.normalization import BatchNormalization
from keras.layers import Concatenate, Conv2D, Dense, GlobalMaxPooling2D
from keras.layers import Input, LeakyReLU, MaxPooling2D, ReLU, Softmax
from keras.losses import sparse_categorical_crossentropy
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model



# --
# User code
# Note: Depending on how you implement your model, you'll likely have to change the parameters of these
# functions.  They way their shown is just one possble way that the code could be structured.

def create_residual(x0, filters):
    x1 = Conv2D(filters, (3, 3), padding='same', use_bias=False)(x0)
    x1 = BatchNormalization(epsilon=1e-5)(x1)
    x1 = ReLU()(x1)
    x2 = Conv2D(filters, (1, 1), padding='same', use_bias=False)(x0)
    x = Concatenate()([x1, x2])
    return x

def make_model(input_channels, output_classes, residual_block_sizes, scale_alpha, optimizer, lr, momentum):
    print(input_channels, output_classes, residual_block_sizes, scale_alpha, optimizer, lr, momentum)
    """
   inputs = Input((input_channels, 32, 32))
   x = Conv2D(residual_block_sizes[0][0], (3, 3), input_shape=(input_channels, 32, 32), padding='same')(inputs)
   """
    image_shape = (32, 32, input_channels)
    #image_shape = (input_channels, 32, 32)
    inputs = Input(image_shape)
    x = Conv2D(residual_block_sizes[0][0], (3, 3), input_shape=image_shape, padding='same', use_bias=False)(inputs)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = ReLU()(x)
    x = create_residual(x, residual_block_sizes[0][0]) # first residual block
    x = MaxPooling2D((2, 2))(x)
    x = create_residual(x, residual_block_sizes[1][0]) # second residual block
    x = MaxPooling2D((2, 2))(x)
    x = create_residual(x, residual_block_sizes[2][0]) # third residual block
    x = GlobalMaxPooling2D()(x)
 
    x = Dense(2, activation='linear', use_bias=False)(x)
    #x = LeakyReLU(scale_alpha)(x) # is that proper layer for scale alpha?
    print(type(x))
    #x *= scale_alpha
    #x = Dense(2, activation='softmax', use_bias=False)(x)
    x = Softmax()(x)
    model = Model(inputs, x)
 
    if optimizer == "SGD":
        sgd = SGD(lr=lr, momentum=momentum)
        model.compile(optimizer=sgd, loss=sparse_categorical_crossentropy,
            metrics=['sparse_categorical_accuracy'])
    else:
        model.compile(optimizer, loss=sparse_categorical_crossentropy,
                metrics=['sparse_categorical_accuracy'])
    return model


def make_train_dataloader(X, y, batch_size, shuffle):
    # ... your code here ...
    datagen = ImageDataGenerator()
    dataloader = datagen.flow(X, y, batch_size, shuffle=shuffle)
    return dataloader


def make_test_dataloader(X, batch_size, shuffle):
    # ... your code here ...
    datagen = ImageDataGenerator()
    dataloader = datagen.flow(X, None, batch_size, shuffle=shuffle)
    return dataloader


def train_one_epoch(model, dataloader):
    # ... your code here ...
    #history = model.fit_generator(dataloader, epochs=1)
    model.fit_generator(dataloader, epochs=1,steps_per_epoch=steps_per_epoch)
    return model


def predict(model, dataloader):
    # ... your code here ...
    probs = model.predict_generator(dataloader,steps=steps)
    predictions = []

    for i in range(probs.shape[0]):
        max_prob = -1
        best_class = -1
    for j in range(probs.shape[1]):
        if probs[i][j] > max_prob:
            max_prob = probs[i][j]
            best_class = j
    predictions.append(best_class)
    predictions = np.array(predictions)
    return predictions


# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=128)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # --
    # IO
    
    # X_train: tensor of shape (number of train observations, number of image channels, image height, image width)
    # X_test:  tensor of shape (number of train observations, number of image channels, image height, image width)
    # y_train: vector of [0, 1] class labels for each train image
    # y_test:  vector of [0, 1] class labels for each test image (don't look at these to make predictions!)
    
    X_train = np.load('data/cifar2/X_train.npy')
    X_test  = np.load('data/cifar2/X_test.npy')
    y_train = np.load('data/cifar2/y_train.npy')
    y_test  = np.load('data/cifar2/y_test.npy')
    
    # --
    # Define model
    
    model = make_model(
        input_channels=3,
        output_classes=2,
        residual_block_sizes=[
            (16, 32),
            (32, 64),
            (64, 128),
        ],
        scale_alpha=0.125,
        optimizer="SGD",
        lr=args.lr,
        momentum=args.momentum,
    )
    
    # --
    # Train
    
    t = time()
    for epoch in range(args.num_epochs):
        
        # Train
        model = train_one_epoch(
            model=model,
            dataloader=make_train_dataloader(X_train, y_train, batch_size=args.batch_size, shuffle=True)
        )
        
        # Evaluate
        preds = predict(
            model=model,
            dataloader=make_test_dataloader(X_test, batch_size=args.batch_size, shuffle=False)
        )
        
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == X_test.shape[0]
        
        test_acc = (preds == y_test.squeeze()).mean()
        
        print(json.dumps({
            "epoch"    : int(epoch),
            "test_acc" : test_acc,
            "time"     : time() - t
        }))
        sys.stdout.flush()
        
    elapsed = time() - t
    print('elapsed', elapsed, file=sys.stderr)
    
    # --
    # Save results
    
    os.makedirs('results', exist_ok=True)
    
    np.savetxt('results/preds', preds, fmt='%d')
    open('results/elapsed', 'w').write(str(elapsed))
