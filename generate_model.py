import os
import numpy as np
from models.cnn_model import cnn_model
from models.EffNet import EffNet
from models.ResNet import ResNet
from models.Inception import Inception
from models.VGGNet import VGGNet
from models.TLNet import TLNet
from opts import parse_opts



def main(cnn_model, input_shape, activation, freeze_layer):
    
    if cnn_model == 'cnn':
        my_model = cnn_model(
            input_shape=input_shape,
            activation=activation
            )
    elif cnn_model == 'ResNet101V2':
        my_model = ResNet(
            resnet='ResNet101V2',  #'ResNet50V2',
            transfer=transfer,
            freeze_layer=freeze_layer,
            input_shape=input_shape,
            activation=activation,
            )
    elif cnn_model == 'EffNetB4':
        my_model = EffNet(
            effnet='EffNetB4',
            transfer=transfer,
            freeze_layer=freeze_layer,
            input_shape=input_shape,
            activation=activation
            )
    elif cnn_model == 'TLNet':
        my_model = TLNet(
            resnet='ResNet101V2',
            input_shape=input_shape,
            activation=activation
            )
    elif cnn_model == 'Xception':
        my_model = Inception(
            #inception='InceptionV3', 
            inception='Xception',
            transfer=transfer,
            freeze_layer=freeze_layer,
            input_shape=input_shape,
            activation=activation
            )
    elif cnn_model == 'VGG16':
        my_model = VGGNet(
            VGG='VGG16',
            transfer=transfer,
            freeze_layer=freeze_layer,
            input_shape=input_shape,
            activation=activation
            )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--manual_seed', default=1234, type=int, help='Mannual seed')
    parser.add_argument('--cnn_model', default='ResNet', type=str, help='cnn model names')
    parser.add_argument('--model_depth', default=101, type=str, help='model depth (18|34|50|101|152|200)')
    parser.add_argument('--input_shape', default=2, type=str, help='model output classes')
    parser.add_argument('--in_channels', default=1, type=str, help='model input channels (1|3)')
    parser.add_argument('--sample_size', default=128, type=str, help='image size')
    args = parser.parse_args()

    model = main(cnn_name=args.cnn_name,
                 model_depth=args.model_depth,
                 n_classes=args.n_classes,
                 in_channels=args.in_channels,
                 sample_size=args.sample_sizes
                )
