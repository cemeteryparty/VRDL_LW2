"""
Derived from https://github.com/fizyr/keras-retinanet/blob/main/keras_retinanet/bin/train.py
"""
import os
import sys
import argparse
from scipy.io import savemat

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from keras_retinanet import layers
from keras_retinanet import losses
from keras_retinanet import models
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.callbacks.eval import Evaluate
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.anchors import make_shapes_callback
from keras_retinanet.utils.config import read_config_file
from keras_retinanet.utils.config import parse_anchor_parameters
from keras_retinanet.utils.config import parse_pyramid_levels
from keras_retinanet.utils.image import random_visual_effect_generator
from keras_retinanet.utils.model import freeze as freeze_model
from keras_retinanet.utils.tf_version import check_tf_version
from keras_retinanet.utils.transform import random_transform_generator


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.
    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, num_classes, weights, args):
    """ Creates three models (model, training_model, prediction_model).
    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        args: parseargs args object.
    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """
    modifier = freeze_model if args.freeze_backbone else None
    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors   = None
    pyramid_levels = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)
        num_anchors   = anchor_params.num_anchors()
    if args.config and 'pyramid_levels' in args.config:
        pyramid_levels = parse_pyramid_levels(args.config)

    model = model_with_weights(
        backbone_retinanet(
            num_classes, 
            num_anchors=num_anchors, 
            modifier=modifier, 
            pyramid_levels=pyramid_levels
        ),
        weights=weights,
        skip_mismatch=True
    )
    training_model = model

    """ make prediction model """
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params, pyramid_levels=pyramid_levels)

    """ compile model """
    opt = keras.optimizers.Adam(
        learning_rate=ExponentialDecay(args.lr, decay_steps=args.steps, decay_rate=0.98),
        clipnorm=args.optimizer_clipnorm
    )
    # opt = keras.optimizers.Adam(lr=args.lr, clipnorm=args.optimizer_clipnorm)
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=opt
    )
    return model, training_model, prediction_model

def create_callbacks(model, training_model, prediction_model, validation_generator, args):
    """ Creates the callbacks to use during training.
    Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.
    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tfboard_callback = None
    if args.evaluation and validation_generator:
        evaluation = Evaluate(validation_generator, tensorboard=tfboard_callback, weighted_average=args.weighted_average)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=args.backbone, dataset_type=args.dataset_type)
            ),
            verbose=1,
            # save_best_only=True,
            # monitor="mAP",
            # mode='max'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    return callbacks


def create_generators(args, preprocess_image):
    """ Create generators for training and validation.
    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size'       : args.batch_size,
        'config'           : args.config,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'no_resize'        : args.no_resize,
        'preprocess_image' : preprocess_image,
        'group_method'     : args.group_method
    }

    # create random transform generator for augmenting training data
    if args.random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
        )
        visual_effect_generator = random_visual_effect_generator(
            contrast_range=(0.9, 1.1),
            brightness_range=(-.1, .1),
            hue_range=(-0.05, 0.05),
            saturation_range=(0.95, 1.05)
        )
    else:
        transform_generator = random_transform_generator()
        visual_effect_generator = None

    
    train_generator = CSVGenerator(
        args.annotations,
        args.classes,
        transform_generator=transform_generator,
        visual_effect_generator=visual_effect_generator,
        **common_args
    )

    if args.val_annotations:
        validation_generator = CSVGenerator(
            args.val_annotations,
            args.classes,
            shuffle_groups=False,
            **common_args
        )
    else:
        validation_generator = None

    return train_generator, validation_generator


def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.
    Args
        parsed_args: parser.parse_args()
    Returns
        parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError("Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size, parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError("Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu, parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    return parsed_args


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)
    parser.add_argument('--backbone',         help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu',        help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force',  help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--initial-epoch',    help='Epoch from which to begin the train, useful if resuming from snapshot.', type=int, default=0)
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--lr',               help='Learning rate.', type=float, default=1e-5)
    parser.add_argument('--optimizer-clipnorm', help='Clipnorm parameter for  optimizer.', type=float, default=0.001)
    parser.add_argument('--snapshot-path',    help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir',  help='Log directory for Tensorboard output', default='')  # default='./logs') => https://github.com/tensorflow/tensorflow/pull/34870
    parser.add_argument('--tensorboard-freq', help='Update frequency for Tensorboard output. Values \'epoch\', \'batch\' or int', default='epoch')
    parser.add_argument('--no-snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',    help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--no-resize',        help='Don''t rescale the image.', action='store_true')
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file.')
    parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')
    parser.add_argument('--compute-val-loss', help='Compute validation loss during training', dest='compute_val_loss', action='store_true')
    parser.add_argument('--reduce-lr-patience', help='Reduce learning rate after validation loss decreases over reduce_lr_patience epochs', type=int, default=2)
    parser.add_argument('--reduce-lr-factor', help='When learning rate is reduced due to reduce_lr_patience, multiply by reduce_lr_factor', type=float, default=0.1)
    parser.add_argument('--group-method',     help='Determines how images are grouped together', type=str, default='ratio', choices=['none', 'random', 'ratio'])

    # Fit generator arguments
    parser.add_argument('--multiprocessing',  help='Use multiprocessing in fit_generator.', action='store_true')
    parser.add_argument('--workers',          help='Number of generator workers.', type=int, default=1)
    parser.add_argument('--max-queue-size',   help='Queue length for multiprocessing workers in fit_generator.', type=int, default=10)

    return check_args(parser.parse_args(args))


def main(args=None):
    """ parse arguments """
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    print(args)
    #raise Exception("Debug Checkpoint")

    """ create object that stores backbone information """
    backbone = models.backbone("resnet50")

    check_tf_version()

    """ create the generators """
    train_generator, validation_generator = create_generators(
        args, backbone.preprocess_image
    )

    """ create the model """
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model            = models.load_model(args.snapshot, backbone_name=args.backbone)
        training_model   = model
        anchor_params    = None
        pyramid_levels   = None
        if args.config and 'anchor_parameters' in args.config:
            anchor_params = parse_anchor_parameters(args.config)
        if args.config and 'pyramid_levels' in args.config:
            pyramid_levels = parse_pyramid_levels(args.config)

        prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params, pyramid_levels=pyramid_levels)
    else:
        """ Load Weight """
        weights = args.weights
        if weights is None and args.imagenet_weights:
            weights = backbone.download_imagenet()
        """ Create Model """
        model, training_model, prediction_model = create_models(
            backbone_retinanet = backbone.retinanet,
            num_classes        = train_generator.num_classes(),
            weights            = weights,
            args               = args
        )

    """ print model summary """
    # print(model.summary())

    """ create the callbacks """
    callbacks = create_callbacks(
        model, training_model, prediction_model,
        validation_generator, args
    )

    if not args.compute_val_loss:
        validation_generator = None

    """ start training """
    history = training_model.fit(
        train_generator,
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=args.steps,
        initial_epoch=args.initial_epoch,
    )
    savemat("output/train_log.mat", history.history)

if __name__ == '__main__':
    main()