#!/usr/bin/env python
import optparse
from os.path import splitext

parser = optparse.OptionParser()

parser.add_option('-a', '--add-size',
    action="append", dest="sizes",
    help="Define model sizes", default=[])
parser.add_option('-z', '--single-size',
    action="store", dest="single_size",
    help="Define single size for all layers of model. Must be used with '--repeat-for' flag.", default="")
parser.add_option('-r', '--repeat-for',
    action="store", dest="repeat_for",
    help="Define number of repeats of single size. Must be used with '--single-size' flag.", default="")
parser.add_option('-i', '--input-size',
    action="store", dest="input_size",
    help="Define input size. Default is the first element of '--add-size'.", default="")
parser.add_option('-b', '--num-batches',
    action="store", dest="num_batches",
    help="Set number of batches", default=1)
parser.add_option('-d', '--deepspeech',
    action="store_true", dest="deepspeech",
    help="Activate the deepspeech mode", default=False)
parser.add_option('--resnet50',
    action="store_true", dest="resnet50",
    help="Activate the resnet50 mode", default=False)
parser.add_option('-o', '--output',
    action="store", dest="output_path",
    help="Set the path to store the output '.tflite' file.", default="model.tflite")
parser.add_option('-A', '--quantize-activations',
    action="store_true", dest="quantize_activations",
    help="Quantizes the activations too", default=False)
parser.add_option('-n', '--no-optimization',
    action="store_true", dest="no_optimization",
    help="Disables All Optimizations", default=False)

options, args = parser.parse_args()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Reshape

model_sizes = []
input_size = 10

if not options.deepspeech and not options.resnet50:
    if options.single_size and options.repeat_for:
        for i in range(int(options.repeat_for)):
            model_sizes.append(int(options.single_size))
        if options.input_size:
            input_size = int(options.input_size)
        else:
            input_size = int(options.single_size)
    else:
        if len(options.sizes) < 2 and not options.input_size:
            print(f"You must pass at least two sizes, one size and input size, or use single size with repeat.")
            exit(-1)
        if options.input_size:
            input_size = int(options.input_size)
        else:
            input_size = int(options.sizes[0])
        for i in options.sizes[0:] if options.input_size else options.sizes[1:]:
            model_sizes.append(int(i))
    

    model = Sequential()
    model.add(Input(shape=(input_size), batch_size=int(options.num_batches)))
    for size in model_sizes:
        model.add(Dense(size, activation='relu'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if not options.no_optimization:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if options.quantize_activations:
            def representative_data_gen():
                import numpy as np
                num_samples = 10
                data = np.random.rand(num_samples, int(options.num_batches), input_size).astype(np.float32)
                for i in range(num_samples):
                    yield [
                        data[i,:,:].reshape(int(options.num_batches), input_size)
                    ]
            converter.representative_dataset = representative_data_gen
            converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
    else:
        converter.optimizations = []
    tflite_model = converter.convert()
    with open(options.output_path, "wb") as f:
        f.write(tflite_model)
elif options.resnet50:
    model = tf.keras.applications.resnet.ResNet50(
        weights='imagenet',
    )
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if not options.no_optimization:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if options.quantize_activations:
            def representative_data_gen():
                import numpy as np
                num_samples = 10
                data = np.random.rand(num_samples, 224, 224, 3).astype(np.float32)
                for i in range(num_samples):
                    yield [
                        data[i,:,:,:].reshape(1, 224, 224, 3)
                    ]
            converter.representative_dataset = representative_data_gen
            converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
    else:
        converter.optimizations = []
    tflite_model = converter.convert()
    extension = ""
    if options.quantize_activations:
        extension = "-i8i8"
    elif options.no_optimization:
        extension = "-f32f32"
    with open(f"{splitext(options.output_path)[0]}{extension}{splitext(options.output_path)[1]}", "wb") as f:
        f.write(tflite_model)
else:
    model = Sequential()
    model.add(Input(shape=(494), batch_size=int(options.num_batches)))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Reshape((-1, 2048)))
    model.add(LSTM(2048, activation='relu', batch_size=1))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(28, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(options.output_path, "wb") as f:
        f.write(tflite_model)


