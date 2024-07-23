"""
 *******************************************************************************
 * ACETONE: Predictable programming framework for ML applications in safety-critical systems
 * Copyright (c) 2024. ONERA
 * This file is part of ACETONE
 *
 * ACETONE is free software ;
 * you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation ;
 * either version 3 of  the License, or (at your option) any later version.
 *
 * ACETONE is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with this program ;
 * if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA
 ******************************************************************************
"""
import tf2onnx
import keras
import onnx
import tensorflow as tf
import sys
layers = [
    keras.layers.Conv2D(16, 5, activation='relu',data_format='channels_first'),
    keras.layers.MaxPooling2D(pool_size=(2, 2),data_format='channels_first'),  
    keras.layers.Conv2D(48, 9, activation='relu',data_format='channels_first'),
    keras.layers.MaxPooling2D(pool_size=(2, 2),data_format='channels_first'),    
    keras.layers.Conv2D(48, 9, activation='relu',data_format='channels_first'),
    keras.layers.MaxPooling2D(pool_size=(2, 2),data_format='channels_first'),    
    keras.layers.Conv2D(32, 5, activation='relu',data_format='channels_first'),
    keras.layers.MaxPooling2D(pool_size=(2, 2),data_format='channels_first'),  
    keras.layers.Reshape((512,)),  
    keras.layers.Dense(64,activation="relu",bias_initializer='random_normal'),
    keras.layers.Dense(1,activation="sigmoid",bias_initializer='random_normal')
]
full_model = keras.Sequential(layers)
full_model.output_names=['y']
full_model.compile()
input_signature = [tf.TensorSpec([ 1, 3,150,150],  tf.float32, name='x')]
onnx_model, _ = tf2onnx.convert.from_keras(full_model, input_signature, opset=18)
print(full_model.summary())
#print (onnx_model)
onnx.save_model(onnx_model, sys.argv[1])