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
from .tfmspec import Sdoc

class ConvPool(keras.layers.Layer):
    '''
    Convolutional and Maxpool layer group
    '''
    def __init__(self,prefix,channel,kernel_size,id):
        super().__init__(name=(f'{prefix}CONVPOOL{id}'))
        self.conv = keras.layers.Conv2D(channel, kernel_size, activation='relu',data_format='channels_first',name=(prefix+'CONV'+str(id)))
        self.maxpool = keras.layers.MaxPooling2D(pool_size=(2, 2),data_format='channels_first',name=(prefix+'MAXPOOL'+str(id)))
    def call(self,x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x

def define_cnn(prefix="TFM_KS_"):
    layers = [
        ConvPool(prefix,16,5,1),
        ConvPool(prefix,48,9,2),
        ConvPool(prefix,48,9,3),
        ConvPool(prefix,32,5,4),
        keras.layers.Reshape((512,),name=f'{prefix}FLATTEN'),  
        keras.layers.Dense(64,activation="relu",bias_initializer='random_normal',name=f"{prefix}DENSE1"),
        keras.layers.Dense(1,activation="sigmoid",bias_initializer='random_normal',name=f"{prefix}TFM_KS_DENSE2")
    ]
    full_model = keras.Sequential(layers,name=f"{prefix}SEQUENTIAL")
    full_model.output_names=['y']
    full_model.compile()
    return full_model

def export_onnx(keras_model,file_path):
    '''
    Exports ONNX model from keras
    keras layer.req is set as doc_string in ONNX corresponding node.
    '''
    input_signature = [tf.TensorSpec([ 1, 3,150,150],  tf.float32, name='x')]
    onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_signature, opset=18)
#print(full_model.summary())
#print (onnx_model)
    for n in onnx_model.graph.node:
        for l in keras_model.layers:
            if l.name in n.name:
                n.doc_string = l.req
                break

    onnx.save_model(onnx_model, file_path)
    return onnx_model

def export_specification(keras_model,file_path):
    '''
    Exports keras model strictdoc specification
    Augment layers with req attribute containing requirement uid
    '''
    doc = Sdoc('TFM Keras requirements','1','PUBLIC')
    for i,l in enumerate(keras_model.layers):
        l.req='TFM-KS-REQ-'+str(i)
        statement = f'''{l.__class__.__name__}(activation={l.activation.__name__})''' if hasattr(l,'activation') else l.__class__.__name__
        doc.new_req(l.req,l.name,statement)
    doc.write(file_path) 
#    print (l.name, l.__class__.__name__,l.activation)

#print(doc)

def main():
    ''' 
    Defines a keras model and output ONNX MLMD and specification
    argv1: onnx file name 
    argv2: specification path
    '''
    keras_model = define_cnn()
    export_specification(keras_model,sys.argv[2]+'/TFM_KERAS_REQ.sdoc')
    onnx_model = export_onnx(keras_model,sys.argv[1])

if __name__ == '__main__':
    main()