"""
 *******************************************************************************
 * ACETONE: Predictable programming framework for ML applications in safety-critical systems
 * Copyright (c) 2024. ONERA
 * Copyright (c) 2024. AIRBUS
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx.checker
import onnx
from collections import OrderedDict
from onnxscript.optimizer import remove_unused
import sys
class MyModel(nn.Module):

    def conv(self,id,ker_size,in_chan,out_chan):
        return nn.Sequential(OrderedDict([
          ('conv'+id, nn.Conv2d(in_chan,out_chan,ker_size)),
          ('relu'+id, nn.ReLU()),
          ('maxpool'+id,nn.MaxPool2d(2))]))

    def __init__(self):
        super(MyModel, self).__init__()
        self.c1 = self.conv('1',5,1,16)
        self.c2 = self.conv('2',9,16,48)
        self.c3 = self.conv('3',9,48,48)
        self.c4 = self.conv('4',5,48,32)
        self.l = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(512,64)),
            ('relu',nn.ReLU()),
            ('linear2',nn.Linear(64,1)),
            ('sigmoid',nn.Sigmoid()),
             ]))

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        ''' do not use flatten which prevents optims. the shape explicit output is required to help ORT optimizer. '''
        x = torch.reshape(x,(-1,512))
        x = self.l(x)
        return x

torch_model = MyModel()
torch_input = torch.randn(1,1,150,150)

onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
''' dynamo export generates lots of burden. the remove_unused scripts cleans up '''
remove_unused.remove_unused_nodes(onnx_program.model_proto)
onnx.checker.check_model(onnx_program.model_proto)
onnx_program.save(sys.argv[1])

onnx_program=torch.onnx.export(torch_model,torch_input,sys.argv[2],do_constant_folding=True,input_names = ['input'],output_names = ['output'])
onnx_program = onnx.load(sys.argv[2])
onnx.checker.check_model(onnx_program)
