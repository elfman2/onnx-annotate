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

'''
 This script is used:
    - to check the model consistency
    - to display model attributes
    - to modify doc_string in onnx model,
    - to perform shape inference
    - to display operator dependancies and versions
    - to convert initializers raw_data to concrete data type
    - to generate an opset onnx file corresponding to OpsetIdProto
    - simplify, optimize the model
    - move intiializers to external data file
    - convert to onnxscript (debug model with pdb)
'''
import onnx
import argparse
import numpy as np
from onnx import (
    version_converter,
    defs,
    IR_VERSION,
    OperatorSetProto,
    OperatorSetIdProto,
    OperatorProto,
    FunctionProto,
    OperatorStatus,
    STABLE
)
from TFM.tfmspec import Sdoc

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='print/modify onnx model strings')
    parser.add_argument('--input', required=True,help='intput onnx model')
    parser.add_argument('--set_model_domain', help='sets the model domain')
    parser.add_argument('--metadata', nargs=1,help='modifies model meta data from json file argument {<model name>:<value>,<model name>:<value>...}')
    parser.add_argument('--doc_string', nargs=1,help='modifies doc_string from json file argument {<model name>:<value>,<model name>:<value>...}')
    parser.add_argument('--shapeinf', action='store_true',help='perform shape inference')
    parser.add_argument('--depend', action='store_true',help='display oeprator dependencies and versions corresponding to opset')
    parser.add_argument('--opset_change',type=int,help='change the opset version')
    parser.add_argument('--silu',action='store_true',help='factorization of Silu activation function (Silu = x * Sigmoid)')
    parser.add_argument('--raw_data_convert', action='store_true',help='convert raw_data to concrete data type')
    parser.add_argument('--gen_opset_model',help='for each input model opset domain, generates an opset protobuf file corresponding to the input model opset id. Ouptut file name is <opset domain (ex:ai.onnx)><GEN_OPSET_MODEL>.onnx')
    parser.add_argument('--simplify', action='store_true',help='model simplifier')
    parser.add_argument('--optimize', action='store_true',help='model optimizer')
    parser.add_argument('--external_data', nargs=1,help='externalize intializers in a Model.name data file')
    parser.add_argument('--onnxscript', nargs=1,help='generates an onnx script from the model. Use of --external_data reduces the script size')
    parser.add_argument('--disable_postcheck', action='store_true',help='Disables the model post checking (to be able to debug through )')
    parser.add_argument('--remove_dead_trees', action='store_true',help='Remove sub trees which do not contribute to output')
    parser.add_argument('--sdoc_trace', nargs=2,help='Export strictdoc trace in file argument ')
    parser.add_argument('--output', help='output modified onnx model')
    args = parser.parse_args()

    onnx_model = onnx.load(args.input)
    onnx.checker.check_model(onnx_model)

    if args.opset_change:
        onnx_model = version_converter.convert_version(onnx_model, args.opset_change)

    print(f'model IR version: {onnx_model.ir_version}')
    if args.set_model_domain:
        onnx_model.domain = args.set_model_domain
    if onnx_model.domain != '':
        print(f'model domain: {onnx_model.domain}')
    else:
        print(f'{bcolors.WARNING}  WARN: missing model domain !{bcolors.ENDC}')

    opsets={}
    for ops in onnx_model.opset_import:
        domain = 'ai.onnx' if ops.domain=='' else ops.domain
        opsets[domain] = ops.version
        print(f"model opset {domain}: {ops.version}")

    if args.remove_dead_trees:
        ''' try colouring tree from output untill input and remove other nodes. '''
        pass
        

    if args.silu:
        from onnxscript.rewriter import pattern,rewrite
        from onnxscript import opset17 as op,ir,script,FLOAT
        from onnxscript.values import Opset
        def x_sigmoid(op,x):
            return x*op.Sigmoid(x)
        local = Opset("ai.onnx.contrib", 1)
        @script(opset=local, default_opset=op)
        def Silu( x ) :
            return x*op.Sigmoid(x)
        def SiluRepl(opset,x):
            return opset.Silu(x,domain="ai.onnx.contrib")

        def apply_rewrite(model):
            rule = pattern.RewriteRule(x_sigmoid, SiluRepl)
            model_with_rewrite_applied = rewrite(model,pattern_rewrite_rules=[rule])
            return model_with_rewrite_applied

        onnx_model = apply_rewrite(onnx_model)
        #print (onnx_model)
        #Silu = X * sigmoid
        '''previous = None
        todelete=[]
        number_of_match=0
        for n in onnx_model.graph.node:
            if n.op_type == 'Mul' and previous.op_type == 'Sigmoid':
                #replace Mul by Silu and cleanup links
                n.op_type = 'Silu'
                inp = n.input[0]
                n.ClearField('input')
                n.input.append(inp)
                previous.ClearField('input')
                n.domain='ai.onnx.contrib'
                todelete.append(previous.name)
                number_of_match+=1
            previous = n
        #filter out Sigmoid nodes
        print (f'Number of match {number_of_match}')
        nodes = [n for n in onnx_model.graph.node  if n.name not in todelete]
        onnx_model.graph.ClearField('node')
        onnx_model.graph.node.extend(nodes)        
        '''
        onnx_model.functions.append(Silu.to_function_proto())
        onnx_model.opset_import.append(onnx.helper.make_opsetid('ai.onnx.contrib',1))
        #print(onnx_model)

    if args.raw_data_convert:
        def tensor_raw_data_to_data_type(tensor):
                npdtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
                d = np.frombuffer(tensor.raw_data,dtype=npdtype)
                field = onnx.helper.tensor_dtype_to_field(tensor.data_type)
                getattr(tensor, field).extend(d)
                tensor.ClearField('raw_data')
        # raw data in intitializers
        for i in onnx_model.graph.initializer:
            if len(i.raw_data) > 0:
                tensor_raw_data_to_data_type(i)
        # raw data in node attributes
        for n in onnx_model.graph.node:
            for a in n.attribute:
                if a.type == onnx.AttributeProto.TENSOR:
                    tensor_raw_data_to_data_type(a.t)

    if args.gen_opset_model:
        for domain in opsets.keys():
            opsetproto = OperatorSetProto()
            opsetproto.magic = "ONNXOPSET"
            opsetproto.ir_version = IR_VERSION
            opsetproto.domain = domain
            opsetproto.opset_version = opsets[domain]
            operators={}
            for x in onnx.defs.get_all_schemas_with_history():
                if not x.has_function and (x.domain==domain or (x.domain=='' and domain=='ai.onnx')):
                    opproto = OperatorProto()
                    opproto.op_type  = x.name
                    opproto.since_version = x.since_version
                    opproto.status = STABLE
                    if x.name not in operators and x.since_version <= opsets[domain] :
                        operators[x.name]=opproto
                    else:
                        if x.since_version <= opsets[domain] and x.since_version > operators[x.name].since_version:
                            operators[x.name]=opproto
            for opproto in operators.keys():
                opsetproto.operator.append(operators[opproto])
            with open(f'{domain}{args.gen_opset_model}','wb') as f:
                f.write(opsetproto.SerializeToString())


    print(f'graph name: {onnx_model.graph.name}')
    if onnx_model.graph.doc_string !='':
        print(f'graph doc_string: {onnx_model.graph.doc_string}')

    if args.depend:
        print('\nOperator dependencies:')
        local_func = [f.name for f in onnx_model.functions]
        op = {}
        for typ in onnx_model.graph.node:
            if typ.op_type not in local_func:
                op[typ.op_type] = defs.get_schema(typ.op_type,opsets['ai.onnx']).since_version
        #prints the onnx operator used in the model except local functions
        sops = [f'{x}\t\tv{op[x]}' for x in sorted(list(op.keys()))]

        print ('\n'.join(sops))

    if args.simplify:
        from onnxsim import simplify
        onnx_model,check = simplify(onnx_model)

    if args.optimize:
        from onnxoptimizer import get_fuse_and_elimination_passes,optimize
        passes = get_fuse_and_elimination_passes()
        onnx_model = optimize(model=onnx_model, passes=passes)

    if args.shapeinf:
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model,True,True,True)


    if args.metadata is not None:
        import json
        with open(args.metadata[0],'r') as f:
            array_repl = json.load(f)
            for k,v in array_repl.items():
                s = onnx.StringStringEntryProto(key=k,value=v)
                onnx_model.metadata_props.append(s)

    if args.doc_string is not None:
        names_dic = dict()
        names_dic[onnx_model.graph.name] = onnx_model.graph
        for n in onnx_model.graph.input:
            names_dic[n.name] = n
        for n in onnx_model.graph.output:
            names_dic[n.name] = n
        ''' value_info is created by shape inference for every activation tensor 
        but it is not working with domain other than ai.onnx (ex: com.microsoft) '''
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model,True,True,True)
        for n in onnx_model.graph.value_info:
            names_dic[n.name] = n
        for n in onnx_model.graph.node:
            names_dic[n.name] = n
        for n in onnx_model.graph.initializer:
            names_dic[n.name] = n
        import json
        with open(args.doc_string[0],'r') as f:
            array_repl = json.load(f)
            for repl_key in array_repl.keys():
                if repl_key in names_dic:
                    names_dic[repl_key].doc_string = array_repl[repl_key]
                else:
                    print(f'  {bcolors.WARNING}WARN: doc_string file {args.doc_string[0]}, key:{repl_key} not found in the model{bcolors.ENDC}')

    if args.external_data:
        from onnx.external_data_helper import convert_model_to_external_data
        convert_model_to_external_data(onnx_model, all_tensors_to_one_file=True, location=args.external_data[0] , size_threshold=10, convert_attribute=True)

    if args.sdoc_trace:
        doc = Sdoc(args.sdoc_trace[1],'1','PUBLIC')
        for i,n in enumerate(onnx_model.graph.node):
            doc.new_req('MLMD_REQ_'+str(i),n.name,'ONNX node',n.doc_string)
        doc.write(args.sdoc_trace[0])

    if args.output:
        onnx.save(onnx_model,args.output)

    if not args.disable_postcheck:
        onnx.checker.check_model(onnx_model)

    if args.onnxscript:
        from onnxscript import proto2python
        python_code = proto2python(onnx_model, use_operators=True, inline_const=True)
        with open(args.onnxscript[0],'w') as script:
            script.write(python_code)

