from invoke import task

@task
def clean(c, docs=False, bytecode=False, extra=''):
    patterns = ['build','doc/02_specifications/**','MLMD/*.onnx','MLMD/*.ort','MLMD/*.config']
    if docs:
        patterns.append('docs/_build')
    if bytecode:
        patterns.append('**/*.pyc')
    if extra:
        patterns.append(extra)
    for pattern in patterns:
        c.run("rm -rf {}".format(pattern))


@task
def build(c, docs=False):
     ortcommand = 'python -m onnxruntime.tools.convert_onnx_models_to_ort --optimization_style=Fixed  --save_optimized_onnx_model --output_dir MLMD MLMD'
     c.run("python -m TFM.tfm-keras MLMD/KX.onnx doc/02_specifications")
     c.run("python -m TFM.tfm-torch MLMD/TDX.onnx MLMD/LTX.onnx doc/02_specifications")
     c.run("python -m annotate.mlmd_annotate --input MLMD/KX.onnx --sdoc_trace doc/02_specifications/MLMD_KERAS_REQ.sdoc MLMD_KS")
     c.run(ortcommand)
