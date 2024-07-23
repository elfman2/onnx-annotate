from invoke import task

@task
def clean(c, docs=False, bytecode=False, extra=''):
    patterns = ['build']
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
     c.run("python -m TFM.tfm-keras MLMD/KX.onnx")
     c.run("python -m TFM.tfm-torch MLMD/TDX.onnx MLMD/LTX.onnx")
     c.run(ortcommand)
