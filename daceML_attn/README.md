# Optimizing Different Attention Mechanisms for Data Movement

### Robert Wakefield-Carl, Ashwin Muralidharan

## Setup Guide
The easiest way to set up everything is to use conda and set up as follows. To run the optimized encoder from the data movement paper, we have included a simple setup script called ``patch.sh`` located in the top directory that you can use to set everything up.

### Configure conda environment
    conda install -n base conda-libmamba-solver
    conda config --set solver libmamba
    conda update -n base conda
    conda create -n py37 python=3.7
    conda init
    conda activate py37
    conda install mkl-include
You may have to run ``source ~/.bashrc`` or restart your terminal after running conda init for the first time.

### Install python packages
    pip install -r requirements.txt

### Insall onxx runtime
DaCeML requires a patched version of the runtime that can be installed as follows. It might be useful to include the export statement in your ``~/.bashrc``.

    wget https://github.com/orausch/onnxruntime/releases/download/v2/onnxruntime-daceml-patched.tgz
    tar -xzf onnxruntime-daceml-patched.tgz
    export ORT_RELEASE=</path/to/extracted/runtime>

### Install VSCode Extensions
If you want to interact with the generated sdfgs you can install the vscode extensions that are included in the ``VSCode_Extensions`` folder. When launching the dace backend you may need to activate the created conda environment then try again.

## Running the attention mechanisms
Each attention mechanism implementation is in a separate file named the mechanism implemented. You can run each to profile a given attention or encoder layer. See the code for how to select which to run and how to enable DaCeML optimizations. To view the sdfg for a given attention mechanism look inside the ``.dacecache`` folder.

Refer to the DaCe and DaCeML docs for more information

https://spcldace.readthedocs.io/en/latest/index.html

https://daceml.readthedocs.io/en/latest/index.html
