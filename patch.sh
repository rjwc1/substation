#!/bin/bash
export PYTHONPATH=$(pwd)
pip3 install transformers tabulate dace
# cp -f ./dtypes.py ./dace/dace/dtypes.py
# cp -f ./newast.py ./dace/dace/frontend/python/newast.py
# cd ./dace
# python3 ./setup.py build
# python3 ./setup.py install
# pip install --editable .
cd ./gpuwait
python3 ./setup.py build
python3 ./setup.py install
cd ../pycudaprof
python3 ./setup.py build
python3 ./setup.py install
cd ../tc_profiling
bash ./compile.sh
cd ..
rm -r ../results
mkdir ../results
cd ./tc_profiling
python3 ./einsum_perms.py --b 8 --j 512 --h 16 --i 1024 --output ../../results
cd ../pytorch_module
python3 run_encoder.py --num-heads 16 --emb-size 1024 --max-seq-len 128 --batch-size 96 --layout layouts-bert-b96-l128-h16-e1024.pickle
#python3 ./benchmark.py --kernel softmax --size "H=16,B=96,J=128,K=128,U=4096,N=1024,P=64" --time --out-dir ../../results
#cd ../config_selection
#python3 ./parse_tc_results.py ../../results
#python3 ./optimize.py --output_config my_layouts.pickle ../../results