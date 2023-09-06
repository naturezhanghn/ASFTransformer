# CUDA_VISIBLE_DEVICES=1
python setup.py develop --no_cuda_ext
python basicsr/train.py -opt ./Turbulence/Options/ASFTransformer_nature_algorithm_simulated_data.yml