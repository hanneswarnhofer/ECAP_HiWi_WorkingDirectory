#!/bin/bash
conda activate astro_dl
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

echo 'export ASTRODLENV='$CONDA_DEFAULT_ENV >>$HOME/.bashrc
echo 'export PYTHONPATH='$PWD/../:'$PYTHONPATH' >>$HOME/.bashrc
echo 'export PYTHONPATH='$PWD'/../../':'$PYTHONPATH' >>$HOME/.bashrc

# to enable CUDA support -> re-activate env
conda activate astro_dl
