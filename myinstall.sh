# CMAKE_POSITION_INDEPENDENT_CODE=ON NVCC_PREPEND_FLAGS="--forward-unknown-opts" DS_BUILD_OPS=1 DS_BUILD_SPARSE_ATTN=0 DS_BUILD_EVOFORMER_ATTN=0 DS_BUILD_QUANTIZER=0 DS_BUILD_AIO=0 DS_BUILD_TRANSFORMER_INFERENCE=0 pip install . --global-option="build_ext" --global-option="-j48"

set +x
module load cudatoolkit-standalone/11.8.0 craype/2.7.15 cray-dsmml/0.2.2 libfabric/1.11.0.4.125 craype-network-ofi cray-pmi/6.1.2 cray-pmi-lib/6.0.17 cray-pals/1.1.7 cray-libpals/1.1.7 PrgEnv-gnu/8.3.3 conda/2023-01-10-unstable gcc/11.2.0 cray-mpich/8.1.16 cray-hdf5-parallel/1.12.1.3 cmake/3.23.2
export PATH=/home/am6429/.conda/envs/dspeed_env/bin:/soft/datascience/conda/2023-01-10/mconda3/condabin:/soft/buildtools/cmake/cmake-3.23.2/cmake-3.23.2-linux-x86_64/bin:/opt/cray/pe/hdf5-parallel/1.12.1.3/bin:/opt/cray/pe/hdf5/1.12.1.3/bin:/opt/cray/pe/gcc/11.2.0/bin:/soft/compilers/cudatoolkit/cuda-11.8.0/bin:/soft/libraries/nccl/nccl_2.16.2-1+cuda11.8_x86_64/include:/opt/cray/pe/pals/1.1.7/bin:/opt/cray/libfabric/1.11.0.4.125/bin:/opt/cray/pe/craype/2.7.15/bin:/opt/clmgr/sbin:/opt/clmgr/bin:/opt/sgi/sbin:/opt/sgi/bin:/home/am6429/.local/bin:/usr/local/bin:/usr/bin:/bin:/opt/c3/bin:/usr/lib/mit/bin:/usr/lib/mit/sbin:/opt/pbs/bin:/sbin:/home/am6429/veloc-build/include:/home/am6429/veloc-build/bin:/opt/cray/pe/bin
export LD_LIBRARY_PATH=/opt/cray/pe/gcc/11.2.0/snos/lib64:/soft/compilers/cudatoolkit/cuda-11.8.0/extras/CUPTI/lib64:/soft/compilers/cudatoolkit/cuda-11.8.0/lib64:/soft/libraries/trt/TensorRT-8.5.2.2.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/lib:/soft/libraries/nccl/nccl_2.16.2-1+cuda11.8_x86_64/lib:/soft/libraries/cudnn/cudnn-11-linux-x64-v8.6.0.163/lib:/opt/cray/libfabric/1.11.0.4.125/lib64:/home/am6429/veloc-build/lib:/home/am6429/veloc-build/lib64:/home/am6429/nvcomp/lib
export LD_LIBRARY_PATH=/usr/lib64/:$LD_LIBRARY_PATH
CMAKE_POSITION_INDEPENDENT_CODE=ON NVCC_PREPEND_FLAGS="--forward-unknown-opts" DS_BUILD_AIO=1 DS_BUILD_CCL_COMM=1 DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_CPU_ADAGRAD=1 DS_BUILD_FUSED_LAMB=1 DS_BUILD_QUANTIZER=1 DS_BUILD_RANDOM_LTD=1 DS_BUILD_TRANSFORMER=1 DS_BUILD_STOCHASTIC_TRANSFORMER=1 DS_BUILD_UTILS=1 DS_BUILD_VELOC_CKPT=1 pip install . --global-option="build_ext" --global-option="-j48"
set -x