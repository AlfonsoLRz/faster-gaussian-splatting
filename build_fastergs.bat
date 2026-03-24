python ./scripts/install.py -m FasterGS
cd C:\Github\Forks\nerficg\src\Methods\FasterGS\FasterGSCudaBackend
python setup.py build_ext --inplace
python -c "import torch; from FasterGSCudaBackend.FasterGSCudaBackend import _C; print(_C)"