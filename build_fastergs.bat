"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
set DISTUTILS_USE_SDK=1
python ./scripts/install.py -m FasterGS
cd C:\Github\Forks\nerficg\src\Methods\FasterGS\FasterGSCudaBackend
python setup.py build_ext --inplace
python -c "import torch; from FasterGSCudaBackend.FasterGSCudaBackend import _C; print(_C)"