from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='dcn',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='deform_conv_cuda',
            sources=[
                'src/deform_conv_cuda.cpp',
                'src/deform_conv_cuda_kernel.cu'
            ],
        ),
        cpp_extension.CUDAExtension(
            name='deform_pool_cuda',
            sources=[
                'src/deform_pool_cuda.cpp',
                'src/deform_pool_cuda_kernel.cu'
            ],
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
