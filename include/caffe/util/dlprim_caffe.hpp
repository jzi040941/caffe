#pragma once
#include <dlprim/tensor.hpp>
namespace caffe {
namespace dputil {

    template<typename T>
    dlprim::Shape to_shape(std::vector<T> const &caffe_shape)
    {
        return dlprim::Shape::from_range(caffe_shape.begin(),caffe_shape.end());
    }

    template<typename T>
    dlprim::Tensor blob_data_or_diff_to_tensor(Blob<T> &blob,bool is_data)
    {
        auto ptr = is_data ? blob.mutable_gpu_data() : blob.mutable_gpu_diff();
        cl_mem mem = reinterpret_cast<cl_mem>(ptr);
        dlprim::DataType dt = dlprim::TypeTraits<T>::data_type;
        dlprim::Shape shape = to_shape(blob.shape());
        return dlprim::Tensor(cl::Buffer(mem,true),0,shape,dt);
    }

    template<typename T>
    dlprim::Tensor blob_data_or_diff_to_tensor(Blob<T> const &blob,bool is_data)
    {
        auto ptr = is_data ? blob.gpu_data() : blob.gpu_diff();
        cl_mem mem = reinterpret_cast<cl_mem>(ptr);
        dlprim::DataType dt = dlprim::TypeTraits<T>::data_type;
        dlprim::Shape shape = to_shape(blob.shape());
        return dlprim::Tensor(cl::Buffer(mem,true),0,shape,dt);
    }
    template<typename T>
    dlprim::Tensor blob_data_to_tensor(Blob<T> const &blob)
    {
        return blob_data_or_diff_to_tensor(blob,true);
    }
    template<typename T>
    dlprim::Tensor blob_diff_to_tensor(Blob<T> const &blob)
    {
        return blob_data_or_diff_to_tensor(blob,false);
    }
    template<typename T>
    dlprim::Tensor blob_data_to_tensor(Blob<T> &blob)
    {
        return blob_data_or_diff_to_tensor(blob,true);
    }
    template<typename T>
    dlprim::Tensor blob_diff_to_tensor(Blob<T> &blob)
    {
        return blob_data_or_diff_to_tensor(blob,false);
    }

    dlprim::Context context_from_id(int dev_id)
    {
        viennacl::ocl::context &vctx = viennacl::ocl::get_context(dev_id);
        cl::Context cl_ctx(vctx.handle().get(),true);
        cl::Device cl_dev(vctx.current_device().id(),true);
        cl_platform_id pid = cl_dev.getInfo<CL_DEVICE_PLATFORM>();
        cl::Platform cl_plat(pid,true);
        dlprim::Context ctx(cl_ctx,cl_plat,cl_dev);
        return ctx;
    }

    dlprim::ExecutionContext ec_from_id(int dev_id)
    {
        viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
        cl_command_queue queue = ctx.get_queue().handle().get();
        cl::CommandQueue q(queue,true);
        return dlprim::ExecutionContext(q);
    }
} // dputil
}
