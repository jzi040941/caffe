#ifdef USE_DLPRIM
#include <algorithm>
#include <vector>

#include <dlprim/core_ops.hpp>
#include "caffe/layers/dlprim_conv_layer.hpp"
#include "caffe/util/dlprim_caffe.hpp"

namespace caffe {

template <typename Dtype>
void DLPrimConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  this->use_colbuffer_ = false;


  Reshape(bottom, top);
}

template <typename Dtype>
void DLPrimConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  this->use_colbuffer_ = false;

  ConvolutionLayer<Dtype>::Reshape(bottom, top);

  bool shapes_changed = false;
  if (fwd_.get() != nullptr) {
    auto &new_in_sh = bottom[0]->shape();
    auto &new_out_sh = top[0]->shape();
    bool in_eq = in_sh_.size() == new_in_sh.size()
                 && in_sh_[0] >= new_in_sh[0] 
                 && std::equal(in_sh_.begin()+1,in_sh_.end(),new_in_sh.begin()+1);
    bool out_eq = out_sh_.size() == new_out_sh.size()
                 && out_sh_[0] >= new_out_sh[0] 
                 && std::equal(out_sh_.begin()+1,out_sh_.end(),new_out_sh.begin()+1);
    shapes_changed = !in_eq || !out_eq;
  }
  in_sh_ = bottom[0]->shape();
  out_sh_ = top[0]->shape();

  if (fwd_.get() == nullptr || shapes_changed) {
    dlprim::Convolution2DConfigBase cfg;
    int_tp* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
    int_tp* pad_data = this->pad_.mutable_cpu_data();
    int_tp* stride_data = this->stride_.mutable_cpu_data();
    int_tp* dilation_data = this->dilation_.mutable_cpu_data();

    std::vector<int_tp> kernel_vec;
    std::vector<int_tp> pad_vec;
    std::vector<int_tp> stride_vec;
    std::vector<int_tp> dilation_vec;

    //CHECK_EQ(num_spatial_axes_,2);

    for (int_tp i = 0; i < this->num_spatial_axes_; ++i) {
        kernel_vec.push_back(kernel_shape_data[i]);
        pad_vec.push_back(pad_data[i]);
        stride_vec.push_back(stride_data[i]);
        dilation_vec.push_back(dilation_data[i]);
    }

    
    dlprim::Convolution2DConfigBase config;
	config.channels_in = in_sh_[1];
	config.channels_out = out_sh_[1];
    config.kernel[0] = kernel_vec[0];
    config.kernel[1] = kernel_vec[1];
	config.stride[0] = stride_vec[0];
	config.stride[1] = stride_vec[1];
	config.dilate[0] = dilation_vec[0];
	config.dilate[1] = dilation_vec[1];
	config.pad[0] = pad_vec[0];
	config.pad[1] = pad_vec[1];
	config.groups = this->group_;
    
    dlprim::core::Conv2DSettings core_config(config,dputil::to_shape(in_sh_),dlprim::TypeTraits<Dtype>::data_type);

    dlprim::Context ctx = dputil::context_from_id(this->device_->id());


    fwd_ = std::move(dlprim::core::Conv2DForward::create(ctx,core_config,this->bias_term_));
    bwd_data_ = std::move(dlprim::core::Conv2DBackwardData::create(ctx,core_config));
    bwd_filter_ = std::move(dlprim::core::Conv2DBackwardFilter::create(ctx,core_config));
    if(this->bias_term_) {
        bwd_bias_ = std::move(dlprim::core::BiasBackwardFilter::create(ctx,dputil::to_shape(out_sh_)));
    }
    size_t ws_size = std::max(fwd_->workspace(),bwd_data_->workspace());
    ws_size = std::max(ws_size,bwd_filter_->workspace());
    if(this->bias_term_)
        ws_size = std::max(ws_size,bwd_bias_->workspace());

    if(ws_size > 0)
        ws_.reset(new dlprim::Tensor(ctx,dlprim::Shape(ws_size)));
    else
        ws_.reset(new dlprim::Tensor());

    //config.weights_backward = this->param_propagate_down_[0];
    //config.bias_backward = this->param_propagate_down_[1];

    //CHECK_EQ(std::is_same<Dtype, float>::value,true);
  }
}

template<typename Dtype>
DLPrimConvolutionLayer<Dtype>::DLPrimConvolutionLayer(const LayerParameter& param) : ConvolutionLayer<Dtype>(param) 
{
}

template <typename Dtype>
DLPrimConvolutionLayer<Dtype>::~DLPrimConvolutionLayer() {
}

template <typename Dtype>
void DLPrimConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  dlprim::Tensor weight = dputil::blob_data_to_tensor(*this->blobs_[0]);
  dlprim::Tensor bias;
  if (this->bias_term_) {
     bias = dputil::blob_data_to_tensor(*this->blobs_[1]);
  }

  dlprim::ExecutionContext ec = dputil::ec_from_id(this->device_->id());
  for (int_tp i = 0; i < bottom.size(); ++i) {
    dlprim::Tensor x = dputil::blob_data_to_tensor(*bottom[i]);
    dlprim::Tensor y = dputil::blob_data_to_tensor(*top[i]);
    fwd_->enqueue(x,
                  weight,
                  (this->bias_term_ ? &bias : nullptr),
                  y,
                  *ws_,
                  0.0f,
                  ec);
  }
}

template <typename Dtype>
void DLPrimConvolutionLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  dlprim::Tensor W  = dputil::blob_data_to_tensor(*this->blobs_[0]);
  dlprim::Tensor dW = dputil::blob_diff_to_tensor(*this->blobs_[0]);
  dlprim::Tensor dB;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    dB = dputil::blob_diff_to_tensor(*this->blobs_[1]);
  }


  dlprim::ExecutionContext ec = dputil::ec_from_id(this->device_->id());
  for (int_tp i = 0; i < top.size(); ++i) {
    dlprim::Tensor dY = dputil::blob_diff_to_tensor(*top[i]);
    dlprim::Tensor  X = dputil::blob_data_to_tensor(*bottom[i]);
    dlprim::Tensor dX = dputil::blob_diff_to_tensor(*bottom[i]);
    if(propagate_down[i]) {
      // enqueue(Tensor &dx,Tensor &w,Tensor &dy,Tensor &ws,float factor,ExecutionContext const &e) = 0;
      bwd_data_->enqueue(dX,W,dY,*ws_,0.0f,ec);
    }
    if(this->param_propagate_down_[0]) {
      bwd_filter_->enqueue(X,dW,dY,*ws_,1.0f,ec);
    }
    if(this->bias_term_ && this->param_propagate_down_[1]) {
      bwd_bias_->enqueue(dY,dB,*ws_,1.0f,ec);
    }
  }
}


INSTANTIATE_CLASS(DLPrimConvolutionLayer);


}   // namespace caffe
#endif  // USE_LIBDNN
