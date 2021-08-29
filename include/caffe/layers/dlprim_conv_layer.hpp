#if 1 //def USE_DLPRIM
#ifndef CAFFE_DLPRIM_CONV_LAYER_HPP_
#define CAFFE_DLPRIM_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"

namespace dlprim {
    namespace core {
        class Conv2DForward;
        class Conv2DBackwardData;
        class Conv2DBackwardFilter;
        class BiasBackwardFilter;
    }
    class Tensor;
}

namespace caffe {

template <typename Dtype>
class DLPrimConvolutionLayer : public ConvolutionLayer<Dtype> {
 public:
  explicit DLPrimConvolutionLayer(const LayerParameter& param);
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~DLPrimConvolutionLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


 private:
  std::vector<int_tp> in_sh_,out_sh_;
  std::unique_ptr<dlprim::core::Conv2DForward> fwd_;
  std::unique_ptr<dlprim::core::Conv2DBackwardData> bwd_data_;
  std::unique_ptr<dlprim::core::Conv2DBackwardFilter> bwd_filter_;
  std::unique_ptr<dlprim::core::BiasBackwardFilter> bwd_bias_;
  std::unique_ptr<dlprim::Tensor> ws_;
};

}  // namespace caffe

#endif  // CAFFE_DLPRIM_CONV_LAYER_HPP_
#endif  // USE_DLPRIM
