use orion::operators::tensor::{TensorTrait, FP16x16Tensor, Tensor};
use orion::operators::nn::{NNTrait, FP16x16NN};
use orion::numbers::FP16x16;

fn lin1(i: Tensor<FP16x16>, w: Tensor<FP16x16>, b: Tensor<FP16x16>) -> Tensor<FP16x16> {
	NNTrait::<FP16x16>::linear(i, w, b)
}


fn lin2(i: Tensor<FP16x16>, w: Tensor<FP16x16>, b: Tensor<FP16x16>) -> Tensor<FP16x16> {
	NNTrait::<FP16x16>::linear(i, w, b)
}


fn relu3(i: Tensor<FP16x16>) -> Tensor<FP16x16> {
	NNTrait::<FP16x16>::relu(@i)
}


fn lin4(i: Tensor<FP16x16>, w: Tensor<FP16x16>, b: Tensor<FP16x16>) -> Tensor<FP16x16> {
	NNTrait::<FP16x16>::linear(i, w, b)
}


fn logsoftmax5(i: Tensor<FP16x16>) -> Tensor<FP16x16> {
	NNTrait::<FP16x16>::logsoftmax(@i, 0)
}

