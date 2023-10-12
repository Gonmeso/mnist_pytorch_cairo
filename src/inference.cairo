use orion::operators::tensor::{TensorTrait, FP16x16Tensor, Tensor};
use orion::operators::nn::{NNTrait, FP16x16NN};
use orion::numbers::FP16x16;
use layer1::weights::tensor1 as w1;
use layer1::bias::tensor1 as b1;
use layer3::weights::tensor3 as w3;
use layer3::bias::tensor3 as b3;
use mnist_pytorch::functions as f;
use mnist_pytorch::input::input;

    
fn main() -> Tensor<FP16x16>{
	let _0 = input();
	let _1 = f::lin2(_0, w1(), b1());
	let _2 = f::relu3(_1);
	let _3 = f::lin4(_2, w3(), b3());
	let _4 = f::logsoftmax5(_3);
	_4
}