use array::{SpanTrait, ArrayTrait};
use orion::operators::tensor::{TensorTrait, FP16x16Tensor, Tensor};
use orion::numbers::{FixedTrait, FP16x16};
fn tensor1() -> Tensor<FP16x16> {
TensorTrait::<FP16x16>::new(array![10].span(), array![FixedTrait::<FP16x16>::new(3880, true), FixedTrait::<FP16x16>::new(8268, true), FixedTrait::<FP16x16>::new(10634, true), FixedTrait::<FP16x16>::new(36590, true), FixedTrait::<FP16x16>::new(25104, false), FixedTrait::<FP16x16>::new(2377, false), FixedTrait::<FP16x16>::new(60221, false), FixedTrait::<FP16x16>::new(12225, true), FixedTrait::<FP16x16>::new(12615, true), FixedTrait::<FP16x16>::new(29904, false)].span())
}

