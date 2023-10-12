use array::{SpanTrait, ArrayTrait};
use orion::operators::tensor::{TensorTrait, FP16x16Tensor, Tensor};
use orion::numbers::{FixedTrait, FP16x16};
fn tensor3() -> Tensor<FP16x16> {
TensorTrait::<FP16x16>::new(array![10].span(), array![FixedTrait::<FP16x16>::new(36874, true), FixedTrait::<FP16x16>::new(67707, false), FixedTrait::<FP16x16>::new(11316, false), FixedTrait::<FP16x16>::new(8227, false), FixedTrait::<FP16x16>::new(30571, true), FixedTrait::<FP16x16>::new(85160, false), FixedTrait::<FP16x16>::new(14821, true), FixedTrait::<FP16x16>::new(55861, false), FixedTrait::<FP16x16>::new(112041, true), FixedTrait::<FP16x16>::new(17334, false)].span())
}

