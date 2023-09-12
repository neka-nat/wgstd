use futures::executor::block_on;

use wgstd::bitonic_sort::bitonic_sort;
use wgstd::context::WgContext;
use wgstd::device_vec::DeviceVec;

fn main() {
    let context = WgContext::new();
    let context = block_on(context);

    let vector: DeviceVec<u32> = DeviceVec::new(&context, 8);
    vector.copy_from_slice(&context, &[1, 3, 2, 5, 4, 6, 7, 8]);

    let sorted = bitonic_sort(&context, &vector);

    let mut result = [0; 8];
    sorted.copy_to_slice(&context, &mut result);

    println!("{:?}", result);
}
