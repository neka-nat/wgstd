use futures::executor::block_on;
use wgstd::{scan_inclusive, WgContext, DeviceVec};

fn main() {
    let context = WgContext::new();
    let context = block_on(context);

    let input_data = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
    println!("Input data:  {:?}", input_data);

    let vector: DeviceVec<u32> = DeviceVec::new(&context, input_data.len());
    vector.copy_from_slice(&context, &input_data);

    scan_inclusive(&context, &vector);

    let mut result = vec![0u32; input_data.len()];
    vector.copy_to_slice(&context, &mut result);

    println!("Scan result: {:?}", result);
    // [1, 3, 6, 10, 15, 21, 28, 36]
}
