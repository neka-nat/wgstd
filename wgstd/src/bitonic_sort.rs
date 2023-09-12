use super::context::WgContext;
use super::device_vec::DeviceVec;

use bytemuck;
use wgpu::util::DeviceExt;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBinding, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
    ShaderStages,
};

const BITONIC_SORT_STEP_KERNEL: &str = r#"
    @group(0) @binding(0) var<storage, read_write> buffer: array<{{ T }}>;

    struct SortParams {
        j: u32,
        k: u32
    };

    @group(0) @binding(1) var<uniform> params: SortParams;

    @compute
    @workgroup_size(1)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let i = u32(global_id.x);
        let ixj = i ^ params.j;

        if (ixj > i) {
            if (((i & params.k) == 0u && buffer[i] > buffer[ixj]) || ((i & params.k) != 0u && buffer[i] < buffer[ixj])) {
                let tmp = buffer[i];
                buffer[i] = buffer[ixj];
                buffer[ixj] = tmp;
            }
        }
    }
"#;

pub fn bitonic_sort<'a, T>(context: &WgContext, vector: &'a DeviceVec<T>) -> &'a DeviceVec<T>
where
    T: bytemuck::Pod + bytemuck::Zeroable,
{
    let shader = context.device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(
            BITONIC_SORT_STEP_KERNEL
                .replace("{{ T }}", std::any::type_name::<T>())
                .into(),
        ),
    });

    let bind_group_layout = context
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(4),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(8),
                    },
                    count: None,
                },
            ],
        });

    let pipeline_layout = context
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = context
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

    let mut encoder = context
        .device
        .create_command_encoder(&CommandEncoderDescriptor { label: None });
    let size = vector.buffer.size() as usize / std::mem::size_of::<T>(); // Assuming the buffer holds T values.
    let mut j: usize;
    for k in (2..=size).map(|x| x.next_power_of_two()) {
        j = k >> 1;
        while j > 0 {
            let params = [j as u32, k as u32];
            let params_buffer =
                context
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("SortParamsBuffer"),
                        contents: bytemuck::cast_slice(&params),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC,
                    });

            // Create a new bind group with the params buffer
            let bind_group = context.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &vector.buffer,
                            offset: 0,
                            size: None,
                        }),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &params_buffer,
                            offset: 0,
                            size: None,
                        }),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: None });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(size as u32, 1, 1);
            j >>= 1;
        }
    }

    context.queue.submit(std::iter::once(encoder.finish()));

    vector
}
