use wgpu::util::DeviceExt;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBinding, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
    ShaderStages,
};

use super::{context::WgContext, device_vec::DeviceVec};
use bytemuck;

pub fn scan_inclusive<'a, T>(context: &WgContext, vector: &'a DeviceVec<T>) -> &'a DeviceVec<T>
where
    T: bytemuck::Pod + bytemuck::Zeroable,
{
    // WGSL カーネル: 各ステップで offset 分だけ前の要素を加算する。
    // ここでは非常にシンプルな 1 ワークグループ サイズ (workgroup_size(1)) にして
    // そのまま全要素を直列的に動かす実装
    // 大規模な配列を扱うには、より最適化された実装が必要
    const SCAN_STEP_KERNEL: &str = r#"
@group(0) @binding(0) var<storage, read_write> buffer: array<{{ T }}>;

struct ScanParams {
    offset: u32
};

@group(0) @binding(1) var<uniform> params: ScanParams;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.offset) {
        // prefix sum (inclusive): buffer[i] = buffer[i] + buffer[i - offset]
        buffer[i] = buffer[i] + buffer[i - params.offset];
    }
}
"#;

    let shader = context.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ScanStepShader"),
        source: ShaderSource::Wgsl(
            SCAN_STEP_KERNEL.replace("{{ T }}", std::any::type_name::<T>()).into(),
        ),
    });

    let bind_group_layout = context.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("ScanBindGroupLayout"),
        entries: &[
            // buffer (storage, read_write)
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // params (uniform)
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // PipelineLayout の定義
    let pipeline_layout = context
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("ScanPipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    // ComputePipeline の作成
    let pipeline = context
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ScanComputePipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

    let size = vector.buffer.size() as usize / std::mem::size_of::<T>();
    if size <= 1 {
        return vector;
    }

    let mut offset = 1;
    let mut encoder = context
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("ScanCommandEncoder"),
        });

    while offset < size {
        let params = [offset as u32];
        let params_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ScanParamsBuffer"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC,
            });

        let bind_group = context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("ScanBindGroup"),
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

        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("ScanComputePass"),
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(size as u32, 1, 1);
        }

        offset <<= 1;
    }

    context.queue.submit(std::iter::once(encoder.finish()));

    vector
}
