use super::context::WgContext;
use bytemuck;
use std::mem::size_of;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor};

pub struct DeviceVec<T>
where
    T: bytemuck::Pod + bytemuck::Zeroable,
{
    pub buffer: Buffer,
    phantom: std::marker::PhantomData<T>,
}

impl<T> DeviceVec<T>
where
    T: bytemuck::Pod + bytemuck::Zeroable,
{
    pub fn new(context: &WgContext, size: usize) -> Self {
        let buffer = context.device.create_buffer(&BufferDescriptor {
            label: None,
            size: size_of::<T>() as u64 * size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            phantom: std::marker::PhantomData,
        }
    }

    pub fn copy_from_slice(&self, context: &WgContext, slice: &[T]) {
        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });

        let buffer = context.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(slice),
            usage: BufferUsages::COPY_SRC,
        });

        encoder.copy_buffer_to_buffer(
            &buffer,
            0,
            &self.buffer,
            0,
            (size_of::<T>() * slice.len()) as u64,
        );

        context.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn copy_to_slice(&self, context: &WgContext, slice: &mut [T]) {
        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });

        let buffer = context.device.create_buffer(&BufferDescriptor {
            label: None,
            size: (size_of::<T>() * slice.len()) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &self.buffer,
            0,
            &buffer,
            0,
            (size_of::<T>() * slice.len()) as u64,
        );

        context.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

        context.device.poll(wgpu::Maintain::Wait);

        let padded_data = buffer_slice.get_mapped_range();
        slice.copy_from_slice(bytemuck::cast_slice(&padded_data));
    }
}
