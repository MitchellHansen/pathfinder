// pathfinder/metal/src/lib.rs
//
// Copyright © 2019 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A Vulkan implementation of the device abstraction.

#![allow(non_upper_case_globals)]

#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate winit;


use byteorder::{NativeEndian, WriteBytesExt};
use foreign_types::{ForeignType, ForeignTypeRef};
use half::f16;
use pathfinder_geometry::rect::RectI;
use pathfinder_geometry::vector::Vector2I;
use pathfinder_gpu::{BlendFactor, BlendOp, BufferData, BufferTarget, BufferUploadMode, DepthFunc};
use pathfinder_gpu::{Device, Primitive, RenderState, RenderTarget, ShaderKind, StencilFunc};
use pathfinder_gpu::{TextureData, TextureDataRef, TextureFormat, TextureSamplingFlags};
use pathfinder_gpu::{UniformData, VertexAttrClass, VertexAttrDescriptor, VertexAttrType};
use pathfinder_resources::ResourceLoader;
use pathfinder_simd::default::{F32x2, F32x4};
use std::cell::{Cell, RefCell};
use std::mem;
use std::ptr;
use std::rc::Rc;
use std::slice;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use vulkano::device::{Device as VkDevice, Queue};
use vulkano::instance::Instance;
use vulkano::format::*;
use vulkano::swapchain::{Surface, ColorSpace, FullscreenExclusive};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, BufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::DeviceExtensions;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::{SwapchainImage, ImmutableImage, Dimensions, ImageUsage, MipmapsCount, ImageLayout};
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError};
use vulkano::swapchain;
use vulkano::descriptor::descriptor_set::{PersistentDescriptorSetBuf, PersistentDescriptorSet};
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;
use winit::window::{Window, WindowBuilder};
use winit::event_loop::EventLoop;


use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use vulkano::pipeline::shader::{ShaderModule, ShaderInterfaceDef};
use vulkano::sampler::{Sampler, Filter, MipmapMode, SamplerAddressMode};
use vulkano::pipeline::vertex::{VertexDefinition, InputRate, AttributeInfo, IncompatibleVertexDefinitionError, VertexSource};
use shade_runner::Entry;
use vulkano::descriptor::DescriptorSet;


// =================================================================================================
// =================================================================================================
// =================================================================================================

impl RuntimeVertexDef {

    // Going to need to bring in the attributes populated from configure_vertex_attr?
    // This will need to be called in the prepare to draw methinks
    pub fn from_shader(shader: VulkanShader) -> RuntimeVertexDef {

        let mut buffers = Vec::new();
        let mut vertex_buffer_ids = Vec::new();
        let mut attributes = Vec::new();
        let mut num_vertices = u32::max_value();


        for (attribute_id, attribute) in primitive.attributes().enumerate() {

            // get the name of the attribute

            // get the vertex count
            shader.entry.layout.layout_data.descriptions
            let infos = AttributeInfo {
                offset: 0,
                format: match (accessor.data_type(), accessor.dimensions()) {
                    (DataType::I8, Dimensions::Scalar) => Format::R8Snorm,
                    (DataType::U8, Dimensions::Scalar) => Format::R8Unorm,
                    (DataType::F32, Dimensions::Vec2) => Format::R32G32Sfloat,
                    (DataType::F32, Dimensions::Vec3) => Format::R32G32B32Sfloat,
                    (DataType::F32, Dimensions::Vec4) => Format::R32G32B32A32Sfloat,
                    _ => unimplemented!()
                },
            };

            let view = accessor.view();
            buffers.push((attribute_id as u32, view.stride().unwrap_or(accessor.size()), InputRate::Vertex));
            attributes.push((name, attribute_id as u32, infos));
            vertex_buffer_ids.push((view.buffer().index(), view.offset() + accessor.offset()));
        }

        RuntimeVertexDef {
            buffers: buffers,
            vertex_buffer_ids: vertex_buffer_ids,
            num_vertices: num_vertices,
            attributes: attributes,
        }
    }

    /// Returns the indices of the buffers to bind as vertex buffers and the byte offset, when
    /// drawing the primitive.
    pub fn vertex_buffer_ids(&self) -> &[(usize, usize)] {
        &self.vertex_buffer_ids
    }
}

unsafe impl<I> VertexDefinition<I> for RuntimeVertexDef
    where I: ShaderInterfaceDef
{
    type BuffersIter = VecIntoIter<(u32, usize, InputRate)>;
    type AttribsIter = VecIntoIter<(u32, u32, AttributeInfo)>;

    fn definition(&self, interface: &I)
                  -> Result<(Self::BuffersIter, Self::AttribsIter), IncompatibleVertexDefinitionError>
    {
        let buffers_iter = self.buffers.clone().into_iter();

        let mut attribs_iter = self.attributes.iter().map(|&(ref name, buffer_id, ref infos)| {
            let attrib_loc = interface
                .elements()
                .find(|e| e.name.as_ref().map(|n| &n[..]) == Some(&name[..]))
                .unwrap()
                .location.start;
            (attrib_loc as u32, buffer_id, AttributeInfo { offset: infos.offset, format: infos.format })
        }).collect::<Vec<_>>();

        // Add dummy attributes.
        for binding in interface.elements() {
            if attribs_iter.iter().any(|a| a.0 == binding.location.start) {
                continue;
            }

            attribs_iter.push((binding.location.start, 0,
                               AttributeInfo { offset: 0, format: binding.format }));
        }

        Ok((buffers_iter, attribs_iter.into_iter()))
    }
}

unsafe impl VertexSource<Vec<Arc<dyn BufferAccess + Send + Sync>>> for RuntimeVertexDef {
    fn decode(&self, bufs: Vec<Arc<dyn BufferAccess + Send + Sync>>)
              -> (Vec<Box<dyn BufferAccess + Send + Sync>>, usize, usize)
    {
        (bufs.into_iter().map(|b| Box::new(b) as Box<_>).collect(), self.num_vertices as usize, 1)
    }
}

// =================================================================================================
// =================================================================================================
// =================================================================================================


const FIRST_VERTEX_BUFFER_INDEX: u64 = 1;

pub struct VulkanDevice {
    device: Arc<VkDevice>,

    surface: Arc<Surface<Window>>,
    queue: Arc<Queue>,

    gpu_future: Box<dyn GpuFuture>,

    command_buffers: RefCell<Vec<AutoCommandBufferBuilder>>,

    // main_depth_stencil_texture: Texture,
    // command_queue: CommandQueue,
    // command_buffers: RefCell<Vec<CommandBuffer>>,
    // samplers: Vec<SamplerState>,
    // shared_event: SharedEvent,
    // shared_event_listener: SharedEventListener,
    // next_timer_query_event_value: Cell<u64>,
}

impl VulkanDevice {
    #[inline]
    pub fn new(swapchain: Arc<Swapchain<Window>>,
               images: Vec<Arc<SwapchainImages<Window>>>,
               device: Arc<vulkano::device::Device>,
               queue: Arc<Queue>) -> VulkanDevice {

        let render_pass = Arc::new(vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        ).unwrap());


        VulkanDevice {
            device: device.clone(),
            surface: swapchain.surface().clone(),
            queue: queue.clone(),
            gpu_future: Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>,

            command_buffers: RefCell::new(vec![]),
        }
    }

    pub fn present_drawable(&mut self) {
        self.begin_commands();
        self.command_buffers.borrow_mut().last().unwrap().present_drawable(&self.drawable);
        self.end_commands();
        self.drawable = self.layer.next_drawable().unwrap().retain();
    }
}


fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec!(viewport));

    images.iter().map(|image| {
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .build().unwrap()
        ) as Arc<dyn FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>()
}
// ================================================================================================
// ================================= IMPL DEVICE ==================================================

struct VulkanBuffer(buffer);

struct VulkanFrameBuffer();

struct VulkanProgram{fragment: VulkanShader, vertex: VulkanShader}

struct VulkanShader{entry: Entry, other: Arc<ShaderModule>}

struct VulkanTexture(texture, size, sampler);

// texture is just
struct VulkanTextureDataReceiver();

struct VulkanTimerQuery();

struct VulkanUniform();

struct VulkanVertexArray {
    descriptor: RuntimeVertexDef,
    vertex_buffers: RefCell<Vec<VulkanBuffer>>,
    index_buffer: RefCell<Option<VulkanBuffer>>,
}

struct VulkanVertexAttribute {
    name: String,
    index: u32,
    a_type: vulkano::format::Format
}


impl Device for VulkanDevice {
    type Buffer = VulkanBuffer;
    type Framebuffer = VulkanFrameBuffer;
    type Program = VulkanProgram;
    type Shader = VulkanShader;
    type Texture = VulkanTexture;
    type TextureDataReceiver = VulkanTextureDataReceiver;
    type TimerQuery = VulkanTimerQuery;
    type Uniform = VulkanUniform;
    type VertexArray = VulkanVertexArray;
    type VertexAttr = VulkanVertexAttribute;


    // Convert Format to the Vulkan enum and create the texture for size
    fn create_texture(&self, format: TextureFormat, size: Vector2I) -> VulkanTexture {
        let format = format.to_vk_texture_format(format);

        let (texture, tex_future) =
            ImmutableImage::uninitialized(
                self.device.clone(),
                Dimensions::Dim2d { width: size.0, height: size.1 },
                format,
                MipmapsCount::One,
                ImageUsage::all(),
                ImageLayout::Undefined,
                self.queue.clone(),
            ).unwrap();

        VulkanTexture(texture, size, None)
    }

    fn create_texture_from_data(&self, format: TextureFormat, size: Vector2I, data: TextureDataRef)
                                -> VulkanTexture {
        let format = format.to_vk_texture_format(format);

        let (texture, tex_future) =
            ImmutableImage::from_buffer(
                match data {
                    TextureDataRef::U8(d) => { d.into() }
                    TextureDataRef::F16(d) => { d.into() }
                    TextureDataRef::F32(d) => { d.into() }
                },
                Dimensions::Dim2d { width: size.0, height: size.1 },
                format,
                self.queue.clone(),
            ).unwrap();

        VulkanTexture(texture, size, None)
    }

    // Going to opt to not implement this one
    fn create_shader_from_source(&self, _: &str, source: &[u8], _: ShaderKind) -> VulkanShader {
        unimplemented!();
    }

    #[inline]
    fn create_shader(
        &self,
        resources: &dyn ResourceLoader,
        name: &str,
        kind: ShaderKind,
    ) -> Self::Shader {
        let suffix = match kind {
            ShaderKind::Vertex => 'v',
            ShaderKind::Fragment => 'f',
        };

        let path = format!("shaders/gl3/{}.{}s.glsl", name, suffix);

        // TODO: =============================================== vvvvvvvvvvvv
        let compiled_shader = shade_runner::load(path, None, Self::convert_sr(kind), None)
            .expect("Shader didn't compile");

        let vulkano_entry =
            shade_runner::parse(&compiled_shader)
                .expect("failed to parse");

        VulkanShader {
            entry: vulkano_entry,
            other: unsafe {
                ShaderModule::from_words(device.clone(), &compiled_shader.spriv.clone())
            }.unwrap(),
        }
    }

    // Program is just the combination of the two shaders
    fn create_program_from_shaders(&self,
                                   _: &dyn ResourceLoader,
                                   _: &str,
                                   vertex_shader: VulkanShader,
                                   fragment_shader: VulkanShader)
                                   -> VulkanProgram {
        VulkanProgram(fragment_shader, vertex_shader)
    }

    // Returns a vertex array that will be configured by configure_vertex_attr and bound by bind_buffer
    // The opengl just returns an empty VertexBufferObject
    // The vulkan equivalent would be
    fn create_vertex_array(&self) -> VulkanVertexArray {

        // descriptor: VertexDescriptor::new().retain(),
        // vertex_buffers: RefCell::new(vec![]),
        // index_buffer: RefCell::new(None),




        VulkanVertexArray()
    }

    // returns a container which is allocated by allocate_buffer<T> and bound to the vertex array in bind_buffer
    // Does literally the same thing as create_vertex_array but for other buffers
    fn create_buffer(&self) -> VulkanBuffer {
        VulkanBuffer()
    }

    fn configure_vertex_attr(&self,
                             vertex_array: &VulkanVertexArray, // create_vertex_array
                             attr: &VulkanVertexAttribute,     // get_vertex_attribute
                             descriptor: &VertexAttrDescriptor // input
    ) {

        debug_assert_ne!(descriptor.stride, 0);

        let attribute_index = attr.index;

        let t = attr.a_type;

        // I need to store both the layout/descriptor and attributes
       // let vertex_array.descriptor.0

        



        // for attributes :
        //
        // set format
        // set offset
        // set buffer index
        // (attributes)


        // for descriptors :
        //
        // set Vertex or PerInstance
        // set step rate ( instance count? )
        // set stride
        // (layouts)

        /*

        attr: VulkanVertexAttribute {
            name: String,
            index: u32,
            a_type: Something::R16Unorm
        }

        vertex_array: VulkanVertexArray {
            vertex_buffers: RefCell<Vec<VulkanBuffer>>,
            index_buffer: RefCell<Option<VulkanBuffer>>,
        };

        descriptor: &VertexAttrDescriptor {
            pub size: usize,
            pub class: VertexAttrClass, (int float ufloat)
            pub attr_type: VertexAttrType, (f32, i16, u8, etc.)
            pub stride: usize,
            pub offset: usize,
            pub divisor: u32,
            pub buffer_index: u32,
        }

        */

        unimplemented!();
    }

    fn get_vertex_attr(&self, program: &VulkanProgram, name: &str)
        -> Option<VulkanVertexAttribute> {

        /* Look for the vertex attribute by name from the vertex shader
        and then return it's (name, type, index) */

        for i in program.vertex.entry.input.unwrap().inputs {
            if i.name == name {
                return Some(VulkanVertexAttribute{
                    name: i.name.unwrap().into_string(),
                    index: i.location.start,
                    a_type: i.format
                })
            }
        }
        None
    }


    fn bind_buffer(&self,
                   vertex_array: &VulkanVertexArray,
                   buffer: &VulkanBuffer,
                   target: BufferTarget) {

        match target {
            BufferTarget::Vertex => {
                vertex_array.vertex_buffers.borrow_mut().push((*buffer).clone())
            }
            BufferTarget::Index => {
                *vertex_array.index_buffer.borrow_mut() = Some((*buffer).clone())
            }
        }

        unimplemented!();
    }

    fn create_framebuffer(&self, texture: VulkanTexture) -> VulkanFrameBuffer {

        // images: &[Arc<SwapchainImage<Window>>],
        // render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
        // dynamic_state: &mut DynamicState,

        // Vec<Arc<dyn FramebufferAbstract + Send + Sync>>

        let dimensions = images[0].dimensions();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            depth_range: 0.0..1.0,
        };

        let mut dynamic_state = DynamicState::default();
        dynamic_state.viewports = Some(vec!(viewport));

        images.iter().map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(texture.0.clone()).unwrap()
                    .build().unwrap()
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        }).collect::<Vec<_>>();

        unimplemented!();
    }


    fn get_uniform(&self, program : &Self::Program, name: &str) -> VulkanUniform {

        let uniform_buffer = CpuBufferPool::uniform_buffer(self.device.clone());



        // let set = Arc::new(PersistentDescriptorSet::start(pipeline.clone(), 0)
        //     .add_buffer(uniform_buffer).unwrap()
        //     .build().unwrap()
        // );

        unimplemented!();
    }


    fn allocate_buffer<T>(&self,
                          buffer: &VulkanBuffer,
                          data: BufferData<T>,
                          _: BufferTarget,
                          mode: BufferUploadMode) {
        let host_cache = match mode {
            BufferUploadMode::Static => false,
            BufferUploadMode::Dynamic => true,
        };

        match data {
            BufferData::Uninitialized(size) => {
                let size = (size * mem::size_of::<T>()) as usize;
                let new_buffer = unsafe {
                    CpuAccessibleBuffer::uninitialized_array(self.device.clone(), size, BufferUsage::all(), host_cache);
                };
                *buffer.0.borrow_mut() = Some(new_buffer);
            }
            BufferData::Memory(slice) => {
                let size = (slice.len() * mem::size_of::<T>()) as u64;
                let new_buffer = CpuAccessibleBuffer::from_data(self.device.clone(), BufferUsage::all(), host_cache, slice);
                // let new_buffer = self.device.new_buffer_with_data(slice.as_ptr() as *const _,
                //                                                   size,
                //                                                   options);
                *buffer.0.borrow_mut() = Some(new_buffer);
            }
        }
        unimplemented!();
    }

    #[inline]
    fn framebuffer_texture<'f>(&self, framebuffer: &'f VulkanFrameBuffer) -> &'f VulkanTexture {
        unimplemented!();
    }

    #[inline]
    fn destroy_framebuffer(&self, framebuffer: VulkanFrameBuffer) -> VulkanTexture {
        unimplemented!();
    }

    // Just a translation method to convert VulkanTexture
    fn texture_format(&self, texture: &VulkanTexture) -> TextureFormat {
        match texture.texture.pixel_format() {
            Format::R8Unorm => TextureFormat::R8,
            Format::R16Sfloat => TextureFormat::R16F,
            Format::R8G8B8A8Unorm => TextureFormat::RGBA8,
            Format::R16G16B16A16Sfloat => TextureFormat::RGBA16F,
            Format::R32G32B32A32Sfloat => TextureFormat::RGBA32F,
            _ => panic!("Unexpected Vulkan texture format!"),
        }
    }

    fn texture_size(&self, texture: &VulkanTexture) -> Vector2I {
        texture.1
    }

    // This appears to set 4 teture parameters ( all for target TEXTURE_2D ) :
    //      TEXTURE_MIN_FILTER
    //      TEXTURE_MAG_FILTER
    //      TEXTURE_WRAP_S
    //      TEXTURE_WRAP_T

    fn set_texture_sampling_mode(&self, texture: &VulkanTexture, flags: TextureSamplingFlags) {

        // get the sampler
        let s = Sampler::new(
            self.device.clone(),
            if flags.contains(TextureSamplingFlags::NEAREST_MAG) {
                Filter::Nearest
            } else {
                Filter::Linear
            },
            if flags.contains(TextureSamplingFlags::NEAREST_MIN) {
                Filter::Nearest
            } else {
                Filter::Linear
            },
            MipmapMode::Nearest, // mipmap mode
            if flags.contains(TextureSamplingFlags::REPEAT_U) {
                SamplerAddressMode::Repeat
            } else {
                SamplerAddressMode::ClampToEdge
            },
            if flags.contains(TextureSamplingFlags::REPEAT_V) {
                SamplerAddressMode::Repeat
            } else {
                SamplerAddressMode::ClampToEdge
            },
            SamplerAddressMode::Repeat, // address w
            0.0, // mip lod bias
            1.0, // max anisotrophy
            0.0, // min lod
            1.0, // max lod
        ).unwrap();

        // add the sampler to the texture representation (maybe not use tuple struct)
        texture.2 = Some(s);

        unimplemented!();
    }

    fn upload_to_texture(&self, texture: &VulkanTexture, rect: RectI, data: TextureDataRef) {

        // So this looks like it's going to have to first upload this data to an image, and then
        // upload that image data to another via VkImageBlit
        //
        // OR it looks like I can also just allocate the image, and then add copy_image onto the
        // command buffer

        assert!(rect.size().x() >= 0);
        assert!(rect.size().y() >= 0);
        assert!(rect.max_x() <= texture.size.x());
        assert!(rect.max_y() <= texture.size.y());

        unimplemented!();
    }

    fn read_pixels(&self, target: &RenderTarget<VulkanDevice>, viewport: RectI)
                   -> VulkanTextureDataReceiver {
        // In the Metal code this is getting the render target from : render_target_color_texture
        // It then 'retains' the texture and puts it into a MetalTextureDataReceiverInfo
        // This info block is then put into a MetalTextureDataReceiver

        // So in VK we should probably allocate and vkCmdCopyImage on the command buffer
        //
        unimplemented!();
    }

    fn begin_commands(&self) {
        self.command_buffers.borrow_mut().push(
            AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), self.queue.family())
                .unwrap()
        );
    }

    fn end_commands(&self) {
        self.command_buffers.borrow_mut().last().unwrap().build().unwrap();

        if recreate_swapchain {
            let dimensions = if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
                [dimensions.0, dimensions.1]
            } else {
                return;
            };
            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => return,
                Err(err) => panic!("{:?}", err)
            };
            swapchain = new_swapchain;
            framebuffers = window_size_dependent_setup(&new_images, render_pass.clone(), &mut dynamic_state);
            recreate_swapchain = false;
        }

        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                return;
            }
            Err(err) => panic!("{:?}", err)
        };

        self.gpu_future.cleanup_finished();

        self.gpu_future.join(acquire_future)
            .then_execute(queue.clone(), command_buffer).unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();


        match future {
            Ok(future) => {
                previous_frame_end = Box::new(future) as Box<_>;
            }
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
            }
        }


        // let future = future
        //     .then_execute(self.queue.clone(), command_buffer).unwrap()
        //     .then_swapchain_present(self.queue.clone(), self.swapchain.clone().unwrap().clone(), image_num)
        //     .then_signal_fence_and_flush();
    }


    fn draw_arrays(&self, index_count: u32, render_state: &RenderState<VulkanDevice>) {

        //self.set_uniform(render_state.program, )

        // create the runtime vertex definitions from the render state

        let pipeline = Arc::new(GraphicsPipeline::start()
            //.vertex_input(SingleBufferDefinition::<V>::new())
            .vertex_input_single_buffer()

            .vertex_shader((render_state.program.1).0.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader((render_state.program.0).0.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap());

        let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None, compare_mask: None, write_mask: None, reference: None };

        let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);
        let mut recreate_swapchain = false;


        // pub fn get_descriptor_set(&self,
        //                           pipeline: Arc<dyn GraphicsPipelineAbstract + Sync + Send>,
        //                           sampler: Arc<Sampler>) -> Box<dyn DescriptorSet + Send + Sync> {
        //     let o: Box<dyn DescriptorSet + Send + Sync> = Box::new(
        let d = PersistentDescriptorSet::start(
            (pipeline as Arc<dyn GraphicsPipelineAbstract + Sync + Send>).clone()
                )
                    .add_sampled_image(self.buffer.clone(), sampler.clone()).unwrap()
                    .build().unwrap();
        //    o
        //}

        // The metal code does a 'prepare to draw' op here.
        //      Creates the descriptor + attachment buffers
        //      They do some viewport malarchy, so the dynamic state needs to be in here
        //      Sets up the pipeline, depth stencil, etc. (reflection?)
        //      Also uniforms

        // From this prepare to draw call it looks like it gets the command buffer?

        unimplemented!();
    }

    fn draw_elements(&self, index_count: u32, render_state: &RenderState<VulkanDevice>) {
        /* Render State
            target: &self.draw_render_target(),
            program: &self.filter_text_program.program,
            vertex_array: &self.filter_text_vertex_array.vertex_array,
            primitive: Primitive::Triangles,
            textures: &[&source_texture, &self.gamma_lut_texture],
            uniforms: &uniforms,
            viewport: main_viewport,
            options: RenderOptions {
            clear_ops: ClearOps { color: clear_color, ..ClearOps::default() },
            ..RenderOptions::default()
        */

        // This does the same prepare to draw as above
        // it then calls command_buffer.draw_indexed()

        // let c : Some(= AutoCommandBufferBuilder::new();

        unimplemented!();
    }

    fn draw_elements_instanced(&self,
                               index_count: u32,
                               instance_count: u32,
                               render_state: &RenderState<VulkanDevice>) {

        // This does the same prepare to draw as above
        // it then calls draw_indexed with an instance description
        unimplemented!();
    }

    // This has something to do with creating the frame query
    fn create_timer_query(&self) -> VulkanTimerQuery {
        unimplemented!();
    }

    fn begin_timer_query(&self, query: &VulkanTimerQuery) {
        unimplemented!();
    }

    fn end_timer_query(&self, query: &VulkanTimerQuery) {
        unimplemented!();
    }

    fn try_recv_timer_query(&self, query: &VulkanTimerQuery) -> Option<Duration> {
        unimplemented!();
    }


    fn recv_timer_query(&self, query: &VulkanTimerQuery) -> Duration {
        unimplemented!();
    }


    // This will take the texture data [probably a future] and export it out as a vec of it's primitive type
    fn try_recv_texture_data(&self, receiver: &VulkanTextureDataReceiver) -> Option<TextureData> {

        // let  receiver.try_unwrap()
        unimplemented!();
    }

    // forces the pull of texture data, assering if it cant
    fn recv_texture_data(&self, receiver: &VulkanTextureDataReceiver) -> TextureData {
        unimplemented!();
    }
}

// ===============================================================================================
// ======================================== IMPL VULKAN_DEVICE ===================================

impl VulkanDevice {
    fn get_uniform_index(&self, shader: &VulkanShader, name: &str) -> Option<VulkanUniformIndex> {
        unimplemented!();
    }

    fn populate_uniform_indices_if_necessary(&self,
                                             uniform: &VulkanUniform,
                                             program: &VulkanProgram) {
        unimplemented!();
    }

    fn render_target_color_texture(&self, render_target: &RenderTarget<VulkanDevice>)
                                   -> Texture {
        unimplemented!();
    }

    fn render_target_depth_texture(&self, render_target: &RenderTarget<VulkanDevice>)
                                   -> Option<Texture> {
        unimplemented!();
    }

    fn render_target_has_depth(&self, render_target: &RenderTarget<VulkanDevice>) -> bool {
        unimplemented!();
    }

    //
    fn prepare_to_draw(&self, render_state: &RenderState<VulkanDevice>) -> AutoCommandBufferBuilder {

        // create a new command buffer builder
        let mut command_buffer = command_buffer.begin_render_pass(
            framebuffers[image_num].clone(), false, clear_values.clone(),
        ).unwrap();

        command_buffer = command_buffer.draw(
            shader.get_pipeline().clone(),
            // Multiple vertex buffers must have their definition in the pipeline!
            &self.dynamic_state.clone(), vec![vertex_buffer],
            vec![descriptor_set], (),
        ).unwrap();
    }

    fn populate_shader_uniforms_if_necessary(&self,
                                             shader: &VulkanShader,
                                             reflection: &RenderPipelineReflectionRef) {
        unimplemented!();
    }

    fn create_argument_buffer(&self, shader: &VulkanShader) -> Option<Buffer> {
        unimplemented!();
    }

    fn set_uniform(&self,
                   argument_index: VulkanUniformIndex,
                   argument_encoder: &ArgumentEncoder,
                   uniform_data: &UniformData,
                   buffer: &Buffer,
                   buffer_offset: u64,
                   render_command_encoder: &RenderCommandEncoderRef,
                   render_state: &RenderState<VulkanDevice>) {
        unimplemented!();
    }

    fn prepare_pipeline_color_attachment_for_render(
        &self,
        pipeline_color_attachment: &RenderPipelineColorAttachmentDescriptorRef,
        render_state: &RenderState<VulkanDevice>) {
        unimplemented!();
    }

    //
    fn create_render_pass_descriptor(&self, render_state: &RenderState<VulkanDevice>)
                                     -> RenderPassDescriptor {
        unimplemented!();
    }

    fn set_depth_stencil_state(&self,
                               encoder: &RenderCommandEncoderRef,
                               render_state: &RenderState<VulkanDevice>) {
        unimplemented!();
    }

    fn texture_format(&self, texture: &Texture) -> Option<TextureFormat> {
        unimplemented!();
    }

    fn set_viewport(&self, encoder: &RenderCommandEncoderRef, viewport: &RectI) {
        unimplemented!();
    }

    fn synchronize_texture(&self, texture: &Texture, block: RcBlock<(*mut Object, ), ()>) {
        unimplemented!();
    }
}

// ==================================================================================================
// ========================================    DATATYPES    =========================================


// ==================================================================================================
// =============================== HELPERS & CONVERSIONS ============================================

trait ShaderKindExt {
    fn to_shade_runner_shader_kind(kind: ShaderKind) -> shade_runner::ShaderKind;
}

impl ShaderKindExt for ShaderKind {
    fn to_shade_runner_shader_kind(kind: ShaderKind) -> shade_runner::ShaderKind {
        match kind {
            ShaderKind::Vertex => { shade_runner::ShaderKind::Vertex }
            ShaderKind::Fragment => { shade_runner::ShaderKind::Fragment }
        }
    }
}

trait TextureFormatExt {
    fn to_vk_texture_format(kind: TextureFormat) -> vulkano::format::Format;
}

impl TextureFormatExt for TextureFormat {
    fn to_vk_texture_format(format: TextureFormat) -> vulkano::format::Format {
        match format {
            TextureFormat::R8 => Format::R8Unorm,
            TextureFormat::R16F => Format::R16Sfloat,
            TextureFormat::RGBA8 => Format::R8G8B8A8Unorm,
            TextureFormat::RGBA16F => Format::R16G16B16A16Sfloat,
            TextureFormat::RGBA32F => Format::R32G32B32A32Sfloat,
        }
    }
}

trait DeviceExtra {
    fn create_depth_stencil_texture(&self, size: Vector2I) -> Texture;
}

impl DeviceExtra for vulkano::Device {
    fn create_depth_stencil_texture(&self, size: Vector2I) -> Texture {
        unimplemented!();
    }
}

// trait BlendFactorExt {
//     fn to_metal_blend_factor(self) -> MTLBlendFactor;
// }
//
// impl BlendFactorExt for BlendFactor {
//     #[inline]
//     fn to_metal_blend_factor(self) -> MTLBlendFactor {
//         unimplemented!();
//         // match self {
//         //     BlendFactor::Zero => MTLBlendFactor::Zero,
//         //     BlendFactor::One => MTLBlendFactor::One,
//         //     BlendFactor::SrcAlpha => MTLBlendFactor::SourceAlpha,
//         //     BlendFactor::OneMinusSrcAlpha => MTLBlendFactor::OneMinusSourceAlpha,
//         //     BlendFactor::DestAlpha => MTLBlendFactor::DestinationAlpha,
//         //     BlendFactor::OneMinusDestAlpha => MTLBlendFactor::OneMinusDestinationAlpha,
//         //     BlendFactor::DestColor => MTLBlendFactor::DestinationColor,
//         // }
//     }
// }
//
// trait BlendOpExt {
//     fn to_metal_blend_op(self) -> MTLBlendOperation;
// }
//
// impl BlendOpExt for BlendOp {
//     #[inline]
//     fn to_metal_blend_op(self) -> MTLBlendOperation {
//         unimplemented!();
//         // match self {
//         //     BlendOp::Add => MTLBlendOperation::Add,
//         //     BlendOp::Subtract => MTLBlendOperation::Subtract,
//         //     BlendOp::ReverseSubtract => MTLBlendOperation::ReverseSubtract,
//         //     BlendOp::Min => MTLBlendOperation::Min,
//         //     BlendOp::Max => MTLBlendOperation::Max,
//         // }
//     }
// }
//
// trait DepthFuncExt {
//     fn to_metal_compare_function(self) -> MTLCompareFunction;
// }
//
// impl DepthFuncExt for DepthFunc {
//     fn to_metal_compare_function(self) -> MTLCompareFunction {
//         unimplemented!();
//         //match self {
//         //    DepthFunc::Less => MTLCompareFunction::Less,
//         //    DepthFunc::Always => MTLCompareFunction::Always,
//         //}
//     }
// }
//
// trait PrimitiveExt {
//     fn to_metal_primitive(self) -> MTLPrimitiveType;
// }
//
// impl PrimitiveExt for Primitive {
//     fn to_metal_primitive(self) -> MTLPrimitiveType {
//         unimplemented!();
//         // match self {
//         //     Primitive::Triangles => MTLPrimitiveType::Triangle,
//         //     Primitive::Lines => MTLPrimitiveType::Line,
//         // }
//     }
// }
//
// trait StencilFuncExt {
//     fn to_metal_compare_function(self) -> MTLCompareFunction;
// }
//
// impl StencilFuncExt for StencilFunc {
//     fn to_metal_compare_function(self) -> MTLCompareFunction {
//         unimplemented!();
//         // match self {
//         //     StencilFunc::Always => MTLCompareFunction::Always,
//         //     StencilFunc::Equal => MTLCompareFunction::Equal,
//         // }
//     }
// }

trait UniformDataExt {
    fn as_bytes(&self) -> Option<&[u8]>;
}

impl UniformDataExt for UniformData {
    fn as_bytes(&self) -> Option<&[u8]> {
        unimplemented!();
        // unsafe {
        //     match *self {
        //         UniformData::TextureUnit(_) => None,
        //         UniformData::Float(ref data) => {
        //             Some(slice::from_raw_parts(data as *const f32 as *const u8, 4 * 1))
        //         }
        //         UniformData::IVec3(ref data) => {
        //             Some(slice::from_raw_parts(data as *const i32 as *const u8, 4 * 3))
        //         }
        //         UniformData::Int(ref data) => {
        //             Some(slice::from_raw_parts(data as *const i32 as *const u8, 4 * 1))
        //         }
        //         UniformData::Mat2(ref data) => {
        //             Some(slice::from_raw_parts(data as *const F32x4 as *const u8, 4 * 4))
        //         }
        //         UniformData::Mat4(ref data) => {
        //             Some(slice::from_raw_parts(&data[0] as *const F32x4 as *const u8, 4 * 16))
        //         }
        //         UniformData::Vec2(ref data) => {
        //             Some(slice::from_raw_parts(data as *const F32x2 as *const u8, 4 * 2))
        //         }
        //         UniformData::Vec3(ref data) => {
        //             Some(slice::from_raw_parts(data as *const f32 as *const u8, 4 * 3))
        //         }
        //         UniformData::Vec4(ref data) => {
        //             Some(slice::from_raw_parts(data as *const F32x4 as *const u8, 4 * 4))
        //         }
        //     }
        // }
    }
}

// trait VertexAttrTypeExt {
//     fn to_vk_type(self) -> GLuint;
// }
//
// impl VertexAttrTypeExt for VertexAttrType {
//     fn to_vk_type(self) -> GLuint {
//         match self {
//             VertexAttrType::F32 => vulkano::,
//             VertexAttrType::I16 => gl::SHORT,
//             VertexAttrType::I8  => gl::BYTE,
//             VertexAttrType::U16 => gl::UNSIGNED_SHORT,
//             VertexAttrType::U8  => gl::UNSIGNED_BYTE,
//         }
//     }
// }
//
// trait TextureFormatExt: Sized {
//     fn from_metal_pixel_format(metal_pixel_format: MTLPixelFormat) -> Option<Self>;
// }
//
// impl TextureFormatExt for TextureFormat {
//     fn from_metal_pixel_format(metal_pixel_format: MTLPixelFormat) -> Option<TextureFormat> {
//         unimplemented!();
//         // match metal_pixel_format {
//         //     MTLPixelFormat::R8Unorm => Some(TextureFormat::R8),
//         //     MTLPixelFormat::R16Float => Some(TextureFormat::R16F),
//         //     MTLPixelFormat::RGBA8Unorm => Some(TextureFormat::RGBA8),
//         //     MTLPixelFormat::BGRA8Unorm => {
//         //         // FIXME(pcwalton): This is wrong! But it prevents a crash for now.
//         //         Some(TextureFormat::RGBA8)
//         //     }
//         //     _ => None,
//         // }
//     }
// }

// Synchronization helpers

fn try_recv_timer_query_with_guard(guard: &mut MutexGuard<VulkanTimerQueryData>)
                                   -> Option<Duration> {
    unimplemented!();
    // match (guard.start_time, guard.end_time) {
    //     (Some(start_time), Some(end_time)) => Some(end_time - start_time),
    //     _ => None,
    // }
}

/// Download Texture into pixels
impl VulkanTextureDataReceiver {
    fn download(&self) {
        unimplemented!();
    }
}

fn try_recv_texture_data_with_guard(guard: &mut MutexGuard<VulkanTextureDataReceiverState>)
                                    -> Option<TextureData> {
    unimplemented!();
}


