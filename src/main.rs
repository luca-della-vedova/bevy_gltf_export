use bevy::prelude::*;
use bevy::render::mesh::Indices;
use bevy::render::render_resource::{Extent3d, PrimitiveTopology, TextureDimension, TextureFormat};
use gltf_json as json;

use std::{fs, mem};

use json::validation::Checked::Valid;
use std::borrow::Cow;
use std::io::Write;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Output {
    /// Output standard glTF.
    Standard,

    /// Output binary glTF.
    Binary,
}

#[derive(Debug)]
struct BuffersWrapper {
    buffer: json::Buffer,
    buffer_views: Vec<json::buffer::View>,
    accessors: Vec<json::Accessor>,
    data: Vec<u8>,
}

impl BuffersWrapper {
    fn new(output: &Output) -> Self {
        Self {
            buffer: json::Buffer {
                byte_length: 0,
                extensions: Default::default(),
                extras: Default::default(),
                name: None,
                uri: if *output == Output::Standard {
                    Some("buffer0.bin".into())
                } else {
                    None
                },
            },
            buffer_views: vec![],
            accessors: vec![],
            data: vec![],
        }
    }

    fn push_buffer<T>(&mut self, data: Vec<T>) -> u32 {
        let last_view_idx = self.buffer_views.len() as u32;
        let bytes = to_padded_byte_vector(data);
        self.buffer_views.push(json::buffer::View {
            buffer: json::Index::new(0),
            byte_length: bytes.len() as u32,
            byte_offset: Some(self.buffer.byte_length),
            byte_stride: Some(mem::size_of::<T>() as u32),
            extensions: Default::default(),
            extras: Default::default(),
            name: None,
            target: Some(Valid(json::buffer::Target::ArrayBuffer)),
        });
        self.data.extend(bytes);
        self.buffer.byte_length = self.data.len() as u32;
        last_view_idx
    }

    fn push_accessor(&mut self, accessor: json::Accessor) -> json::Index<json::Accessor> {
        let last_accessor_idx = self.accessors.len() as u32;
        self.accessors.push(accessor);
        json::Index::<json::Accessor>::new(last_accessor_idx)
    }

    fn build(
        self,
    ) -> (
        Vec<json::Buffer>,
        Vec<json::buffer::View>,
        Vec<json::Accessor>,
        Vec<u8>,
    ) {
        (
            vec![self.buffer],
            self.buffer_views,
            self.accessors,
            self.data,
        )
    }
}

fn align_to_multiple_of_four(n: &mut u32) {
    *n = (*n + 3) & !3;
}

fn to_padded_byte_vector<T>(vec: Vec<T>) -> Vec<u8> {
    let byte_length = vec.len() * mem::size_of::<T>();
    let byte_capacity = vec.capacity() * mem::size_of::<T>();
    let alloc = vec.into_boxed_slice();
    let ptr = Box::<[T]>::into_raw(alloc) as *mut u8;
    // TODO(luca) can we get rid of the unsafe here?
    let mut new_vec = unsafe { Vec::from_raw_parts(ptr, byte_length, byte_capacity) };
    while new_vec.len() % 4 != 0 {
        new_vec.push(0); // pad to multiple of four bytes
    }
    new_vec
}

fn image_getter(id: &Handle<Image>) -> Option<Image> {
    let extent = Extent3d {
        width: 512,
        height: 512,
        ..Default::default()
    };
    Some(Image::new_fill(
        extent,
        TextureDimension::D2,
        &[0, 255, 0, 0],
        TextureFormat::Rgba8Unorm,
    ))
}

fn to_gltf_material(
    buffers: &mut BuffersWrapper,
    mat: &StandardMaterial,
) -> (json::Material, Vec<json::Texture>, Vec<json::Image>) {
    let mut textures = vec![];
    let mut images = vec![];
    // TODO(luca) implement samplers
    if let Some(base_color_texture) = &mat.base_color_texture {
        let image = image_getter(base_color_texture).unwrap();
        let texture = json::Index::<json::Image>::new(0);
        /*
        let buffer_view_index = buffer_views.len() as u32;
        buffer_views.push(json::buffer::View {
            buffer: json::Index::new(0),
            byte_length: buffer_length,
            byte_offset: Some(buffer.byte_length),
            byte_stride: Some(mem::size_of::<u32>() as u32),
            extensions: Default::default(),
            extras: Default::default(),
            name: None,
            target: Some(Valid(json::buffer::Target::ArrayBuffer)),
        });
        images.push(json::Image {
            data: Some(image.data),
            //mime_type: Some(),
            ..Default::default()
        });
        textures.push(texture);
        */
    }
    (
        json::Material {
            pbr_metallic_roughness: json::material::PbrMetallicRoughness {
                base_color_factor: json::material::PbrBaseColorFactor(mat.base_color.as_rgba_f32()),
                // TODO(luca) other properties here
                ..Default::default()
            },
            ..Default::default()
        },
        textures,
        images,
    )
}

fn get_indices_data(
    buffers: &mut BuffersWrapper,
    mesh: &Mesh,
) -> Option<json::Index<json::Accessor>> {
    let Some(mesh_indices) = mesh.indices() else {
        return None;
    };
    let count = mesh_indices.len() as u32;
    let indices = mesh_indices.iter().map(|idx| idx as u32).collect();
    let view_idx = buffers.push_buffer(indices);
    // TODO(luca) Indexes can also be u8 / u16
    let accessor_idx = buffers.push_accessor(json::Accessor {
        buffer_view: Some(json::Index::new(view_idx)),
        byte_offset: None,
        count,
        component_type: Valid(json::accessor::GenericComponentType(
            json::accessor::ComponentType::U32,
        )),
        extensions: Default::default(),
        extras: Default::default(),
        type_: Valid(json::accessor::Type::Scalar),
        min: None,
        max: None,
        name: None,
        normalized: false,
        sparse: None,
    });
    Some(accessor_idx)
}

fn create_bevy_sample_mesh() -> (Mesh, StandardMaterial) {
    // Create a new mesh using a triangle list topology, where each set of 3 vertices composes a triangle.
    (
        Mesh::new(PrimitiveTopology::TriangleList)
            // Add 4 vertices, each with its own position attribute (coordinate in
            // 3D space), for each of the corners of the parallelogram.
            .with_inserted_attribute(
                Mesh::ATTRIBUTE_POSITION,
                vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 0.0],
                    [2.0, 2.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
            )
            // Assign a UV coordinate to each vertex.
            .with_inserted_attribute(
                Mesh::ATTRIBUTE_UV_0,
                vec![[0.0, 1.0], [0.5, 0.0], [1.0, 0.0], [0.5, 1.0]],
            )
            // Assign normals (everything points outwards)
            .with_inserted_attribute(
                Mesh::ATTRIBUTE_NORMAL,
                vec![
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
            )
            // After defining all the vertices and their attributes, build each triangle using the
            // indices of the vertices that make it up in a counter-clockwise order.
            .with_indices(Some(Indices::U32(vec![
                // First triangle
                0, 3, 1, // Second triangle
                1, 3, 2,
            ]))),
        StandardMaterial {
            base_color: Color::rgba(1.0, 0.0, 0.0, 1.0),
            base_color_texture: Some(Handle::default()),
            ..Default::default()
        },
    )
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

fn export_mesh(output: Output) -> Result<(), ()> {
    let (mesh, material) = create_bevy_sample_mesh();
    let mut buffers = BuffersWrapper::new(&output);

    let Some(bevy::render::mesh::VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        return Err(());
    };
    let Some(bevy::render::mesh::VertexAttributeValues::Float32x3(normals)) =
        mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
    else {
        return Err(());
    };

    let mut vertices = positions
        .iter()
        .zip(normals.iter())
        .map(|(p, n)| Vertex {
            position: *p,
            normal: *n,
            uv: Default::default(),
        })
        .collect::<Vec<_>>();

    if let Some(bevy::render::mesh::VertexAttributeValues::Float32x2(uvs)) =
        mesh.attribute(Mesh::ATTRIBUTE_UV_0)
    {
        for (idx, uv) in uvs.iter().enumerate() {
            vertices[idx].uv = *uv;
        }
    }

    // Find min and max for bounding box
    let mut min = [f32::MAX, f32::MAX, f32::MAX];
    let mut max = [f32::MIN, f32::MIN, f32::MIN];

    for p in positions.iter() {
        for i in 0..3 {
            min[i] = f32::min(min[i], p[i]);
            max[i] = f32::max(max[i], p[i]);
        }
    }

    println!("Positions len in {}", vertices.len());
    let num_vertices = vertices.len() as u32;
    let view_idx = buffers.push_buffer(vertices);
    let positions = json::Accessor {
        buffer_view: Some(json::Index::new(view_idx)),
        byte_offset: Some(0),
        count: num_vertices,
        component_type: Valid(json::accessor::GenericComponentType(
            json::accessor::ComponentType::F32,
        )),
        extensions: Default::default(),
        extras: Default::default(),
        type_: Valid(json::accessor::Type::Vec3),
        min: Some(json::Value::from(Vec::from(min))),
        max: Some(json::Value::from(Vec::from(max))),
        name: None,
        normalized: false,
        sparse: None,
    };
    let normals = json::Accessor {
        buffer_view: Some(json::Index::new(view_idx)),
        byte_offset: Some((3 * mem::size_of::<f32>()) as u32),
        count: num_vertices,
        component_type: Valid(json::accessor::GenericComponentType(
            json::accessor::ComponentType::F32,
        )),
        extensions: Default::default(),
        extras: Default::default(),
        type_: Valid(json::accessor::Type::Vec3),
        min: None,
        max: None,
        name: None,
        normalized: false,
        sparse: None,
    };
    let uvs = json::Accessor {
        buffer_view: Some(json::Index::new(view_idx)),
        byte_offset: Some((6 * mem::size_of::<f32>()) as u32),
        count: num_vertices,
        component_type: Valid(json::accessor::GenericComponentType(
            json::accessor::ComponentType::F32,
        )),
        extensions: Default::default(),
        extras: Default::default(),
        type_: Valid(json::accessor::Type::Vec2),
        min: None,
        max: None,
        name: None,
        normalized: false,
        sparse: None,
    };
    let positions = buffers.push_accessor(positions);
    let normals = buffers.push_accessor(normals);
    let uvs = buffers.push_accessor(uvs);
    let indices = get_indices_data(&mut buffers, &mesh);
    let (material, textures, images) = to_gltf_material(&mut buffers, &material);

    let primitive = json::mesh::Primitive {
        attributes: {
            let mut map = std::collections::BTreeMap::new();
            map.insert(Valid(json::mesh::Semantic::Positions), positions);
            map.insert(Valid(json::mesh::Semantic::Normals), normals);
            map.insert(Valid(json::mesh::Semantic::TexCoords(0)), uvs);
            map
        },
        extensions: Default::default(),
        extras: Default::default(),
        indices,
        material: Some(json::Index::<json::Material>::new(0)),
        mode: Valid(json::mesh::Mode::Triangles),
        targets: None,
    };

    let mesh = json::Mesh {
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        primitives: vec![primitive],
        weights: None,
    };

    let node = json::Node {
        camera: None,
        children: None,
        extensions: Default::default(),
        extras: Default::default(),
        matrix: None,
        mesh: Some(json::Index::new(0)),
        name: None,
        rotation: None,
        scale: None,
        translation: None,
        skin: None,
        weights: None,
    };
    let (buffers, buffer_views, accessors, buffer_bytes) = buffers.build();

    let root = json::Root {
        accessors,
        buffers,
        buffer_views,
        meshes: vec![mesh],
        nodes: vec![node],
        scenes: vec![json::Scene {
            extensions: Default::default(),
            extras: Default::default(),
            name: None,
            nodes: vec![json::Index::new(0)],
        }],
        materials: vec![material],
        textures,
        images,
        ..Default::default()
    };

    match output {
        Output::Standard => {
            let _ = fs::create_dir("triangle");

            let writer = fs::File::create("triangle/triangle.gltf").expect("I/O error");
            json::serialize::to_writer_pretty(writer, &root).expect("Serialization error");

            let mut writer = fs::File::create("triangle/buffer0.bin").expect("I/O error");
            writer.write_all(&buffer_bytes).expect("I/O error");
        }
        Output::Binary => {
            let json_string = json::serialize::to_string(&root).expect("Serialization error");
            let mut json_offset = json_string.len() as u32;
            let buf_length = buffer_bytes.len() as u32;
            align_to_multiple_of_four(&mut json_offset);
            let glb = gltf::binary::Glb {
                header: gltf::binary::Header {
                    magic: *b"glTF",
                    version: 2,
                    length: json_offset + buf_length,
                },
                bin: Some(Cow::Owned(buffer_bytes)),
                json: Cow::Owned(json_string.into_bytes()),
            };
            let writer = std::fs::File::create("triangle.glb").expect("I/O error");
            glb.to_writer(writer).expect("glTF binary output error");
        }
    }

    Ok(())
}

fn main() {
    println!("Hello, world!");
    export_mesh(Output::Binary).unwrap();
}
