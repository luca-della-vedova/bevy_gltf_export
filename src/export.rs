use bevy_asset::Handle;
use bevy_pbr::StandardMaterial;
use bevy_render::prelude::*;
use gltf_json as json;

use std::mem;

use json::validation::Checked::Valid;
use std::borrow::Cow;

#[derive(Debug)]
struct BuffersWrapper {
    buffer: json::Buffer,
    buffer_views: Vec<json::buffer::View>,
    accessors: Vec<json::Accessor>,
    data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub enum MeshExportError {
    MissingVertexPosition,
    MissingVertexNormal,
    TextureNotFound,
    ImageConversionFailed,
    SerializationError,
}

impl BuffersWrapper {
    fn new() -> Self {
        Self {
            buffer: json::Buffer {
                byte_length: 0,
                extensions: Default::default(),
                extras: Default::default(),
                name: None,
                uri: None,
            },
            buffer_views: vec![],
            accessors: vec![],
            data: vec![],
        }
    }

    fn push_buffer<T>(&mut self, data: Vec<T>) -> json::Index<json::buffer::View> {
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
        json::Index::new((self.buffer_views.len() as u32) - 1)
    }

    fn push_accessor(&mut self, accessor: json::Accessor) -> json::Index<json::Accessor> {
        self.accessors.push(accessor);
        json::Index::<json::Accessor>::new(self.accessors.len() as u32 - 1)
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

fn to_gltf_material(
    buffers: &mut BuffersWrapper,
    mat: &StandardMaterial,
    image_getter: fn(&Handle<Image>) -> Option<Image>,
) -> Result<(json::Material, Vec<json::Texture>, Vec<json::Image>), MeshExportError> {
    let mut textures = vec![];
    let mut images = vec![];
    let mut material = json::Material {
        pbr_metallic_roughness: json::material::PbrMetallicRoughness {
            base_color_factor: json::material::PbrBaseColorFactor(mat.base_color.as_rgba_f32()),
            // TODO(luca) other properties here
            ..Default::default()
        },
        ..Default::default()
    };

    // TODO(luca) implement samplers and other types of textures
    if let Some(base_color_texture) = &mat.base_color_texture {
        let image = image_getter(base_color_texture).ok_or(MeshExportError::TextureNotFound)?;
        let texture_idx = json::Index::<json::Image>::new(0);
        let image_size = image.size();
        let image_buffer: image::ImageBuffer<image::Rgba<_>, _> =
            image::ImageBuffer::from_raw(image_size[0] as u32, image_size[1] as u32, image.data)
                .ok_or(MeshExportError::ImageConversionFailed)?;
        let mut bytes: Vec<u8> = Vec::new();
        image_buffer
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageOutputFormat::Png,
            )
            .or(Err(MeshExportError::ImageConversionFailed))?;

        let view_idx = buffers.push_buffer(bytes);
        images.push(json::Image {
            buffer_view: Some(view_idx),
            mime_type: None,
            uri: None,
            extensions: None,
            name: None,
            extras: Default::default(),
        });
        textures.push(json::Texture {
            sampler: None,
            source: texture_idx,
            extensions: None,
            name: None,
            extras: Default::default(),
        });
        material.pbr_metallic_roughness.base_color_texture = Some(json::texture::Info {
            index: json::Index::new(0),
            tex_coord: 0,
            extensions: None,
            extras: Default::default(),
        });
    }
    Ok((material, textures, images))
}

struct VerticesData {
    positions: json::Index<json::Accessor>,
    normals: json::Index<json::Accessor>,
    uvs: json::Index<json::Accessor>,
    indices: Option<json::Index<json::Accessor>>,
}

fn get_vertices_data(
    buffers: &mut BuffersWrapper,
    mesh: &Mesh,
) -> Result<VerticesData, MeshExportError> {
    let Some(bevy_render::mesh::VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        return Err(MeshExportError::MissingVertexPosition);
    };
    let Some(bevy_render::mesh::VertexAttributeValues::Float32x3(normals)) =
        mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
    else {
        return Err(MeshExportError::MissingVertexNormal);
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

    if let Some(bevy_render::mesh::VertexAttributeValues::Float32x2(uvs)) =
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

    let num_vertices = vertices.len() as u32;
    let view_idx = buffers.push_buffer(vertices);
    let positions = json::Accessor {
        buffer_view: Some(view_idx),
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
        buffer_view: Some(view_idx),
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
        buffer_view: Some(view_idx),
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
    let indices = get_indices_data(buffers, mesh);
    Ok(VerticesData {
        positions,
        normals,
        uvs,
        indices,
    })
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
        buffer_view: Some(view_idx),
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

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

pub fn export_mesh(
    mesh: Mesh,
    material: StandardMaterial,
    image_getter: fn(&Handle<Image>) -> Option<Image>,
) -> Result<Vec<u8>, MeshExportError> {
    let mut buffers = BuffersWrapper::new();
    let vertices_data = get_vertices_data(&mut buffers, &mesh)?;

    let (material, textures, images) = to_gltf_material(&mut buffers, &material, image_getter)?;

    let primitive = json::mesh::Primitive {
        attributes: {
            let mut map = std::collections::BTreeMap::new();
            map.insert(
                Valid(json::mesh::Semantic::Positions),
                vertices_data.positions,
            );
            map.insert(Valid(json::mesh::Semantic::Normals), vertices_data.normals);
            map.insert(Valid(json::mesh::Semantic::TexCoords(0)), vertices_data.uvs);
            map
        },
        extensions: Default::default(),
        extras: Default::default(),
        indices: vertices_data.indices,
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

    let json_string =
        json::serialize::to_string(&root).or(Err(MeshExportError::SerializationError))?;
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
    glb.to_vec().or(Err(MeshExportError::SerializationError))
}
