use bevy_asset::Handle;
use bevy_pbr::StandardMaterial;
use bevy_render::prelude::*;
use gltf_json as json;

use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::mem;

use json::validation::Checked::Valid;
use std::borrow::Cow;
use thiserror::Error;

#[derive(Debug, Clone)]
struct HashableAccessor(json::Accessor);

impl PartialEq for HashableAccessor {
    fn eq(&self, other: &Self) -> bool {
        self.0.buffer_view == other.0.buffer_view &&
        self.0.byte_offset == other.0.byte_offset &&
        self.0.count == other.0.count &&
        // TODO(luca) other components
        //self.0.component_type == other.0.component_type &&
        self.0.type_ == other.0.type_
    }
}

impl Eq for HashableAccessor {}

impl Hash for HashableAccessor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.buffer_view.hash(state);
        self.0.byte_offset.hash(state);
        self.0.count.hash(state);
        if let Valid(t) = self.0.component_type.as_ref() {
            t.0.as_gl_enum().hash(state);
        }
        if let Valid(t) = self.0.type_.as_ref() {
            t.multiplicity().hash(state);
        }
        // TODO(luca) we don't use other fields for now
        // TODO(luca) we can't hash min and max because they are f64
    }
}

#[derive(Debug)]
struct BuffersWrapper {
    buffer: json::Buffer,
    buffer_views: Vec<json::buffer::View>,
    accessors: Vec<json::Accessor>,
    images: Vec<json::Image>,
    textures: Vec<json::Texture>,
    data: Vec<u8>,
    buffer_map: HashMap<Vec<u8>, json::Index<json::buffer::View>>,
    accessor_map: HashMap<HashableAccessor, json::Index<json::Accessor>>,
}

#[derive(Debug, Clone, Error)]
pub enum MeshExportError {
    #[error("Mesh missing vertex positions attribute")]
    MissingVertexPosition,
    #[error("Mesh missing vertex normals attribute")]
    MissingVertexNormal,
    #[error("Texture not found")]
    TextureNotFound,
    #[error("Image conversion failed")]
    ImageConversionFailed,
    #[error("Failed serializing to bytes")]
    SerializationError,
    #[error("Failed converting index to u32 {0}")]
    U32CastError(std::num::TryFromIntError),
}

/// Used as a parameter for external facing functions
#[derive(Debug, Clone)]
pub struct GltfPose {
    pub translation: [f32; 3],
    // Unit quaternion
    pub rotation: [f32; 4],
    pub scale: Option<[f32; 3]>,
}

impl Default for GltfPose {
    fn default() -> Self {
        Self {
            translation: Default::default(),
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: None,
        }
    }
}

#[derive(Debug, Default)]
pub struct CompressGltfOptions {
    pub skip_materials: bool,
    // TODO(luca) implement merging for materials and textures
    //pub merge_materials: bool,
    //pub merge_textures: bool,
}

impl CompressGltfOptions {
    pub fn skip_materials() -> Self {
        Self {
            skip_materials: true,
        }
    }
}

#[derive(Debug, Default)]
pub struct GltfExportResult {
    root: json::Root,
    data: Vec<u8>,
}

impl GltfExportResult {
    pub fn to_bytes(self) -> Result<Vec<u8>, MeshExportError> {
        let json_string =
            json::serialize::to_string(&self.root).or(Err(MeshExportError::SerializationError))?;
        let mut json_offset = json_string
            .len()
            .try_into()
            .map_err(MeshExportError::U32CastError)?;
        let buf_length: u32 = self
            .data
            .len()
            .try_into()
            .map_err(MeshExportError::U32CastError)?;
        align_to_multiple_of_four(&mut json_offset);
        let glb = gltf::binary::Glb {
            header: gltf::binary::Header {
                magic: *b"glTF",
                version: 2,
                length: json_offset + buf_length,
            },
            bin: Some(Cow::Owned(self.data)),
            json: Cow::Owned(json_string.into_bytes()),
        };
        glb.to_vec().or(Err(MeshExportError::SerializationError))
    }
}

impl BuffersWrapper {
    fn new() -> Self {
        Self {
            buffer: json::Buffer {
                byte_length: 0_u64.into(),
                extensions: Default::default(),
                extras: Default::default(),
                name: None,
                uri: None,
            },
            buffer_views: vec![],
            accessors: vec![],
            images: vec![],
            textures: vec![],
            data: vec![],
            buffer_map: Default::default(),
            accessor_map: Default::default(),
        }
    }

    // Returns true if the buffer was added, false if it was found
    fn push_buffer<T>(&mut self, data: Vec<T>) -> (json::Index<json::buffer::View>, bool) {
        let bytes = to_padded_byte_vector(data);
        // Look for the buffer in the map first
        if let Some(idx) = self.buffer_map.get(&bytes) {
            return (*idx, false);
        }
        self.buffer_views.push(json::buffer::View {
            buffer: json::Index::new(0),
            byte_length: bytes.len().into(),
            byte_offset: Some(self.buffer.byte_length),
            byte_stride: Some(json::buffer::Stride(mem::size_of::<T>())),
            extensions: Default::default(),
            extras: Default::default(),
            name: None,
            target: Some(Valid(json::buffer::Target::ArrayBuffer)),
        });
        let idx = json::Index::new((self.buffer_views.len() as u32) - 1);
        self.buffer_map.insert(bytes.clone(), idx);
        self.data.extend(bytes);
        self.buffer.byte_length = self.data.len().into();
        (idx, true)
    }

    fn push_accessor(&mut self, accessor: json::Accessor) -> json::Index<json::Accessor> {
        let hashable = HashableAccessor(accessor.clone());
        if let Some(idx) = self.accessor_map.get(&hashable) {
            *idx
        } else {
            self.accessors.push(accessor);
            let idx = json::Index::new(self.accessors.len() as u32 - 1);
            self.accessor_map.insert(hashable, idx);
            idx
        }
    }

    fn push_image(&mut self, image: Image) -> Result<json::Index<json::Texture>, MeshExportError> {
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

        let (view_idx, added) = self.push_buffer(bytes);
        let texture_idx = if added {
            let image_idx = json::Index::new(self.images.len().try_into().unwrap());
            let texture_idx = json::Index::new(self.images.len().try_into().unwrap());
            self.images.push(json::Image {
                buffer_view: Some(view_idx),
                mime_type: Some(json::image::MimeType("image/png".to_owned())),
                uri: None,
                extensions: None,
                name: None,
                extras: Default::default(),
            });
            self.textures.push(json::Texture {
                sampler: None,
                source: image_idx,
                extensions: None,
                name: None,
                extras: Default::default(),
            });
            texture_idx
        } else {
            // Find the matching image
            let img_idx = json::Index::new(
                self.images
                    .iter()
                    .enumerate()
                    .find(|(_, img)| img.buffer_view == Some(view_idx))
                    .unwrap()
                    .0
                    .try_into()
                    .unwrap(),
            );
            // Now find the texture
            json::Index::new(
                self.textures
                    .iter()
                    .enumerate()
                    .find(|(_, texture)| texture.source == img_idx)
                    .unwrap()
                    .0
                    .try_into()
                    .unwrap(),
            )
        };
        Ok(texture_idx)
    }

    fn build(
        self,
    ) -> (
        Vec<json::Buffer>,
        Vec<json::buffer::View>,
        Vec<json::Accessor>,
        Vec<json::Image>,
        Vec<json::Texture>,
        Vec<u8>,
    ) {
        (
            vec![self.buffer],
            self.buffer_views,
            self.accessors,
            self.images,
            self.textures,
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

fn to_gltf_material<F: Fn(&Handle<Image>) -> Option<Image>>(
    buffers: &mut BuffersWrapper,
    mat: Option<&StandardMaterial>,
    image_getter: &F,
) -> Result<Option<json::Material>, MeshExportError> {
    let Some(mat) = mat else {
        return Ok(None);
    };
    let mut material = json::Material {
        pbr_metallic_roughness: json::material::PbrMetallicRoughness {
            base_color_factor: json::material::PbrBaseColorFactor(mat.base_color.as_rgba_f32()),
            metallic_factor: json::material::StrengthFactor(mat.metallic),
            roughness_factor: json::material::StrengthFactor(mat.perceptual_roughness),
            // TODO(luca) other properties here
            ..Default::default()
        },
        ..Default::default()
    };

    // TODO(luca) implement samplers and other types of textures
    if let Some(base_color_texture) = &mat.base_color_texture {
        let image = image_getter(base_color_texture).ok_or(MeshExportError::TextureNotFound)?;
        let texture_idx = buffers.push_image(image)?;
        material.pbr_metallic_roughness.base_color_texture = Some(json::texture::Info {
            index: texture_idx,
            tex_coord: 0,
            extensions: None,
            extras: Default::default(),
        });
    }
    Ok(Some(material))
}

struct VerticesData {
    positions: json::Index<json::Accessor>,
    normals: json::Index<json::Accessor>,
    uvs: Option<json::Index<json::Accessor>>,
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

    // Find min and max for bounding box
    let mut min = [f32::MAX, f32::MAX, f32::MAX];
    let mut max = [f32::MIN, f32::MIN, f32::MIN];

    for p in positions.iter() {
        for i in 0..3 {
            min[i] = f32::min(min[i], p[i]);
            max[i] = f32::max(max[i], p[i]);
        }
    }

    let num_vertices = positions.len().into();
    let position_view_idx = buffers.push_buffer(positions.clone()).0;
    let positions = json::Accessor {
        buffer_view: Some(position_view_idx),
        byte_offset: None,
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
    let normals_view_idx = buffers.push_buffer(normals.clone()).0;
    let normals = json::Accessor {
        buffer_view: Some(normals_view_idx),
        byte_offset: None,
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
    let uvs = if let Some(bevy_render::mesh::VertexAttributeValues::Float32x2(uvs)) =
        mesh.attribute(Mesh::ATTRIBUTE_UV_0)
    {
        let uvs_view_idx = buffers.push_buffer(uvs.clone()).0;
        let uvs = json::Accessor {
            buffer_view: Some(uvs_view_idx),
            byte_offset: None,
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
        Some(buffers.push_accessor(uvs))
    } else {
        None
    };

    let positions = buffers.push_accessor(positions);
    let normals = buffers.push_accessor(normals);
    let indices = get_indices_data(buffers, mesh.indices());
    Ok(VerticesData {
        positions,
        normals,
        uvs,
        indices,
    })
}

fn get_indices_data(
    buffers: &mut BuffersWrapper,
    indices: Option<&bevy_render::mesh::Indices>,
) -> Option<json::Index<json::Accessor>> {
    let Some(mesh_indices) = indices else {
        return None;
    };
    let count = mesh_indices.len().into();
    let indices = mesh_indices.iter().map(|idx| idx as u32).collect();
    let view_idx = buffers.push_buffer(indices).0;
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

#[derive(Clone, Debug)]
pub struct MeshData<'a> {
    pub mesh: &'a Mesh,
    pub material: Option<&'a StandardMaterial>,
    pub pose: Option<GltfPose>,
}

pub fn export_meshes<
    'a,
    F: Fn(&Handle<Image>) -> Option<Image>,
    I: IntoIterator<Item = MeshData<'a>>,
>(
    meshes: I,
    target_name: Option<String>,
    image_getter: F,
    options: CompressGltfOptions,
) -> Result<GltfExportResult, MeshExportError> {
    let mut buffers = BuffersWrapper::new();
    let mut root = json::Root {
        scenes: vec![json::Scene {
            extensions: Default::default(),
            extras: Default::default(),
            name: target_name,
            nodes: vec![],
        }],
        ..Default::default()
    };
    for data in meshes.into_iter() {
        let meshes_offset = root.meshes.len();
        let materials_offset = root.materials.len();

        let vertices_data = get_vertices_data(&mut buffers, data.mesh)?;

        let material = to_gltf_material(
            &mut buffers,
            data.material,
            &image_getter,
        )?;

        let mut primitive = json::mesh::Primitive {
            attributes: {
                let mut map = BTreeMap::new();
                map.insert(
                    Valid(json::mesh::Semantic::Positions),
                    vertices_data.positions,
                );
                map.insert(Valid(json::mesh::Semantic::Normals), vertices_data.normals);
                if let Some(uvs) = vertices_data.uvs {
                    map.insert(Valid(json::mesh::Semantic::TexCoords(0)), uvs);
                }
                map
            },
            extensions: Default::default(),
            extras: Default::default(),
            indices: None,
            material: None,
            mode: Valid(json::mesh::Mode::Triangles),
            targets: None,
        };
        if let Some(indices) = vertices_data.indices {
            primitive.indices = Some(indices);
        }
        if material.is_some() {
            primitive.material = Some(json::Index::new(
                materials_offset
                    .try_into()
                    .map_err(MeshExportError::U32CastError)?,
            ));
        }

        let mesh_idx = root
            .meshes
            .iter()
            .enumerate()
            .find(|(_, mesh)| {
                let p = &mesh.primitives[0];
                // TODO(luca) implement equality for materials as well
                //p.material == primitive.material &&
                p.attributes == primitive.attributes
                    && p.indices == primitive.indices
                    && p.mode == primitive.mode
            })
            .map(|elem| elem.0)
            .unwrap_or_else(|| {
                let mesh = json::Mesh {
                    extensions: Default::default(),
                    extras: Default::default(),
                    name: None,
                    primitives: vec![primitive],
                    weights: None,
                };
                root.meshes.push(mesh);
                meshes_offset
            });

        let node = json::Node {
            camera: None,
            children: None,
            extensions: Default::default(),
            extras: Default::default(),
            matrix: None,
            mesh: Some(json::Index::new(
                mesh_idx.try_into().map_err(MeshExportError::U32CastError)?,
            )),
            name: None,
            rotation: data
                .pose
                .as_ref()
                .map(|p| json::scene::UnitQuaternion(p.rotation)),
            scale: data.pose.as_ref().and_then(|p| p.scale),
            translation: data.pose.map(|p| p.translation),
            skin: None,
            weights: None,
        };
        root.nodes.push(node);

        if let Some(material) = material {
            root.materials.push(material);
        }
    }
    root.scenes[0].nodes = (0u32..root
        .nodes
        .len()
        .try_into()
        .map_err(MeshExportError::U32CastError)?)
        .map(json::Index::new)
        .collect();
    let (buffers, buffer_views, accessors, images, textures, buffer_bytes) = buffers.build();
    root.accessors = accessors;
    root.images = images;
    root.textures = textures;
    root.buffers = buffers;
    root.buffer_views = buffer_views;
    for (idx, ref mut mat) in root.materials.iter_mut().enumerate() {
        mat.name = Some(format!("Material_{}", idx));
    }
    for (idx, ref mut texture) in root.textures.iter_mut().enumerate() {
        texture.name = Some(format!("Texture_{}", idx));
    }
    for (idx, ref mut image) in root.images.iter_mut().enumerate() {
        image.name = Some(format!("Image_{}", idx));
    }
    Ok(GltfExportResult {
        root,
        data: buffer_bytes,
    })
}
