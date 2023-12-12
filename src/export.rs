use bevy_asset::Handle;
use bevy_pbr::StandardMaterial;
use bevy_render::prelude::*;
use gltf_json as json;

use std::collections::{BTreeSet, HashMap};
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
    MissingVertexUv,
    TextureNotFound,
    ImageConversionFailed,
    SerializationError,
}

/// Used as a parameter for external facing functions
#[derive(Debug, Clone, Default)]
pub struct GltfPose {
    pub translation: [f32; 3],
    // Unit quaternion
    pub rotation: [f32; 4],
}

#[derive(Debug)]
pub struct CompressGltfOptions {
    pub merge_images: bool,
    // TODO(luca) implement merging for materials and textures
    //pub merge_materials: bool,
    pub merge_textures: bool,
}

impl CompressGltfOptions {
    pub fn maximum() -> Self {
        Self {
            merge_images: true,
            merge_textures: true,
        }
    }
}

fn are_textures_equal(t1: &json::Texture, t2: &json::Texture) -> bool {
    // Extensions and extras are ignored
    t1.sampler == t2.sampler && t1.source == t2.source
}

#[derive(Debug, Default)]
pub struct GltfExportResult {
    root: json::Root,
    data: Vec<u8>,
}

impl GltfExportResult {
    pub fn combine_with<T: IntoIterator<Item = GltfExportResult>>(
        mut self,
        others: T,
        options: CompressGltfOptions,
    ) -> Self {
        let mut buffer_map: HashMap<Vec<u8>, json::Index<json::buffer::View>> = HashMap::new();
        let mut others = others.into_iter().peekable();
        if others.peek().is_none() {
            return self;
        };
        if options.merge_images {
            // Build cache
            buffer_map = self.build_image_cache();
        }
        for g2 in others {
            // For now just merge, TODO(luca) if materials are duplicated, just change the reference
            let buffer_views_offset = self.root.buffer_views.len();
            let accessors_offset = self.root.accessors.len();
            let meshes_offset = self.root.meshes.len();
            let nodes_offset = self.root.nodes.len();
            let materials_offset = self.root.materials.len();
            let textures_offset = self.root.textures.len();
            let images_offset = self.root.images.len();
            // Now merge the vectors and update the references
            // TODO(luca) remove all try_intos and as u32 for fallible overflow checks
            // TODO(luca) remove all unwraps on options below
            let mut buffers_to_remove = BTreeSet::new();
            for mut scene_node in g2.root.scenes[0].nodes.iter().cloned() {
                scene_node =
                    json::Index::new((scene_node.value() + nodes_offset).try_into().unwrap());
                self.root.scenes[0].nodes.push(scene_node);
            }
            for mut node in g2.root.nodes.into_iter() {
                if let Some(ref mut node_mesh) = node.mesh {
                    *node_mesh =
                        json::Index::new((node_mesh.value() + meshes_offset).try_into().unwrap());
                }
                self.root.nodes.push(node);
            }
            for mut mesh in g2.root.meshes.into_iter() {
                for primitive in mesh.primitives.iter_mut() {
                    if let Some(ref mut primitive_material) = primitive.material {
                        *primitive_material = json::Index::new(
                            (primitive_material.value() + materials_offset)
                                .try_into()
                                .unwrap(),
                        );
                    }
                    // Update all attributes and indices
                    for index in primitive.attributes.values_mut() {
                        *index = json::Index::new(
                            (index.value() + accessors_offset).try_into().unwrap(),
                        );
                    }
                    if let Some(ref mut indices) = primitive.indices {
                        *indices = json::Index::new(
                            (indices.value() + accessors_offset).try_into().unwrap(),
                        );
                    }
                }
                self.root.meshes.push(mesh);
            }
            for mut image in g2.root.images.into_iter() {
                if let Some(ref mut image_buffer_view) = image.buffer_view {
                    // Check if it's in the map, if it is reuse an existing buffer, otherwise allocate
                    // a new one
                    let Some(view) = g2.root.buffer_views.get(image_buffer_view.value()) else {
                        // Error!
                        continue;
                    };
                    let start = view.byte_offset.unwrap_or_default() as usize;
                    let end = start + (view.byte_length as usize);
                    match buffer_map.get(&g2.data[start..end]) {
                        Some(value) => {
                            buffers_to_remove.insert(*image_buffer_view);
                            *image_buffer_view = *value;
                        }
                        None => {
                            // Keep it, insert in the hashmap
                            *image_buffer_view = json::Index::new(
                                (image_buffer_view.value() + buffer_views_offset)
                                    .try_into()
                                    .unwrap(),
                            );
                            buffer_map.insert(g2.data[start..end].to_vec(), *image_buffer_view);
                        }
                    }
                }
                self.root.images.push(image);
            }
            let mut found_textures_map = HashMap::new();
            for (idx, mut texture) in g2.root.textures.into_iter().enumerate() {
                // TODO(luca) Implement a hash function to make this constant time
                if let Some(pos) = self
                    .root
                    .textures
                    .iter()
                    .position(|t| are_textures_equal(t, &texture))
                {
                    found_textures_map.insert(idx, pos);
                } else {
                    texture.source = json::Index::new(
                        (texture.source.value() + images_offset).try_into().unwrap(),
                    );
                    self.root.textures.push(texture);
                }
            }
            for mut material in g2.root.materials.into_iter() {
                if let Some(ref mut base_color_texture) =
                    material.pbr_metallic_roughness.base_color_texture
                {
                    // TODO(luca) put a check for the optimization boolean flag instead of default
                    // on
                    if let Some(pos) = found_textures_map.get(&base_color_texture.index.value()) {
                        base_color_texture.index = json::Index::new((*pos).try_into().unwrap());
                    } else {
                        base_color_texture.index = json::Index::new(
                            (base_color_texture.index.value() + textures_offset)
                                .try_into()
                                .unwrap(),
                        );
                    }
                }
                self.root.materials.push(material);
            }
            // Now merge the buffers
            let mut w1 = BuffersWrapper {
                buffer: self.root.buffers.into_iter().next().unwrap(),
                buffer_views: self.root.buffer_views,
                accessors: self.root.accessors,
                data: self.data,
            };
            let mut w2 = BuffersWrapper {
                buffer: g2.root.buffers.into_iter().next().unwrap(),
                buffer_views: g2.root.buffer_views,
                accessors: g2.root.accessors,
                data: g2.data,
            };
            w2.clear_buffers(buffers_to_remove);
            w1.merge_with(w2);
            let (buffer, buffer_views, accessors, data) = w1.build();
            self.root.buffers = buffer;
            self.root.buffer_views = buffer_views;
            self.root.accessors = accessors;
            self.data = data;
        }
        // Reassign names to textures, materials and images
        for (idx, ref mut mat) in self.root.materials.iter_mut().enumerate() {
            mat.name = Some(format!("Material_{}", idx));
        }
        for (idx, ref mut texture) in self.root.textures.iter_mut().enumerate() {
            texture.name = Some(format!("Texture_{}", idx));
        }
        for (idx, ref mut image) in self.root.images.iter_mut().enumerate() {
            image.name = Some(format!("Image_{}", idx));
        }
        self
    }

    pub fn to_bytes(self) -> Result<Vec<u8>, MeshExportError> {
        let json_string =
            json::serialize::to_string(&self.root).or(Err(MeshExportError::SerializationError))?;
        let mut json_offset = json_string.len() as u32;
        let buf_length = self.data.len() as u32;
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

    fn build_image_cache(&mut self) -> HashMap<Vec<u8>, json::Index<json::buffer::View>> {
        // We build a hashmap of image buffers and remove duplicates / fix accessors if needed.
        let mut buffer_map = HashMap::new();
        for image in self.root.images.iter() {
            // Fetch the buffer
            let Some(buffer_index) = image.buffer_view else {
                continue;
            };
            let Some(view) = self.root.buffer_views.get(buffer_index.value()) else {
                // Error!
                continue;
            };
            // TODO(luca) support multiple buffers, not really possible in glb though
            // TODO(luca) take byte_stride into account, will probably require cloning though
            let start = view.byte_offset.unwrap_or_default() as usize;
            let end = start + (view.byte_length as usize);
            let bytes_slice = &self.data[start..end];

            if !buffer_map.contains_key(bytes_slice) {
                // Map the index of the buffer to keep
                buffer_map.insert(bytes_slice.to_vec(), buffer_index);
            } else {
                // This won't really optimize cases in which the starting mesh has multiple
                // copies of the buffer, for now just log a warning it's not a big deal
                println!("Warning, duplicated key found");
            }
        }
        buffer_map
    }
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

    fn merge_with(&mut self, other: BuffersWrapper) {
        self.accessors
            .reserve(self.accessors.len() + other.accessors.len());
        let original_byte_length = self.buffer.byte_length;
        let buffer_view_offset = self.buffer_views.len();
        self.buffer_views
            .reserve(self.buffer_views.len() + other.buffer_views.len());
        // Merge the buffer views
        for mut view in other.buffer_views.into_iter() {
            if let Some(ref mut byte_offset) = view.byte_offset {
                *byte_offset += original_byte_length;
            }
            self.buffer_views.push(view);
        }
        // Now merge the accessors
        for mut accessor in other.accessors.into_iter() {
            if let Some(ref mut buffer_view) = accessor.buffer_view {
                *buffer_view = json::Index::new((buffer_view.value() + buffer_view_offset) as u32);
            }
            // TODO(luca) take byte offset into account?
            self.accessors.push(accessor);
        }
        // Now merge the data, it doesn't need padding
        self.data.extend(other.data);
        self.buffer.byte_length = self.data.len() as u32;
    }

    fn clear_buffers(&mut self, buffers_to_remove: BTreeSet<json::Index<json::buffer::View>>) {
        // Iterate in inverse order to not invalidate indexes
        for remove in buffers_to_remove.iter().rev() {
            // Note this panics if the index is invalid!
            let view = self.buffer_views.remove(remove.value());
            let start = view.byte_offset.unwrap_or_default() as usize;
            let end = start + (view.byte_length as usize);
            self.data.drain(start..end);
            self.buffer.byte_length -= view.byte_length;
            // TODO*luca) We assume that accessors don't point to this buffer, this is OK for
            // images since they don't use accessors
            for accessor in self.accessors.iter() {
                if accessor.buffer_view.is_some_and(|v| v == *remove) {
                    // Error!
                    continue;
                }
            }
        }
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
    mat: &StandardMaterial,
    image_getter: F,
) -> Result<(json::Material, Vec<json::Texture>, Vec<json::Image>), MeshExportError> {
    let mut textures = vec![];
    let mut images = vec![];
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
            mime_type: Some(json::image::MimeType("image/png".to_owned())),
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
    mut mesh: Mesh,
) -> Result<VerticesData, MeshExportError> {
    let Some(bevy_render::mesh::VertexAttributeValues::Float32x3(positions)) =
        mesh.remove_attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        return Err(MeshExportError::MissingVertexPosition);
    };
    let Some(bevy_render::mesh::VertexAttributeValues::Float32x3(normals)) =
        mesh.remove_attribute(Mesh::ATTRIBUTE_NORMAL)
    else {
        return Err(MeshExportError::MissingVertexNormal);
    };

    let Some(bevy_render::mesh::VertexAttributeValues::Float32x2(uvs)) =
        mesh.remove_attribute(Mesh::ATTRIBUTE_UV_0)
    else {
        return Err(MeshExportError::MissingVertexUv);
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

    let num_vertices = positions.len() as u32;
    let position_view_idx = buffers.push_buffer(positions);
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
    let normals_view_idx = buffers.push_buffer(normals);
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
    let uvs_view_idx = buffers.push_buffer(uvs);
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
    let positions = buffers.push_accessor(positions);
    let normals = buffers.push_accessor(normals);
    let uvs = buffers.push_accessor(uvs);
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

pub fn export_mesh<F: Fn(&Handle<Image>) -> Option<Image>>(
    mesh: Mesh,
    material: StandardMaterial,
    pose: Option<GltfPose>,
    image_getter: F,
) -> Result<GltfExportResult, MeshExportError> {
    let mut buffers = BuffersWrapper::new();
    let vertices_data = get_vertices_data(&mut buffers, mesh)?;

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
        rotation: pose
            .as_ref()
            .map(|p| json::scene::UnitQuaternion(p.rotation)),
        scale: None,
        translation: pose.map(|p| p.translation),
        skin: None,
        weights: None,
    };
    let (buffers, buffer_views, accessors, buffer_bytes) = buffers.build();

    Ok(GltfExportResult {
        root: json::Root {
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
        },
        data: buffer_bytes,
    })
}
