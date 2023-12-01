use bevy::prelude::*;
use bevy::render::mesh::Indices;
use bevy::render::render_resource::PrimitiveTopology;
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

struct GltfIndicesData {
    buffer_view: json::buffer::View,
    accessor: json::Accessor,
    accessor_index: json::Index<json::Accessor>,
    indices: Vec<u32>,
}

fn align_to_multiple_of_four(n: &mut u32) {
    *n = (*n + 3) & !3;
}

fn to_padded_byte_vector<T>(vec: Vec<T>) -> Vec<u8> {
    let byte_length = vec.len() * mem::size_of::<T>();
    let byte_capacity = vec.capacity() * mem::size_of::<T>();
    let alloc = vec.into_boxed_slice();
    let ptr = Box::<[T]>::into_raw(alloc) as *mut u8;
    let mut new_vec = unsafe { Vec::from_raw_parts(ptr, byte_length, byte_capacity) };
    while new_vec.len() % 4 != 0 {
        new_vec.push(0); // pad to multiple of four bytes
    }
    new_vec
}

fn to_gltf_material(mat: &StandardMaterial) -> json::Material {
    let mut json_mat = json::Material::default();
    json::Material {
        pbr_metallic_roughness: json::material::PbrMetallicRoughness {
            base_color_factor: json::material::PbrBaseColorFactor(mat.base_color.as_rgba_f32()),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn get_indices_data(current_offset: u32, mesh: &Mesh) -> Option<GltfIndicesData> {
    let Some(mesh_indices) = mesh.indices() else {
        return None;
    };
    let buffer_length = (mesh_indices.len() * mem::size_of::<u32>()) as u32;
    let buffer_view = json::buffer::View {
        buffer: json::Index::new(0),
        byte_length: buffer_length,
        byte_offset: Some(current_offset),
        byte_stride: Some(mem::size_of::<u32>() as u32),
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        target: Some(Valid(json::buffer::Target::ArrayBuffer)),
    };
    let accessor = json::Accessor {
        buffer_view: Some(json::Index::new(1)),
        byte_offset: None,
        count: mesh_indices.len() as u32,
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
    };
    // TODO(luca) Indexes can also be u8 / u16
    let indices = mesh_indices.iter().map(|idx| idx as u32).collect();
    // TODO(luca) remove this hardcoded number
    let accessor_index = json::Index::<json::Accessor>::new(3);
    Some(GltfIndicesData {
        buffer_view,
        accessor,
        accessor_index,
        indices,
    })
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
            ..Default::default()
        },
    )
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct BevyVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct BevyIndex(u32);

fn export_mesh(output: Output) -> Result<(), ()> {
    let (mesh, material) = create_bevy_sample_mesh();
    let material = to_gltf_material(&material);

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
        .map(|(p, n)| BevyVertex {
            position: p.clone(),
            normal: n.clone(),
            uv: Default::default(),
        })
        .collect::<Vec<_>>();

    if let Some(bevy::render::mesh::VertexAttributeValues::Float32x2(uvs)) =
        mesh.attribute(Mesh::ATTRIBUTE_UV_0)
    {
        for (idx, uv) in uvs.iter().enumerate() {
            vertices[idx].uv = uv.clone();
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
    let buffer_length = (vertices.len() * mem::size_of::<BevyVertex>()) as u32;
    let buffer = json::Buffer {
        byte_length: buffer_length,
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        uri: if output == Output::Standard {
            Some("buffer0.bin".into())
        } else {
            None
        },
    };
    let buffer_view = json::buffer::View {
        buffer: json::Index::new(0),
        byte_length: buffer.byte_length,
        byte_offset: None,
        byte_stride: Some(mem::size_of::<BevyVertex>() as u32),
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        target: Some(Valid(json::buffer::Target::ArrayBuffer)),
    };
    let positions = json::Accessor {
        buffer_view: Some(json::Index::new(0)),
        byte_offset: Some(0),
        count: vertices.len() as u32,
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
        buffer_view: Some(json::Index::new(0)),
        byte_offset: Some((3 * mem::size_of::<f32>()) as u32),
        count: vertices.len() as u32,
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
        buffer_view: Some(json::Index::new(0)),
        byte_offset: Some((6 * mem::size_of::<f32>()) as u32),
        count: vertices.len() as u32,
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
    let indices_data = get_indices_data(buffer_length, &mesh);

    let primitive = json::mesh::Primitive {
        attributes: {
            let mut map = std::collections::BTreeMap::new();
            map.insert(Valid(json::mesh::Semantic::Positions), json::Index::new(0));
            map.insert(Valid(json::mesh::Semantic::Normals), json::Index::new(1));
            map.insert(
                Valid(json::mesh::Semantic::TexCoords(0)),
                json::Index::new(2),
            );
            map
        },
        extensions: Default::default(),
        extras: Default::default(),
        indices: None,
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

    let mut root = json::Root {
        accessors: vec![positions, normals, uvs],
        buffers: vec![buffer],
        buffer_views: vec![buffer_view],
        meshes: vec![mesh],
        nodes: vec![node],
        scenes: vec![json::Scene {
            extensions: Default::default(),
            extras: Default::default(),
            name: None,
            nodes: vec![json::Index::new(0)],
        }],
        materials: vec![material],
        ..Default::default()
    };
    let indices = if let Some(data) = indices_data {
        root.accessors.push(data.accessor);
        root.buffers[0].byte_length += (data.indices.len() + mem::size_of::<u32>()) as u32;
        root.buffer_views.push(data.buffer_view);
        root.meshes[0].primitives[0].indices = Some(data.accessor_index);
        Some(data.indices)
    } else {
        None
    };

    match output {
        Output::Standard => {
            let _ = fs::create_dir("triangle");

            let writer = fs::File::create("triangle/triangle.gltf").expect("I/O error");
            json::serialize::to_writer_pretty(writer, &root).expect("Serialization error");

            let bin = to_padded_byte_vector(vertices);
            let mut writer = fs::File::create("triangle/buffer0.bin").expect("I/O error");
            writer.write_all(&bin).expect("I/O error");
        }
        Output::Binary => {
            let mut buf_length = buffer_length;
            let mut buffer = to_padded_byte_vector(vertices);
            if let Some(indices) = indices {
                buf_length += (indices.len() * mem::size_of::<u32>()) as u32;
                buffer.extend(to_padded_byte_vector(indices));
            }
            let json_string = json::serialize::to_string(&root).expect("Serialization error");
            let mut json_offset = json_string.len() as u32;
            align_to_multiple_of_four(&mut json_offset);
            let glb = gltf::binary::Glb {
                header: gltf::binary::Header {
                    magic: *b"glTF",
                    version: 2,
                    length: json_offset + buf_length,
                },
                bin: Some(Cow::Owned(buffer)),
                json: Cow::Owned(json_string.into_bytes()),
            };
            let writer = std::fs::File::create("triangle.glb").expect("I/O error");
            glb.to_writer(writer).expect("glTF binary output error");
        }
    }

    Ok(())
}

/*
fn export(output: Output) {
    let triangle_vertices = vec![
        Vertex {
            position: [0.0, 0.5, 0.0],
            color: [1.0, 0.0, 0.0],
        },
        Vertex {
            position: [-0.5, -0.5, 0.0],
            color: [0.0, 1.0, 0.0],
        },
        Vertex {
            position: [0.5, -0.5, 0.0],
            color: [0.0, 0.0, 1.0],
        },
    ];

    let (min, max) = bounding_coords(&triangle_vertices);

    let buffer_length = (triangle_vertices.len() * mem::size_of::<Vertex>()) as u32;
    let buffer = json::Buffer {
        byte_length: buffer_length,
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        uri: if output == Output::Standard {
            Some("buffer0.bin".into())
        } else {
            None
        },
    };
    let buffer_view = json::buffer::View {
        buffer: json::Index::new(0),
        byte_length: buffer.byte_length,
        byte_offset: None,
        byte_stride: Some(mem::size_of::<Vertex>() as u32),
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        target: Some(Valid(json::buffer::Target::ArrayBuffer)),
    };
    let positions = json::Accessor {
        buffer_view: Some(json::Index::new(0)),
        byte_offset: Some(0),
        count: triangle_vertices.len() as u32,
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
    let colors = json::Accessor {
        buffer_view: Some(json::Index::new(0)),
        byte_offset: Some((3 * mem::size_of::<f32>()) as u32),
        count: triangle_vertices.len() as u32,
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

    let primitive = json::mesh::Primitive {
        attributes: {
            let mut map = std::collections::BTreeMap::new();
            map.insert(Valid(json::mesh::Semantic::Positions), json::Index::new(0));
            map.insert(Valid(json::mesh::Semantic::Colors(0)), json::Index::new(1));
            map
        },
        extensions: Default::default(),
        extras: Default::default(),
        indices: None,
        material: None,
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

    let root = json::Root {
        accessors: vec![positions, colors],
        buffers: vec![buffer],
        buffer_views: vec![buffer_view],
        meshes: vec![mesh],
        nodes: vec![node],
        scenes: vec![json::Scene {
            extensions: Default::default(),
            extras: Default::default(),
            name: None,
            nodes: vec![json::Index::new(0)],
        }],
        ..Default::default()
    };

    match output {
        Output::Standard => {
            let _ = fs::create_dir("triangle");

            let writer = fs::File::create("triangle/triangle.gltf").expect("I/O error");
            json::serialize::to_writer_pretty(writer, &root).expect("Serialization error");

            let bin = to_padded_byte_vector(triangle_vertices);
            let mut writer = fs::File::create("triangle/buffer0.bin").expect("I/O error");
            writer.write_all(&bin).expect("I/O error");
        }
        Output::Binary => {
            let json_string = json::serialize::to_string(&root).expect("Serialization error");
            let mut json_offset = json_string.len() as u32;
            align_to_multiple_of_four(&mut json_offset);
            let glb = gltf::binary::Glb {
                header: gltf::binary::Header {
                    magic: *b"glTF",
                    version: 2,
                    length: json_offset + buffer_length,
                },
                bin: Some(Cow::Owned(to_padded_byte_vector(triangle_vertices))),
                json: Cow::Owned(json_string.into_bytes()),
            };
            let writer = std::fs::File::create("triangle.glb").expect("I/O error");
            glb.to_writer(writer).expect("glTF binary output error");
        }
    }
}
*/

fn main() {
    println!("Hello, world!");
    export_mesh(Output::Binary).unwrap();
}
