use bevy_asset::Handle;
use bevy_pbr::StandardMaterial;
use bevy_render::mesh::Indices;
use bevy_render::prelude::*;
use bevy_render::render_resource::{PrimitiveTopology, Extent3d, TextureDimension, TextureFormat};
use bevy_render::render_asset::RenderAssetUsages;
use bevy_transform::prelude::Transform;

use bevy_gltf_export::{export_meshes, CompressGltfOptions, MeshData, MeshExportError};

fn create_bevy_sample_mesh() -> (Mesh, StandardMaterial) {
    // Create a new mesh using a triangle list topology, where each set of 3 vertices composes a triangle.
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    // Add 4 vertices, each with its own position attribute (coordinate in
    // 3D space), for each of the corners of the parallelogram.
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
    );
    // Assign a UV coordinate to each vertex.
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_UV_0,
        vec![[0.0, 1.0], [0.5, 0.0], [1.0, 0.0], [0.5, 1.0]],
    );
    // Assign normals (everything points outwards)
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        vec![
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
    );
    // After defining all the vertices and their attributes, build each triangle using the
    // indices of the vertices that make it up in a counter-clockwise order.
    mesh.insert_indices(Indices::U32(vec![
        // First triangle
        0, 3, 1, // Second triangle
        1, 3, 2,
    ]));
    (
        mesh,
        StandardMaterial {
            base_color: Color::rgba(1.0, 1.0, 1.0, 1.0),
            base_color_texture: Some(Handle::default()),
            ..Default::default()
        },
    )
}

fn create_bevy_sample_mesh2() -> (Mesh, StandardMaterial) {
    // Create a new mesh using a triangle list topology, where each set of 3 vertices composes a triangle.
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    // Add 4 vertices, each with its own position attribute (coordinate in
    // 3D space), for each of the corners of the parallelogram.
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![
            [10.0, 10.0, 10.0],
            [11.0, 12.0, 10.0],
            [12.0, 12.0, 10.0],
            [11.0, 10.0, 10.0],
        ],
    );
    // Assign a UV coordinate to each vertex.
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_UV_0,
        vec![[0.0, 1.0], [0.5, 0.0], [1.0, 0.0], [0.5, 1.0]],
    );
    // Assign normals (everything points outwards)
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        vec![
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
    );
    // After defining all the vertices and their attributes, build each triangle using the
    // indices of the vertices that make it up in a counter-clockwise order.
    mesh.insert_indices(Indices::U32(vec![
        // First triangle
        0, 3, 1, // Second triangle
        1, 3, 2,
    ]));
    (
        mesh,
        StandardMaterial {
            base_color: Color::rgba(1.0, 0.0, 0.0, 1.0),
            base_color_texture: Some(Handle::default()),
            ..Default::default()
        },
    )
}

// Dummy image getter that returns a solid blue texture
fn sample_image_getter(_id: &Handle<Image>) -> Option<Image> {
    let extent = Extent3d {
        width: 512,
        height: 512,
        ..Default::default()
    };
    Some(Image::new_fill(
        extent,
        TextureDimension::D2,
        &[0, 0, 255, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::default(),
    ))
}

fn export_test_mesh() -> Result<Vec<u8>, MeshExportError> {
    let (mesh, material) = create_bevy_sample_mesh();
    let (mesh2, material2) = create_bevy_sample_mesh2();
    let data1 = MeshData {
        mesh: &mesh,
        material: Some(&material),
        transform: None,
    };
    let data2 = MeshData {
        mesh: &mesh2,
        material: Some(&material2),
        transform: None,
    };
    let data3 = MeshData {
        mesh: &mesh,
        material: Some(&material2),
        transform: Some(Transform {
            translation: [2.0, 2.0, 2.0].into(),
            ..Default::default()
        }),
    };
    let res = export_meshes(
        [data1, data2, data3],
        None,
        sample_image_getter,
        CompressGltfOptions::default(),
    )?;
    res.to_bytes()
}

fn main() {
    let vec = export_test_mesh().unwrap();
    std::fs::write("triangle.glb", vec).unwrap();
}
