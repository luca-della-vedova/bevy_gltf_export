use bevy::prelude::*;
use bevy::render::mesh::Indices;
use bevy::render::render_resource::PrimitiveTopology;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use bevy_gltf_export::{export_mesh, MeshExportError};

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
            base_color: Color::rgba(1.0, 1.0, 1.0, 1.0),
            base_color_texture: Some(Handle::default()),
            ..Default::default()
        },
    )
}

// Dummy imag getter that returns a solid blue texture
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
    ))
}

fn export_test_mesh() -> Result<Vec<u8>, MeshExportError> {
    let (mesh, material) = create_bevy_sample_mesh();
    export_mesh(mesh, material, sample_image_getter)
}

fn main() {
    println!("Hello, world!");
    let vec = export_test_mesh().unwrap();
    std::fs::write("triangle.glb", vec).unwrap();
}
