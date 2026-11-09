//! Meshes in the level editor: the hierarchy icon and the Create menu's
//! primitive meshes, which the plugin's root registers through the level
//! editor's extension points (`ui_level_editor::extend`). Each add is one
//! undoable command.

use meshes_component::StaticMeshComponent;
use ui_level_editor::LevelEditorState;
use ui_level_editor::commands::{CommandResult, TypedComponent};
use ui_level_editor::extend::add_object;
use ui_level_editor::scene_edit::{ObjectType, Transform};

/// The hierarchy's icon for an object holding a mesh (static or skeletal).
pub fn object_icon(
    world: &pulsar_scenedb::World,
    object: pulsar_scenedb::Entity,
) -> Option<(ui::IconName, gpui::Hsla)> {
    pulsar_scene_model::attachments::instances(world, object)
        .iter()
        .any(|instance| {
            world.get::<StaticMeshComponent>(*instance).is_some()
                || world
                    .get::<meshes_component::components::SkeletalMeshComponent>(*instance)
                    .is_some()
        })
        .then_some((
            ui::IconName::Box,
            ui::hierarchical_tree::tree_colors::CODE_PURPLE,
        ))
}

/// A primitive mesh (`meshes/primitives/<file>`) at the origin.
fn primitive(state: &mut LevelEditorState, name: &str, file: &str) -> CommandResult {
    let mesh = StaticMeshComponent::for_mesh_asset(&format!("meshes/primitives/{file}"));
    add_object(
        state,
        name,
        ObjectType::Empty,
        Transform::default(),
        vec![TypedComponent::new(mesh)],
    )
}

/// One of the Create menu's primitive meshes.
pub struct Primitive {
    /// The entry's stable id.
    pub id: &'static str,
    /// The Create menu's text.
    pub menu_label: fn() -> String,
    /// Its place in the menu (meshes are the 200s).
    pub order: i32,
    /// Add it as one undoable command.
    pub create: fn(&mut LevelEditorState) -> CommandResult,
}

/// The Create menu's primitive meshes.
pub const PRIMITIVES: [Primitive; 4] = [
    Primitive {
        id: "mesh_cube",
        menu_label: || rust_i18n::t!("LevelEditor.Create.Cube").to_string(),
        order: 200,
        create: |state| primitive(state, "Cube", "SM_Cube.fbx"),
    },
    Primitive {
        id: "mesh_sphere",
        menu_label: || rust_i18n::t!("LevelEditor.Create.Sphere").to_string(),
        order: 210,
        create: |state| primitive(state, "Sphere", "SM_Sphere.fbx"),
    },
    Primitive {
        id: "mesh_cylinder",
        menu_label: || rust_i18n::t!("LevelEditor.Create.Cylinder").to_string(),
        order: 220,
        create: |state| primitive(state, "Cylinder", "SM_Cylinder.fbx"),
    },
    Primitive {
        id: "mesh_plane",
        menu_label: || rust_i18n::t!("LevelEditor.Create.Plane").to_string(),
        order: 230,
        create: |state| primitive(state, "Plane", "SM_Plane.fbx"),
    },
];
