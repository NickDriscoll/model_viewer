//Including the other source files
mod glutil;
mod prims;
mod routines;
mod structs;

use crate::{glutil::*, prims::*, routines::*, structs::*};
use chrono::offset::Local;
use gl::types::*;
use glfw::{Action, Context, CursorMode, Key, MouseButton, WindowEvent, WindowMode};
use glyph_brush::{
    rusttype::Scale, BrushAction, BrushError, GlyphBrushBuilder, GlyphCruncher, Section,
};
use image::{DynamicImage, ImageBuffer};
use nalgebra_glm as glm;
use nfd::Response;
use noise::{NoiseFn, OpenSimplex, Seedable};
use openvr::{
    button_id,
    compositor::texture::{ColorSpace, Handle, Texture},
    ApplicationType, Eye, TrackedControllerRole,
};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::BufReader,
    os::raw::c_void,
    sync::mpsc,
    time::{Instant, SystemTime, UNIX_EPOCH},
    {mem, path::Path, ptr, thread},
};
use wavefront_obj::{mtl, obj};

//The distances of the near and far clipping planes from the origin
const NEAR_Z: f32 = 0.1;
const FAR_Z: f32 = 400.0;

//Left eye, Right eye, Companion window
const RENDER_PASSES: usize = 3;

//Things you can request the worker thread to do
enum WorkOrder {
    Model,
    Quit,
}

//Things the worker thread can send back to the main thread
enum WorkResult {
    Model(Option<MeshData>),
}

fn main() {
    //Initialize OpenVR
    let (openvr_context, openvr_system, openvr_compositor, openvr_rendermodels) = unsafe {
        match openvr::init(ApplicationType::Scene) {
            Ok(ctxt) => {
                let system = ctxt.system().unwrap();
                let compositor = ctxt.compositor().unwrap();
                let render_models = ctxt.render_models().unwrap();
                (
                    Some(ctxt),
                    Some(system),
                    Some(compositor),
                    Some(render_models),
                )
            }
            Err(e) => {
                println!("OpenVR initialization error: {}", e);
                (None, None, None, None)
            }
        }
    };

    //Calculate VR render target size
    let render_target_size = match &openvr_system {
        Some(sys) => sys.recommended_render_target_size(),
        None => (0, 0),
    };

    //Init glfw
    let mut glfw = match glfw::init(glfw::FAIL_ON_ERRORS) {
        Ok(g) => g,
        Err(e) => {
            panic!("GLFW init error: {}", e);
        }
    };

    //Using OpenGL 3.3 core, but that could change
    glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(
        glfw::OpenGlProfileHint::Core,
    ));

    //Create window
    let mut window_size = (1920, 1080);
    let (mut window, events) = glfw
        .create_window(
            window_size.0,
            window_size.1,
            "Model viewer",
            WindowMode::Windowed,
        )
        .unwrap();

    //Calculate window's aspect ratio
    let mut aspect_ratio = window_size.0 as f32 / window_size.1 as f32;

    //Configure what kinds of events GLFW will listen for
    window.set_key_polling(true);
    window.set_framebuffer_size_polling(true);
    window.set_mouse_button_polling(true);
    window.set_scroll_polling(true);

    //Load all OpenGL function pointers, GLFW does all the work here
    gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);

    //These OpenGL settings are only set once, so we just do it here
    unsafe {
        gl::Enable(gl::DEPTH_TEST); //Enable depth testing
        gl::DepthFunc(gl::LEQUAL); //Pass the fragment with the smallest z-value. Needs to be <= instead of < because for all skybox pixels z = 1.0
        gl::Enable(gl::FRAMEBUFFER_SRGB); //Enable automatic linear->SRGB space conversion
        gl::Enable(gl::BLEND); //Enable alpha blending
        gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA); //Set blend func to (Cs * alpha + Cd * (1.0 - alpha))
        gl::ClearColor(0.53, 0.81, 0.92, 1.0); //Set clear color. A pleasant blue
    }

    //Compile shaders
    let model_shader =
        unsafe { compile_program_from_files("model_vertex.glsl", "model_fragment.glsl") };
    let instanced_model_shader =
        unsafe { compile_program_from_files("instanced_vertex.glsl", "model_fragment.glsl") };
    let skybox_shader =
        unsafe { compile_program_from_files("skybox_vertex.glsl", "skybox_fragment.glsl") };
    let shadow_map_shader =
        unsafe { compile_program_from_files("shadow_vertex.glsl", "shadow_fragment.glsl") };
    let instanced_shadow_map_shader = unsafe {
        compile_program_from_files("instanced_shadow_vertex.glsl", "shadow_fragment.glsl")
    };
    let glyph_shader =
        unsafe { compile_program_from_files("glyph_vertex.glsl", "glyph_fragment.glsl") };
    let ui_shader = unsafe { compile_program_from_files("ui_vertex.glsl", "ui_fragment.glsl") };

    //Setup the VR rendering target
    let vr_render_target = unsafe { create_vr_render_target(&render_target_size) };
    let openvr_texture_handle = Texture {
        handle: Handle::OpenGLTexture(vr_render_target as usize),
        color_space: ColorSpace::Auto,
    };

    //Create channels for communication with the worker thread
    let (order_tx, order_rx) = mpsc::channel::<WorkOrder>();
    let (result_tx, result_rx) = mpsc::channel::<WorkResult>();

    //Spawn thread to do work
    let worker_handle = thread::spawn(move || {
        loop {
            match order_rx.recv() {
                Ok(WorkOrder::Model) => {
                    let path = file_select();

                    //Send model data back to the main thread
                    if let Some(p) = path {
                        handle_result(result_tx.send(WorkResult::Model(load_wavefront_obj(&p))));
                    }
                }
                Ok(WorkOrder::Quit) => {
                    return;
                }
                Err(e) => {
                    println!("{}", e);
                }
            }
        }
    });

    //Create the cube that will be user to render the skybox
    let (skybox_vao, skybox_indices_count) = unsafe {
        let (vertices, indices) = cube();
        (
            create_vertex_array_object(&vertices, &indices, &[3]),
            indices.len() as i32,
        )
    };

    //Create the skybox cubemap
    let skybox_cubemap = unsafe {
        let name = "siege";
        let paths = [
            &format!("textures/skybox/{}_rt.tga", name), //Right side
            &format!("textures/skybox/{}_lf.tga", name), //Left side
            &format!("textures/skybox/{}_up.tga", name), //Up side
            &format!("textures/skybox/{}_dn.tga", name), //Down side
            &format!("textures/skybox/{}_bk.tga", name), //Back side
            &format!("textures/skybox/{}_ft.tga", name), //Front side
        ];

        let mut cubemap = 0;
        gl::GenTextures(1, &mut cubemap);
        gl::BindTexture(gl::TEXTURE_CUBE_MAP, cubemap);

        //Configure texture
        gl::TexParameteri(
            gl::TEXTURE_CUBE_MAP,
            gl::TEXTURE_WRAP_S,
            gl::CLAMP_TO_EDGE as i32,
        );
        gl::TexParameteri(
            gl::TEXTURE_CUBE_MAP,
            gl::TEXTURE_WRAP_T,
            gl::CLAMP_TO_EDGE as i32,
        );
        gl::TexParameteri(
            gl::TEXTURE_CUBE_MAP,
            gl::TEXTURE_WRAP_R,
            gl::CLAMP_TO_EDGE as i32,
        );
        gl::TexParameteri(
            gl::TEXTURE_CUBE_MAP,
            gl::TEXTURE_MAG_FILTER,
            gl::LINEAR as i32,
        );
        gl::TexParameteri(
            gl::TEXTURE_CUBE_MAP,
            gl::TEXTURE_MIN_FILTER,
            gl::LINEAR as i32,
        );

        //Place each piece of the skybox on the correct face
        for i in 0..6 {
            let image_data = image_data_from_path(paths[i]);
            gl::TexImage2D(
                gl::TEXTURE_CUBE_MAP_POSITIVE_X + i as u32,
                0,
                image_data.internal_format as i32,
                image_data.width as i32,
                image_data.height as i32,
                0,
                image_data.format,
                gl::UNSIGNED_BYTE,
                &image_data.data[0] as *const u8 as *const c_void,
            );
        }
        cubemap
    };

    //OptionVec of meshes
    let mut meshes = OptionVec::with_capacity(1);

    //Create the terrain
    let terrain = {
        const SIMPLEX_SCALE: f64 = 3.0;
        const SCALE: f32 = 200.0;
        const AMPLITUDE: f32 = SCALE / 10.0;
        const WIDTH: usize = 100; //Width (and height) in vertices
        const SUBSQUARE_COUNT: usize = (WIDTH - 1) * (WIDTH - 1);

        //Set up the simplex noise generator
        let simplex_generator = {
            let seed = (SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                * 1000) as u32;
            println!("Seed used for terrain generation: {}", seed);
            OpenSimplex::new().set_seed(seed)
        };

        let surface_normals;
        unsafe {
            const ELEMENT_STRIDE: usize = 8;
            let tris: usize = SUBSQUARE_COUNT * 2;

            //Buffers to be filled
            let mut vertices = vec![0.0; WIDTH * WIDTH * ELEMENT_STRIDE];
            let mut indices: Vec<u16> = Vec::with_capacity(tris * 3);

            //Calculate the positions and tex coords for each vertex
            for i in (0..vertices.len()).step_by(ELEMENT_STRIDE) {
                let xpos: usize = (i / ELEMENT_STRIDE) % WIDTH;
                let zpos: usize = (i / ELEMENT_STRIDE) / WIDTH;

                //Calculate vertex position
                vertices[i] = (xpos as f32 / (WIDTH - 1) as f32) as f32 - 0.5;
                vertices[i + 2] = (zpos as f32 / (WIDTH - 1) as f32) as f32 - 0.5;

                //Retrieve the height from the simplex noise generator
                vertices[i + 1] = simplex_generator.get([
                    vertices[i] as f64 * SIMPLEX_SCALE,
                    vertices[i + 2] as f64 * SIMPLEX_SCALE,
                ]) as f32;

                //Calculate texture coordinates
                vertices[i + 6] = SCALE * (xpos as f32 / (WIDTH - 1) as f32) as f32;
                vertices[i + 7] = SCALE * (zpos as f32 / (WIDTH - 1) as f32) as f32;
            }

            //This loop executes once per subsquare on the plane, and pushes the indices of the two triangles that comprise said subsquare into the indices Vec
            for i in 0..SUBSQUARE_COUNT {
                let xpos = i % (WIDTH - 1);
                let ypos = i / (WIDTH - 1);

                //Push indices for bottom-left triangle
                indices.push((xpos + ypos * WIDTH) as u16);
                indices.push((xpos + ypos * WIDTH + WIDTH) as u16);
                indices.push((xpos + ypos * WIDTH + 1) as u16);

                //Push indices for top-right triangle
                indices.push((xpos + ypos * WIDTH + 1) as u16);
                indices.push((xpos + ypos * WIDTH + WIDTH) as u16);
                indices.push((xpos + ypos * WIDTH + WIDTH + 1) as u16);
            }

            //The ith vertex will be shared by each surface in vertex_surface_map[i]
            let mut vertex_surface_map = Vec::with_capacity(vertices.len() / ELEMENT_STRIDE);
            for _ in 0..(vertices.len() / ELEMENT_STRIDE) {
                vertex_surface_map.push(Vec::new());
            }

            //Calculate a normal vector for each triangle
            surface_normals = {
                let mut norms = Vec::with_capacity(indices.len() / 3);

                //This loop executes once per triangle in the mesh
                for i in (0..indices.len()).step_by(3) {
                    let mut tri_verts = [glm::zero(); 3];

                    for j in 0..3 {
                        let index = indices[i + j];
                        tri_verts[j] = glm::vec4(
                            vertices[index as usize * ELEMENT_STRIDE],
                            vertices[index as usize * ELEMENT_STRIDE + 1],
                            vertices[index as usize * ELEMENT_STRIDE + 2],
                            1.0,
                        );

                        vertex_surface_map[index as usize].push(i / 3);
                    }

                    //Vectors representing two edges of the triangle
                    let u = glm::vec4_to_vec3(&(tri_verts[0] - tri_verts[1]));
                    let v = glm::vec4_to_vec3(&(tri_verts[1] - tri_verts[2]));

                    //The cross product of two vectors on a plane must be normal to that plane
                    let norm = glm::normalize(&glm::cross::<f32, glm::U3>(&u, &v));
                    norms.push(norm);
                }
                norms
            };

            //Calculate vertex normals
            for i in (0..vertices.len()).step_by(ELEMENT_STRIDE) {
                let vertex_number = i / ELEMENT_STRIDE;

                //Calculate the vertex normal itself by averaging the normal vector of each surface it's connected to, then normalizing the result
                let mut averaged_vector: glm::TVec3<f32> = glm::zero();
                for surface_id in vertex_surface_map[vertex_number].iter() {
                    averaged_vector += surface_normals[*surface_id];
                }
                averaged_vector = glm::normalize(&averaged_vector);

                //Write this vertex normal to the proper spot in the vertices array
                for j in 0..3 {
                    vertices[i + 3 + j] = averaged_vector.data[j];
                }
            }

            println!(
                "The generated surface contains {} vertices",
                vertices.len() / ELEMENT_STRIDE
            );
            let vao = create_vertex_array_object(&vertices, &indices, &[3, 3, 2]);
            let model_matrix = glm::scaling(&glm::vec3(SCALE, AMPLITUDE, SCALE));
            let tex = {
                let tex_params = [
                    (gl::TEXTURE_WRAP_S, gl::REPEAT),
                    (gl::TEXTURE_WRAP_T, gl::REPEAT),
                    (gl::TEXTURE_MIN_FILTER, gl::LINEAR),
                    (gl::TEXTURE_MAG_FILTER, gl::LINEAR),
                ];
                load_texture("textures/grass.jpg", &tex_params)
            };
            let mut mesh = Mesh::new(
                vao,
                model_matrix,
                tex,
                vec![0, indices.len() as GLsizei],
                None,
            );
            mesh.specular_coefficient = 0.001;
            meshes.insert(mesh)
        };

        Terrain {
            surface_normals,
            simplex: simplex_generator,
            simplex_scale: SIMPLEX_SCALE,
            scale: SCALE,
            amplitude: AMPLITUDE,
            width: WIDTH,
            subsquare_count: SUBSQUARE_COUNT,
        }
    };

    //This counter is used for both trees and grass
    let mut halton_counter = 1;

    macro_rules! instanced_props {
        ($(($path:expr, $scale:expr)),*) => {
            vec![$(InstancedProp::new($path, &terrain, 200, &mut halton_counter, $scale)),*]
        };
    }

    let instanced_props = instanced_props![
        ("models/tree1.obj", 1.0),
        ("models/tree3.obj", 1.0),
        ("models/tree2.obj", 2.0)
    ];

    //Plant trees
    // let mut instanced_props = Vec::with_capacity(3);
    // instanced_props.push(InstancedProp::new(
    //     "models/tree1.obj",
    //     &terrain,
    //     200,
    //     &mut halton_counter,
    //     1.0,
    // ));
    // instanced_props.push(InstancedProp::new(
    //     "models/tree2.obj",
    //     &terrain,
    //     200,
    //     &mut halton_counter,
    //     1.0,
    // ));
    // instanced_props.push(InstancedProp::new(
    //     "models/tree3.obj",
    //     &terrain,
    //     200,
    //     &mut halton_counter,
    //     2.0,
    // ));

    //Plant grass
    const GRASS_COUNT: usize = 60000;
    let grass_texture = {
        let tex_params = [
            (gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE),
            (gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE),
            (gl::TEXTURE_MIN_FILTER, gl::LINEAR),
            (gl::TEXTURE_MAG_FILTER, gl::LINEAR),
        ];
        unsafe { load_texture("textures/billboardgrass.png", &tex_params) }
    };

    let (grass_vao, grass_indices_count) = unsafe {
        let vertices = vec![
            //Position				Normals						Tex coords
            -0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -0.5,
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
            -0.5, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, -0.5,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0,
        ];
        let indices = vec![0u16, 1, 2, 3, 2, 1, 4, 5, 6, 7, 6, 5];
        let indices_len = indices.len();
        let v = VertexArray {
            vertices,
            indices,
            attribute_offsets: vec![3, 3, 2],
        };
        let vao = instanced_prop_vao(&v, &terrain, GRASS_COUNT, &mut halton_counter, 0.85);
        (vao, indices_len as i32)
    };

    //Variables to keep track of the loaded models
    let model_texture = {
        let tex_params = [
            (gl::TEXTURE_WRAP_S, gl::REPEAT),
            (gl::TEXTURE_WRAP_T, gl::REPEAT),
            (gl::TEXTURE_MIN_FILTER, gl::LINEAR),
            (gl::TEXTURE_MAG_FILTER, gl::LINEAR),
        ];
        unsafe { load_texture("textures/checkerboard.jpg", &tex_params) }
    };
    let model_bounding_sphere_radius = 0.20;
    let mut bound_controller_indices = Vec::new();
    let mut model_indices = OptionVec::new();
    let mut model_to_controller_matrices = Vec::new();

    //Initialize the struct of arrays containing controller related state
    let mut controllers = Controllers::new();

    //Index of the HMD mesh in meshes
    let mut hmd_mesh_index = None;

    //Camera state
    let mut camera = Camera::new(glm::vec3(0.0, 1.0, 0.0));

    //Mouse state
    let mut last_lbutton_state = window.get_mouse_button(MouseButton::Button1);

    //The instant recorded at the beginning of last frame
    let mut last_frame_instant = Instant::now();

    //Tracking space position information
    let mut world_from_tracking: glm::TMat4<f32> = glm::identity();
    let mut tracking_position: glm::TVec4<f32> = glm::zero();

    let audio_device = rodio::default_output_device();

    //Play background music
    let mut is_muted = false;
    let mut bgm_path = String::from("audio/dark_ruins.mp3");
    let bgm_volume = 0.25;
    let mut bgm_sink = match &audio_device {
        Some(device) => Some(play_bgm(device, &bgm_path, bgm_volume)),
        None => {
            println!("Unable to find audio device. Music and sound effects will not play.");
            None
        }
    };

    //Flags
    let mut is_wireframe = false;
    let mut is_lighting = true;
    let mut is_flying = false;
    let mut debug_menu = false;
    let mut taking_screenshot = false;

    //Unit vector pointing towards the sun
    let sun_direction = glm::normalize(&glm::vec4(0.8, 1.0, 1.0, 0.0));

    //Create a framebuffer for the shadow map, and bind a two-dimensional texture to its depth buffer
    let shadow_map_resolution = 4096;
    let shadow_map_cascades = 3;
    let _shadow_map_cascade_cutoffs = [0.0, 10.0, 50.0];
    let (shadow_buffer, shadow_map) = unsafe {
        let mut framebuffer = 0;
        gl::GenFramebuffers(1, &mut framebuffer);

        let mut depth_texture = 0;
        gl::GenTextures(1, &mut depth_texture);
        gl::BindTexture(gl::TEXTURE_2D, depth_texture);
        gl::TexImage2D(
            gl::TEXTURE_2D,
            0,
            gl::DEPTH_COMPONENT as i32,
            shadow_map_resolution * shadow_map_cascades,
            shadow_map_resolution,
            0,
            gl::DEPTH_COMPONENT,
            gl::FLOAT,
            ptr::null(),
        );
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::REPEAT as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::REPEAT as i32);

        gl::BindFramebuffer(gl::FRAMEBUFFER, framebuffer);
        gl::FramebufferTexture2D(
            gl::FRAMEBUFFER,
            gl::DEPTH_ATTACHMENT,
            gl::TEXTURE_2D,
            depth_texture,
            0,
        );

        if gl::CheckFramebufferStatus(gl::FRAMEBUFFER) != gl::FRAMEBUFFER_COMPLETE {
            panic!("Shadow map framebuffer didn't complete");
        }
        gl::BindFramebuffer(gl::FRAMEBUFFER, 0);

        (framebuffer, depth_texture)
    };

    //Time in seconds since the start of the process
    let mut _elapsed_time = 0.0;

    //Set up menu rendering
    let mut pixel_projection = pixel_matrix(window_size); //Matrix that transforms ([0, window_width], [0, window_height]) -> ([-1, 1], [-1, 1])
    let constantia: &[u8] = include_bytes!("../fonts/Constantia.ttf"); //Binary font data
    let mut glyph_brush = GlyphBrushBuilder::using_font_bytes(constantia).build();
    let mut glyph_context =
        unsafe { GlyphContext::new(glyph_shader, glyph_brush.texture_dimensions()) }; //State for all text rendering
    let mut menu_vaos = OptionVec::new(); //TODO: For some reason I decided to give each glyph a vao instead of making them all share a single one

    //Main loop - one iteration = one frame
    while !window.should_close() {
        //Calculate time since the last frame started in seconds
        let time_delta = {
            let frame_instant = Instant::now();
            let dur = frame_instant.duration_since(last_frame_instant);
            last_frame_instant = frame_instant;

            //There's an underlying assumption here that frames will always take less than one second to complete
            (dur.subsec_millis() as f32 / 1000.0) + (dur.subsec_micros() as f32 / 1_000_000.0)
        };
        _elapsed_time += time_delta;

        //Restart music if it stopped
        if let Some(ref sink) = &bgm_sink {
            if sink.empty() {
                add_source_from_file(sink, &bgm_path);
            }
        }

        // If we haven't found the controllers, check for their presence and insert them
        if let Some(ref sys) = openvr_system {
            [
                TrackedControllerRole::LeftHand,
                TrackedControllerRole::RightHand,
            ]
            .iter()
            .enumerate()
            .take(controllers.device_indices.len())
            .for_each(|(i, role)| {
                if controllers.device_indices[i].is_none() {
                    controllers.device_indices[i] =
                        sys.tracked_device_index_for_controller_role(*role);
                }
            });
        }

        //Load controller meshes if necessary
        if controllers.mesh_indices[0].is_none() || controllers.mesh_indices[1].is_none() {
            if let Some(index) = controllers.device_indices[0] {
                if let Some(mesh) = load_openvr_mesh(&openvr_system, &openvr_rendermodels, index) {
                    controllers.mesh_indices[0] = Some(meshes.insert(mesh.clone()));
                    controllers.mesh_indices[1] = Some(meshes.insert(mesh));
                }
            }
        }

        //Load HMD mesh if we haven't already
        if hmd_mesh_index.is_none() {
            if let Some(mut mesh) = load_openvr_mesh(&openvr_system, &openvr_rendermodels, 0) {
                mesh.render_pass_visibilities = [false, false, !camera.attached_to_hmd];
                mesh.specular_coefficient = 128.0;
                hmd_mesh_index = Some(meshes.insert(mesh));
            }
        }

        //Get VR orientation and position data - OpenVR really does all the work here
        let render_poses = openvr_compositor
            .as_ref()
            .map(|comp| comp.wait_get_poses().unwrap().render);

        //Get controller state structs
        for i in 0..Controllers::NUMBER_OF_CONTROLLERS {
            if let (Some(index), Some(sys)) = (controllers.device_indices[i], &openvr_system) {
                controllers.states[i] = sys.controller_state(index);
            }
        }

        //Check if the worker thread has new results to consume
        while let Ok(work_result) = result_rx.try_recv() {
            match work_result {
                WorkResult::Model(option_mesh) => {
                    if let Some(mesh_data) = option_mesh {
                        let vao = unsafe {
                            create_vertex_array_object(
                                &mesh_data.vertex_array.vertices,
                                &mesh_data.vertex_array.indices,
                                &[3, 3, 2],
                            )
                        };
                        let mesh = Mesh::new(
                            vao,
                            uniform_scale(0.3),
                            model_texture,
                            mesh_data.geo_boundaries,
                            Some(mesh_data.materials),
                        );
                        model_indices.insert(meshes.insert(mesh));
                        bound_controller_indices.push(None);
                        model_to_controller_matrices.push(glm::identity());
                    }
                }
            }
        }

        //Handle window events and keyboard inputs
        for (_, event) in glfw::flush_messages(&events) {
            match event {
                WindowEvent::Close => {
                    window.set_should_close(true);
                }
                WindowEvent::FramebufferSize(width, height) => {
                    window_size = (width as u32, height as u32); //Record the window's size
                    aspect_ratio = window_size.0 as f32 / window_size.1 as f32; //Recalculate the window's aspect ratio
                    pixel_projection = pixel_matrix(window_size); //Recalculate the pixel projection matrix
                    println!(
                        "Window resolution updated to {}x{}",
                        window_size.0, window_size.1
                    );
                }
                WindowEvent::Key(key, _, Action::Press, ..) => match key {
                    Key::Escape => {
                        debug_menu = !debug_menu;
                    }
                    Key::W => {
                        camera.velocity += glm::vec4(0.0, 0.0, -camera.speed, 0.0);
                    }
                    Key::S => {
                        camera.velocity += glm::vec4(0.0, 0.0, camera.speed, 0.0);
                    }
                    Key::A => {
                        camera.velocity += glm::vec4(-camera.speed, 0.0, 0.0, 0.0);
                    }
                    Key::D => {
                        camera.velocity += glm::vec4(camera.speed, 0.0, 0.0, 0.0);
                    }
                    Key::I => {
                        camera.fov = 90.0;
                    }
                    Key::LeftShift => {
                        camera.speed_multiplier = 15.0;
                    }
                    Key::LeftControl => {
                        camera.speed_multiplier = 0.1;
                    }
                    _ => {}
                },
                WindowEvent::Key(key, _, Action::Release, ..) => match key {
                    Key::W => {
                        camera.velocity.z -= -camera.speed;
                    }
                    Key::S => {
                        camera.velocity.z -= camera.speed;
                    }
                    Key::A => {
                        camera.velocity.x -= -camera.speed;
                    }
                    Key::D => {
                        camera.velocity.x -= camera.speed;
                    }
                    Key::LeftShift | Key::LeftControl => {
                        camera.speed_multiplier = 1.0;
                    }
                    Key::F1 => {
                        taking_screenshot = true;
                    }
                    _ => {}
                },
                WindowEvent::MouseButton(MouseButton::Button2, Action::Release, _) => {
                    if window.get_cursor_mode() == CursorMode::Normal {
                        window.set_cursor_mode(CursorMode::Disabled);
                        window
                            .set_cursor_pos(window_size.0 as f64 / 2.0, window_size.1 as f64 / 2.0);
                    } else {
                        window.set_cursor_mode(CursorMode::Normal);
                    }
                }
                _ => {}
            }
        }

        //Handle mouse input
        let lbutton_state = window.get_mouse_button(MouseButton::Button1);
        let cursor_pos = window.get_cursor_pos();
        let cursor_delta = [
            cursor_pos.0 - window_size.0 as f64 / 2.0,
            cursor_pos.1 - window_size.1 as f64 / 2.0,
        ];

        //If the cursor is currently captured, calculate how the camera's rotation should change this frame
        if window.get_cursor_mode() == CursorMode::Disabled {
            //No idea what a good range is for this value, but this is working for now
            const MOUSE_SENSITIVITY: f32 = 0.001;
            camera.yaw += cursor_delta[0] as f32 * MOUSE_SENSITIVITY;
            camera.pitch += cursor_delta[1] as f32 * MOUSE_SENSITIVITY;

            //Prevent the camera from flipping upside down by constraining its pitch to the range [-pi/2, pi/2]
            camera.pitch = glm::clamp_scalar(camera.pitch, -glm::half_pi::<f32>(), glm::half_pi());

            //Reset cursor to center of screen
            window.set_cursor_pos(window_size.0 as f64 / 2.0, window_size.1 as f64 / 2.0);
        }

        //Handle controller input
        for i in 0..Controllers::NUMBER_OF_CONTROLLERS {
            if let (Some(device_index), Some(mesh_index), Some(state), Some(_sys), Some(poses)) = (
                controllers.device_indices[i],
                controllers.mesh_indices[i],
                controllers.states[i],
                &openvr_system,
                render_poses,
            ) {
                for j in 0..model_indices.len() {
                    if let Some(loaded_index) = model_indices[j] {
                        let is_colliding = glm::distance(
                            &get_mesh_origin(&meshes[mesh_index]),
                            &get_mesh_origin(&meshes[loaded_index]),
                        ) < model_bounding_sphere_radius;
                        if controllers.pressed_this_frame(i, button_id::GRIP) && is_colliding {
                            //Set the controller's mesh as the mesh the cube mesh is "bound" to
                            bound_controller_indices[j] = Some(i);

                            //Calculate the loaded_mesh-space to controller-space matrix aka inverse(controller.model_matrix) * loaded_mesh.model_matrix
                            if let (Some(cont_mesh), Some(loaded_mesh)) =
                                (&meshes[mesh_index], &meshes[loaded_index])
                            {
                                model_to_controller_matrices[j] =
                                    glm::affine_inverse(cont_mesh.model_matrix)
                                        * loaded_mesh.model_matrix;
                            }
                        }
                    }

                    //If the trigger was released this frame
                    if controllers.released_this_frame(i, button_id::GRIP) {
                        if Some(i) == bound_controller_indices[j] {
                            bound_controller_indices[j] = None;
                        }
                    }
                }

                let yvel = if i == 0 {
                    glm::vec4(
                        0.0,
                        -glm::clamp_scalar(state.axis[1].x * 4.0, 0.0, 1.0),
                        0.0,
                        0.0,
                    )
                } else {
                    glm::vec4(
                        0.0,
                        glm::clamp_scalar(state.axis[1].x * 4.0, 0.0, 1.0),
                        0.0,
                        0.0,
                    )
                };

                let mut movement_vector = yvel;

                //Handle left-hand controls
                let mut sprint_multiplier = 1.0;
                if i == 0 {
                    let tracking_from_controller =
                        openvr_to_mat4(*poses[device_index as usize].device_to_absolute_tracking());

                    //We check to make sure at least one axis isn't zero in order to ensure no division by zero
                    if state.axis[0].x != 0.0 || state.axis[0].y != 0.0 {
                        let mut temp = world_from_tracking
                            * tracking_from_controller
                            * glm::vec4(state.axis[0].x, 0.0, -state.axis[0].y, 0.0);
                        let len = glm::length(&temp);
                        temp.y = 0.0;
                        temp *= len / glm::length(&temp);
                        movement_vector += temp;
                    }

                    if controllers.holding_button(i, button_id::STEAM_VR_TOUCHPAD) {
                        sprint_multiplier = 5.0;
                    }
                }

                //Check if the user toggled flight
                if i == 1 && controllers.released_this_frame(i, button_id::APPLICATION_MENU) {
                    is_flying = !is_flying;
                }

                //Update the position of tracking space
                tracking_position += movement_vector * sprint_multiplier * time_delta;

                //If the player isn't flying, their height is just the terrain height
                if !is_flying {
                    tracking_position.y =
                        terrain.height_at(tracking_position.x, tracking_position.z);
                }
            }
            world_from_tracking = glm::translation(&glm::vec4_to_vec3(&tracking_position));
        }
        controllers.previous_states = controllers.states;

        //Get view matrices (Left eye, Right eye, Companion window)
        let v_matrices = match (&openvr_system, &render_poses) {
            (Some(sys), Some(poses)) => {
                let hmd_to_tracking = openvr_to_mat4(*poses[0].device_to_absolute_tracking());
                let left_eye_to_hmd = openvr_to_mat4(sys.eye_to_head_transform(Eye::Left));
                let right_eye_to_hmd = openvr_to_mat4(sys.eye_to_head_transform(Eye::Right));

                let companion_v_mat = if camera.attached_to_hmd {
                    glm::affine_inverse(world_from_tracking * hmd_to_tracking)
                } else {
                    camera.view_matrix()
                };

                //Need to return inverse(world_from_tracking * hmd_to_tracking * eye_to_hmd)
                [
                    glm::affine_inverse(world_from_tracking * hmd_to_tracking * left_eye_to_hmd),
                    glm::affine_inverse(world_from_tracking * hmd_to_tracking * right_eye_to_hmd),
                    companion_v_mat,
                ]
            }
            _ => [glm::identity(), glm::identity(), camera.view_matrix()],
        };

        //Get projection matrices
        let p_mat = glm::perspective(aspect_ratio, f32::to_radians(camera.fov), NEAR_Z, FAR_Z);
        let p_matrices = match &openvr_system {
            Some(sys) => [
                get_projection_matrix(sys, Eye::Left),
                get_projection_matrix(sys, Eye::Right),
                p_mat,
            ],
            None => [glm::identity(), glm::identity(), p_mat],
        };

        //Update the camera's position
        camera.position += time_delta
            * glm::affine_inverse(v_matrices[2])
            * camera.speed_multiplier
            * camera.velocity; //The process here is, for the camera's velocity vector: View Space -> World Space -> scale by seconds_elapsed

        //Update the OpenVR meshes
        if let Some(poses) = render_poses {
            update_openvr_mesh(&mut meshes, &poses, &world_from_tracking, 0, hmd_mesh_index);
            for i in 0..Controllers::NUMBER_OF_CONTROLLERS {
                if let Some(index) = &controllers.device_indices[i] {
                    update_openvr_mesh(
                        &mut meshes,
                        &poses,
                        &world_from_tracking,
                        *index as usize,
                        controllers.mesh_indices[i],
                    );
                }
            }
        }

        //If a model is being grabbed, update its model matrix
        for i in 0..bound_controller_indices.len() {
            if let Some(index) = bound_controller_indices[i] {
                if let (Some(mesh_index), Some(load_index)) =
                    (controllers.mesh_indices[index], model_indices[i])
                {
                    if let (Some(loaded), Some(controller)) =
                        meshes.two_mut_refs(load_index, mesh_index)
                    {
                        loaded.model_matrix =
                            controller.model_matrix * model_to_controller_matrices[i];
                    }
                }
            }
        }

        //Debug menu data
        let menu_items = [
            "Wireframe",
            "Lighting",
            "Mute",
            "Swap bgm",
            "Toggle camera mode",
            "Spawn model",
        ];
        let menu_commands = [
            Command::ToggleWireframe,
            Command::ToggleLighting,
            Command::ToggleMusic,
            Command::SwitchMusic,
            Command::ToggleFreecam,
            Command::SpawnModel,
        ];
        let x_buffer = 32.0;
        let y_buffer = 32.0;
        let mut y_offset = 18.0;
        let scale = 36.0;
        let border = 10.0;

        //Debug menu update
        if debug_menu {
            for i in 0..menu_items.len() {
                //Create a section, positioning it based on the text scale and the number of buttons already created
                let section = Section {
                    text: menu_items[i],
                    screen_position: (x_buffer, border + y_offset),
                    scale: Scale::uniform(scale),
                    color: [1.0, 1.0, 1.0, 1.0],
                    ..Section::default()
                };
                glyph_brush.queue(section);
                y_offset += y_buffer + scale;

                let bounding_box = match glyph_brush.glyph_bounds(section) {
                    Some(rect) => rect,
                    None => {
                        continue;
                    }
                };

                let left = bounding_box.min.x - border;
                let right = bounding_box.max.x + border;
                let top = bounding_box.min.y - border;
                let bottom = bounding_box.max.y + border;

                let color = if (cursor_pos.0 as f32) > left
                    && (cursor_pos.0 as f32) < right
                    && (cursor_pos.1 as f32) < bottom
                    && (cursor_pos.1 as f32) > top
                {
                    if lbutton_state == Action::Press {
                        [0.0, 0.5, 0.0]
                    } else {
                        if last_lbutton_state == Action::Press {
                            match menu_commands[i] {
                                Command::ToggleWireframe => {
                                    is_wireframe = !is_wireframe;
                                }
                                Command::ToggleLighting => {
                                    is_lighting = !is_lighting;
                                }
                                Command::ToggleMusic => {
                                    if let Some(sink) = &mut bgm_sink {
                                        sink.set_volume(bgm_volume * is_muted as u32 as f32);
                                        is_muted = !is_muted;
                                    }
                                }
                                Command::ToggleFreecam => {
                                    camera.attached_to_hmd = !camera.attached_to_hmd;
                                    if let Some(mesh) = meshes.get_element(hmd_mesh_index) {
                                        mesh.render_pass_visibilities[2] = !camera.attached_to_hmd;
                                    }
                                }
                                Command::SpawnModel => {
                                    handle_result(order_tx.send(WorkOrder::Model));
                                }
                                Command::SwitchMusic => {
                                    if let Some(sink) = &mut bgm_sink {
                                        if let Some(filename) = file_select() {
                                            println!("Playing new background music: {}", filename);
                                            bgm_path = filename;
                                            drop(sink);
                                            if let Some(device) = &audio_device {
                                                bgm_sink =
                                                    Some(play_bgm(device, &bgm_path, bgm_volume));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        [0.0, 0.1, 0.0]
                    }
                } else {
                    [0.0, 0.0, 0.0]
                };

                let vertices = [
                    left, bottom, color[0], color[1], color[2], right, bottom, color[0], color[1],
                    color[2], left, top, color[0], color[1], color[2], right, top, color[0],
                    color[1], color[2],
                ];
                let indices = [0u16, 1, 2, 3, 2, 1];
                //TODO: Avoid recreating the vaos every frame
                menu_vaos
                    .insert(unsafe { create_vertex_array_object(&vertices, &indices, &[2, 3]) });
            }
        }

        //Process the glyphs to be rendered this frame
        unsafe {
            match glyph_brush.process_queued(
                |rect, data| {
                    gl::TextureSubImage2D(
                        glyph_context.texture,
                        0,
                        rect.min.x as _,
                        rect.min.y as _,
                        rect.width() as _,
                        rect.height() as _,
                        gl::RED,
                        gl::UNSIGNED_BYTE,
                        data.as_ptr() as _,
                    );
                },
                |vertex| {
                    let left = vertex.pixel_coords.min.x as f32;
                    let right = vertex.pixel_coords.max.x as f32;
                    let top = vertex.pixel_coords.min.y as f32;
                    let bottom = vertex.pixel_coords.max.y as f32;
                    let texleft = vertex.tex_coords.min.x;
                    let texright = vertex.tex_coords.max.x;
                    let textop = vertex.tex_coords.min.y;
                    let texbottom = vertex.tex_coords.max.y;
                    let z = vertex.z;
                    let color = vertex.color;

                    //We need to return four vertices
                    [
                        left, bottom, z, color[0], color[1], color[2], texleft, texbottom, right,
                        bottom, z, color[0], color[1], color[2], texright, texbottom, left, top, z,
                        color[0], color[1], color[2], texleft, textop, right, top, z, color[0],
                        color[1], color[2], texright, textop,
                    ]
                },
            ) {
                Ok(BrushAction::Draw(verts)) => {
                    if !verts.is_empty() {
                        let mut buffer = Vec::with_capacity(verts.len() * 32);
                        for vert in &verts {
                            for v in vert {
                                buffer.push(*v);
                            }
                        }
                        glyph_context.count = verts.len();

                        let mut indices = vec![0; glyph_context.count * 6];
                        for i in 0..glyph_context.count {
                            indices[i * 6] = 4 * i as u16;
                            indices[i * 6 + 1] = indices[i * 6] + 1;
                            indices[i * 6 + 2] = indices[i * 6] + 2;
                            indices[i * 6 + 3] = indices[i * 6] + 3;
                            indices[i * 6 + 4] = indices[i * 6] + 2;
                            indices[i * 6 + 5] = indices[i * 6] + 1;
                        }

                        gl::DeleteVertexArrays(1, &glyph_context.vao);
                        glyph_context.vao =
                            create_vertex_array_object(&buffer, &indices, &[3, 3, 2]);
                    } else {
                        gl::DeleteVertexArrays(1, &glyph_context.vao);
                        glyph_context.vao = create_vertex_array_object(&[0.0], &[0], &[]);
                    }
                }
                Ok(BrushAction::ReDraw) => {}
                Err(BrushError::TextureTooSmall { .. }) => {
                    println!("Need to resize the glyph texture");
                }
            }
        }

        //----------------Rendering code----------------

        //Set up data for each render pass
        let framebuffers = [vr_render_target, vr_render_target, 0];
        let eyes = [Some(Eye::Left), Some(Eye::Right), None];
        let sizes = [render_target_size, render_target_size, window_size];

        //Calculate the shadow projection volume
        //TODO: This is the most crude shadowing solution possible; I need to implement cascaded shadow maps
        let shadow_view = glm::look_at(
            &glm::vec4_to_vec3(&sun_direction),
            &glm::vec3(0.0, 0.0, 0.0),
            &glm::vec3(0.0, 1.0, 0.0),
        );
        let shadow_viewprojection = {
            let size = 5.0;
            glm::ortho(-size, size, -size, size, -size * 2.0, size * 2.0) * shadow_view
        };

        let render_context = RenderContext::new(
            &p_matrices,
            &v_matrices,
            &sun_direction,
            shadow_map,
            &shadow_viewprojection,
            is_lighting,
        );
        unsafe {
            //Set polygon mode
            if is_wireframe {
                gl::PolygonMode(gl::FRONT_AND_BACK, gl::LINE);
            } else {
                gl::PolygonMode(gl::FRONT_AND_BACK, gl::FILL);
            }

            //Render the shadow map
            gl::BindFramebuffer(gl::FRAMEBUFFER, shadow_buffer);
            gl::Viewport(0, 0, shadow_map_resolution, shadow_map_resolution);
            gl::DrawBuffer(gl::NONE);
            gl::ReadBuffer(gl::NONE);
            gl::Clear(gl::DEPTH_BUFFER_BIT);

            //Render meshes into shadow map
            gl::Enable(gl::CULL_FACE);
            gl::UseProgram(shadow_map_shader);
            gl::Uniform1i(uniform_location(shadow_map_shader, "using_texture"), 0);
            meshes.iter().flatten().for_each(|mesh| {
                let mvp = shadow_viewprojection * mesh.model_matrix;
                gl::UniformMatrix4fv(
                    uniform_location(shadow_map_shader, "shadowMVP"),
                    1,
                    gl::FALSE,
                    &flatten_glm(&mvp) as *const GLfloat,
                );
                gl::BindVertexArray(mesh.vao);
                for i in 0..mesh.geo_boundaries.len() - 1 {
                    gl::DrawElements(
                        gl::TRIANGLES,
                        mesh.geo_boundaries[i + 1] - mesh.geo_boundaries[i],
                        gl::UNSIGNED_SHORT,
                        (mem::size_of::<GLshort>() as i32 * mesh.geo_boundaries[i])
                            as *const c_void,
                    );
                }
            });

            //Rendering instanced meshes
            gl::UseProgram(instanced_shadow_map_shader);
            gl::UniformMatrix4fv(
                uniform_location(instanced_shadow_map_shader, "shadowVP"),
                1,
                gl::FALSE,
                &flatten_glm(&shadow_viewprojection) as *const GLfloat,
            );

            //Render instanced propss into shadow map
            for prop in instanced_props.iter() {
                gl::BindVertexArray(prop.vao);
                gl::Uniform1i(
                    uniform_location(instanced_shadow_map_shader, "using_texture"),
                    0,
                );
                for i in 0..prop.geo_boundaries.len() - 1 {
                    gl::DrawElementsInstanced(
                        gl::TRIANGLES,
                        prop.geo_boundaries[i + 1] - prop.geo_boundaries[i],
                        gl::UNSIGNED_SHORT,
                        (mem::size_of::<GLshort>() as i32 * prop.geo_boundaries[i])
                            as *const c_void,
                        prop.instances as GLsizei,
                    );
                }
            }

            //Render instanced grass into shadow map
            gl::Disable(gl::CULL_FACE);
            gl::BindVertexArray(grass_vao);
            gl::Uniform1i(
                uniform_location(instanced_shadow_map_shader, "using_texture"),
                1,
            );
            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, grass_texture);
            gl::DrawElementsInstanced(
                gl::TRIANGLES,
                grass_indices_count,
                gl::UNSIGNED_SHORT,
                ptr::null(),
                GRASS_COUNT as GLsizei,
            );

            //Turn the color buffer back on now that we're done rendering the shadow map
            gl::DrawBuffer(gl::BACK);

            //Render the output framebuffers (left eye, right eye, companion window)
            for i in 0..framebuffers.len() {
                //Set up render target
                gl::BindFramebuffer(gl::FRAMEBUFFER, framebuffers[i]);
                gl::Viewport(0, 0, sizes[i].0 as GLsizei, sizes[i].1 as GLsizei);

                //Clear the framebuffer's color buffer and depth buffer
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

                //Render the regular meshes
                gl::Enable(gl::CULL_FACE);
                gl::UseProgram(model_shader);
                gl::Uniform1i(uniform_location(model_shader, "tex"), 0);
                gl::Uniform1i(uniform_location(model_shader, "shadow_map"), 1);
                meshes.iter().flatten().for_each(|mesh| {
                    if mesh.render_pass_visibilities[i] {
                        //Calculate model-view-projection for this mesh
                        let mvp = render_context.p_matrices[i]
                            * render_context.v_matrices[i]
                            * mesh.model_matrix;

                        bind_uniforms(
                            model_shader,
                            &["mvp", "model_matrix", "shadow_mvp"],
                            &[
                                &mvp,
                                &mesh.model_matrix,
                                &(render_context.shadow_vp * mesh.model_matrix),
                            ],
                            &["view_position", "light_direction"],
                            &[
                                &render_context.view_positions[i],
                                render_context.light_direction,
                            ],
                            &["lighting"],
                            &[render_context.is_lighting as GLint],
                        );

                        //Bind the textures
                        gl::ActiveTexture(gl::TEXTURE0);
                        gl::BindTexture(gl::TEXTURE_2D, mesh.texture);
                        gl::ActiveTexture(gl::TEXTURE1);
                        gl::BindTexture(gl::TEXTURE_2D, render_context.shadow_map);

                        //Bind the mesh's vertex array object
                        gl::BindVertexArray(mesh.vao);

                        //Check if we're using a material or just a texture
                        match &mesh.materials {
                            Some(mats) => {
                                gl::Uniform1i(
                                    uniform_location(model_shader, "using_material"),
                                    true as i32,
                                );

                                //Draw calls
                                mats.iter()
                                    .enumerate()
                                    .take(mesh.geo_boundaries.len() - 1)
                                    .for_each(|(i, mat)| {
                                        bind_material(model_shader, mat);
                                        gl::DrawElements(
                                            gl::TRIANGLES,
                                            mesh.geo_boundaries[i + 1] - mesh.geo_boundaries[i],
                                            gl::UNSIGNED_SHORT,
                                            (2 * mesh.geo_boundaries[i]) as *const c_void,
                                        );
                                    });
                            }
                            None => {
                                gl::Uniform1i(
                                    uniform_location(model_shader, "using_material"),
                                    false as i32,
                                );
                                gl::Uniform1f(
                                    uniform_location(model_shader, "specular_coefficient"),
                                    mesh.specular_coefficient,
                                );

                                //Draw call
                                for i in 0..mesh.geo_boundaries.len() - 1 {
                                    gl::DrawElements(
                                        gl::TRIANGLES,
                                        mesh.geo_boundaries[i + 1] - mesh.geo_boundaries[i],
                                        gl::UNSIGNED_SHORT,
                                        (2 * mesh.geo_boundaries[i]) as *const c_void,
                                    );
                                }
                            }
                        }
                    }
                });

                //Render the trees with instanced rendering
                gl::UseProgram(instanced_model_shader);
                bind_uniforms(
                    instanced_model_shader,
                    &["view_projection", "shadow_vp"],
                    &[&(p_matrices[i] * v_matrices[i]), &shadow_viewprojection],
                    &["view_position", "light_direction"],
                    &[&render_context.view_positions[i], &sun_direction],
                    &[
                        "using_material",
                        "lighting",
                        "shadow_map",
                        "tex",
                        "shadow_map",
                    ],
                    &[1, is_lighting as GLint, 0, 0, 1],
                );
                gl::ActiveTexture(gl::TEXTURE1);
                gl::BindTexture(gl::TEXTURE_2D, shadow_map);

                //Draw calls
                for prop in instanced_props.iter() {
                    gl::BindVertexArray(prop.vao);
                    for j in 0..prop.geo_boundaries.len() - 1 {
                        bind_material(instanced_model_shader, &prop.materials[j]);
                        gl::DrawElementsInstanced(
                            gl::TRIANGLES,
                            prop.geo_boundaries[j + 1] - prop.geo_boundaries[j],
                            gl::UNSIGNED_SHORT,
                            (mem::size_of::<GLshort>() as i32 * prop.geo_boundaries[j])
                                as *const c_void,
                            prop.instances as GLsizei,
                        );
                    }
                }

                //Render the grass billboards with instanced rendering
                gl::ActiveTexture(gl::TEXTURE0);
                gl::BindTexture(gl::TEXTURE_2D, grass_texture);
                gl::Uniform1i(
                    uniform_location(instanced_model_shader, "using_material"),
                    0,
                );
                gl::BindVertexArray(grass_vao);
                gl::Disable(gl::CULL_FACE); //Disable backface culling before the draw call because we want the grass to be double-sided
                gl::DrawElementsInstanced(
                    gl::TRIANGLES,
                    grass_indices_count,
                    gl::UNSIGNED_SHORT,
                    ptr::null(),
                    GRASS_COUNT as GLsizei,
                );

                //Draw the skybox last to take advantage of early depth testing
                //Don't draw the skybox in wireframe mode
                if !is_wireframe {
                    //Compute the view-projection matrix for the skybox (the conversion functions are just there to nullify the translation component of the view matrix)
                    //The skybox vertices should obviously be rotated along with the camera, but they shouldn't be translated in order to maintain the illusion
                    //that the sky is infinitely far away
                    let skybox_view_projection =
                        p_matrices[i] * glm::mat3_to_mat4(&glm::mat4_to_mat3(&v_matrices[i]));

                    //Render the skybox
                    gl::UseProgram(skybox_shader);
                    gl::UniformMatrix4fv(
                        uniform_location(skybox_shader, "view_projection"),
                        1,
                        gl::FALSE,
                        &flatten_glm(&skybox_view_projection) as *const GLfloat,
                    );
                    gl::BindTexture(gl::TEXTURE_CUBE_MAP, skybox_cubemap);
                    gl::BindVertexArray(skybox_vao);
                    gl::DrawElements(
                        gl::TRIANGLES,
                        skybox_indices_count,
                        gl::UNSIGNED_SHORT,
                        ptr::null(),
                    );
                }

                //Clearing the depth buffer here so that none of the text can end up behind anything
                gl::Clear(gl::DEPTH_BUFFER_BIT);

                //If we're rendering to the companion window
                if i == 2 {
                    //Draw the buttons
                    gl::PolygonMode(gl::FRONT_AND_BACK, gl::FILL);
                    gl::UseProgram(ui_shader);
                    bind_matrix4(ui_shader, "projection", &pixel_projection);
                    menu_vaos.iter().flatten().for_each(|vao| {
                        gl::BindVertexArray(*vao);
                        gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_SHORT, ptr::null());
                        gl::DeleteVertexArrays(1, vao);
                    });
                    menu_vaos.clear();

                    //Draw the glyphs
                    glyph_context.render_glyphs(&pixel_projection);

                    if taking_screenshot {
                        let mut buffer = vec![0u8; (window_size.0 * window_size.1) as usize * 4];
                        gl::ReadPixels(
                            0,
                            0,
                            window_size.0 as GLint,
                            window_size.1 as GLint,
                            gl::RGBA,
                            gl::UNSIGNED_BYTE,
                            buffer.as_mut_slice() as *mut [u8] as *mut c_void,
                        );

                        let dynamic_image =
                            match ImageBuffer::from_raw(window_size.0, window_size.1, buffer) {
                                Some(im) => Some(DynamicImage::ImageRgba8(im).flipv()),
                                None => {
                                    println!("Unable to convert raw to image::DynamicImage");
                                    None
                                }
                            };

                        if let Some(dyn_image) = dynamic_image {
                            //Create the screenshot directory if there isn't one
                            let screenshot_dir = "screenshots";
                            if !Path::new(screenshot_dir).is_dir() {
                                if let Err(e) = fs::create_dir(screenshot_dir) {
                                    println!("Unable to create screenshot directory: {}", e);
                                }
                            }

                            if let Err(e) = dyn_image.save(format!(
                                "{}/{}.png",
                                screenshot_dir,
                                Local::now().format("%F_%H%M%S")
                            )) {
                                println!("Error taking screenshot: {}", e);
                            }
                        }

                        taking_screenshot = false;
                    }
                }

                //Submit render to HMD
                submit_to_hmd(eyes[i], &openvr_compositor, &openvr_texture_handle);
            }
        }

        last_lbutton_state = lbutton_state;

        window.render_context().swap_buffers();
        glfw.poll_events();
    }

    //Shut down the worker thread
    handle_result(order_tx.send(WorkOrder::Quit));
    worker_handle.join().unwrap();

    //Shut down OpenVR
    if let Some(ctxt) = openvr_context {
        unsafe {
            ctxt.shutdown();
        }
    }
}
