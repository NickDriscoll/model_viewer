extern crate gl;
extern crate nalgebra_glm as glm;
use glfw::{Action, Context, CursorMode, Key, MouseButton, WindowMode, WindowEvent};
use openvr::{ApplicationType, button_id, Eye, TrackedControllerRole};
use openvr::compositor::texture::{ColorSpace, Handle, Texture};
use nfd::Response;
use std::collections::HashMap;
use std::fs::{File, read_to_string};
use std::io::BufReader;
use std::os::raw::c_void;
use std::{mem, ptr, thread};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::sync::mpsc;
use wavefront_obj::{mtl, obj};
use noise::{NoiseFn, OpenSimplex, Seedable};
use crate::structs::*;
use crate::glutil::*;
use crate::routines::*;
use self::gl::types::*;

//Including the other source files
mod structs;
mod glutil;
mod routines;

//The distances of the near and far clipping planes from the origin
const NEAR_Z: f32 = 0.1;
const FAR_Z: f32 = 800.0;

//Left eye, Right eye, Companion window
const RENDER_PASSES: usize = 3;

//Things you can request the worker thread to do
enum WorkOrder {
	Model,
	Quit
}

//Things the worker thread can send back to the main thread
enum WorkResult {
	Model(Option<MeshData>)
}

fn main() {
	//Initialize OpenVR
	let (openvr_context, openvr_system, openvr_compositor, openvr_rendermodels) = unsafe {
		match openvr::init(ApplicationType::Scene) {
			Ok(ctxt) => {
				let system = ctxt.system().unwrap();
				let compositor = ctxt.compositor().unwrap();
				let render_models = ctxt.render_models().unwrap();
				(Some(ctxt), Some(system), Some(compositor), Some(render_models))
			}
			Err(e) => {
				println!("OpenVR initialization error: {}", e);
				(None, None, None, None)
			}
		}
	};

	//Calculate VR render target size
	let render_target_size = match &openvr_system {
		Some(sys) => { sys.recommended_render_target_size() }
		None => { (1280, 720) }
	};

	//Init glfw
	let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

	//Using OpenGL 3.3 core, but that could change
	glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
	glfw.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));

	//Create window
	let mut window_size = (1280, 720);
	let (mut window, events) = glfw.create_window(window_size.0, window_size.1, "Model viewer", WindowMode::Windowed).unwrap();

	//Calculate window's aspect ratio
	let mut aspect_ratio = window_size.0 as f32 / window_size.1 as f32;

	//Configure what kinds of events the window will emit
	window.set_key_polling(true);
	window.set_framebuffer_size_polling(true);

	//Load all OpenGL function pointers
	gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);

	//Compile shaders
	let model_shader = unsafe { compile_program_from_files("shaders/model_vertex.glsl", "shaders/model_fragment.glsl") };
	let instanced_model_shader = unsafe { compile_program_from_files("shaders/instanced_vertex.glsl", "shaders/model_fragment.glsl") };
	let skybox_shader = unsafe { compile_program_from_files("shaders/skybox_vertex.glsl", "shaders/skybox_fragment.glsl") };
	let shadow_map_shader = unsafe { compile_program_from_files("shaders/shadow_vertex.glsl", "shaders/shadow_fragment.glsl") };
	let instanced_shadow_map_shader = unsafe { compile_program_from_files("shaders/instanced_shadow_vertex.glsl", "shaders/shadow_fragment.glsl") };

	//Setup the VR rendering target
	let vr_render_target = unsafe { create_vr_render_target(&render_target_size) };
	let openvr_texture_handle = Texture {
		handle: Handle::OpenGLTexture(vr_render_target as usize),
		color_space: ColorSpace::Auto
	};

	//Create channels for communication with the worker thread
	let (order_tx, order_rx) = mpsc::channel::<WorkOrder>();
	let (result_tx, result_rx) = mpsc::channel::<WorkResult>();

	//Spawn thread to do work
	let worker_handle = thread::spawn(move || {
		loop {
			match order_rx.recv() {
				Ok(WorkOrder::Model) => {
					//Invoke file selection dialogue
					let path = match nfd::open_file_dialog(None, None).unwrap() {
						Response::Okay(filename) => { Some(filename) }
						_ => { None }
					};

					//Send model data back to the main thread
					if let Some(p) = path {
						handle_result(result_tx.send(WorkResult::Model(load_wavefront_obj(&p))));
					}					
				}
				Ok(WorkOrder::Quit) => { return; }
				Err(e) => {
					println!("{}", e);
				}
			}
		}
	});

	//Create the cube that will be user to render the skybox
	let skybox_vao = unsafe {
		let vertices = [
			-1.0, -1.0, -1.0,
			1.0, -1.0, -1.0,
			-1.0, 1.0, -1.0,
			1.0, 1.0, -1.0,
			-1.0, -1.0, 1.0,
			-1.0, 1.0, 1.0,
			1.0, -1.0, 1.0,
			1.0, 1.0, 1.0
		];
		let indices = [
			//Front
			0u16, 1, 2,
			3, 2, 1,

			//Left
			0, 2, 4,
			2, 5, 4,

			//Right
			3, 1, 6,
			7, 3, 6,

			//Back
			5, 7, 4,
			7, 6, 4,

			//Bottom
			4, 1, 0,
			4, 6, 1,

			//Top
			7, 5, 2,
			7, 2, 3
		];

		create_vertex_array_object(&vertices, &indices, &[3])
	};

	//Create the skybox cubemap
	let skybox_cubemap = unsafe {
		let mut cubemap = 0;
		gl::GenTextures(1, &mut cubemap);
		gl::BindTexture(gl::TEXTURE_CUBE_MAP, cubemap);
		
		//Configure texture
		gl::TexParameteri(gl::TEXTURE_CUBE_MAP, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
		gl::TexParameteri(gl::TEXTURE_CUBE_MAP, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
		gl::TexParameteri(gl::TEXTURE_CUBE_MAP, gl::TEXTURE_WRAP_R, gl::CLAMP_TO_EDGE as i32);
		gl::TexParameteri(gl::TEXTURE_CUBE_MAP, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
		gl::TexParameteri(gl::TEXTURE_CUBE_MAP, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
		
		const PATHS: [&str; 6] = ["textures/skybox/totality_rt.tga",
								  "textures/skybox/totality_lf.tga",
								  "textures/skybox/totality_up.tga",
								  "textures/skybox/totality_dn.tga",
								  "textures/skybox/totality_bk.tga",
								  "textures/skybox/totality_ft.tga"];

		//Place each piece of the skybox on the correct face
		for i in 0..6 {
			let image_data = image_data_from_path(PATHS[i]);
			gl::TexImage2D(gl::TEXTURE_CUBE_MAP_POSITIVE_X + i as u32,
						   0,
						   image_data.3 as i32,
						   image_data.1 as i32,
						   image_data.2 as i32,
						   0,
						   image_data.3,
						   gl::UNSIGNED_BYTE,
				  		   &image_data.0[0] as *const u8 as *const c_void);
		}
		cubemap
	};

	//OptionVec of meshes
	let mut meshes = OptionVec::with_capacity(10);

	//Set up the simplex noise generator
	let simplex_generator = {
		let seed = (SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() * 1000) as u32;
		println!("Seed used for terrain generation: {}", seed);
		OpenSimplex::new().set_seed(seed)
	};

	//Create the large tessellated surface
	const SIMPLEX_SCALE: f64 = 3.0;
	const TERRAIN_SCALE: f32 = 200.0;
	const TERRAIN_AMPLITUDE: f32 = TERRAIN_SCALE / 10.0;
	const TERRAIN_WIDTH: usize = 100; //Width (and height) in vertices
	const SUBSQUARE_COUNT: usize = (TERRAIN_WIDTH-1)*(TERRAIN_WIDTH-1);
	let surface_normals;
	unsafe {
		const ELEMENT_STRIDE: usize = 8;
		const TRIS: usize = (TERRAIN_WIDTH - 1) * (TERRAIN_WIDTH - 1) * 2;

		//Buffers to be filled
		let mut vertices = vec![0.0; TERRAIN_WIDTH * TERRAIN_WIDTH * ELEMENT_STRIDE];
		let mut indices: Vec<u16> = Vec::with_capacity(TRIS * 3);

		//Calculate the positions and tex coords for each vertex
		for i in (0..vertices.len()).step_by(ELEMENT_STRIDE) {
			let xpos: usize = (i / ELEMENT_STRIDE) % TERRAIN_WIDTH;
			let zpos: usize = (i / ELEMENT_STRIDE) / TERRAIN_WIDTH;

			//Calculate vertex position
			vertices[i] = (xpos as f32 / (TERRAIN_WIDTH - 1) as f32) as f32 - 0.5;
			vertices[i + 2] = (zpos as f32 / (TERRAIN_WIDTH - 1) as f32) as f32 - 0.5;

			//Retrieve the height from the simplex noise generator
			vertices[i + 1] = simplex_generator.get([vertices[i] as f64 * SIMPLEX_SCALE, vertices[i + 2] as f64 * SIMPLEX_SCALE]) as f32;

			//Calculate texture coordinates
			vertices[i + 6] = TERRAIN_SCALE * (xpos as f32 / (TERRAIN_WIDTH - 1) as f32) as f32;
			vertices[i + 7] = TERRAIN_SCALE * (zpos as f32 / (TERRAIN_WIDTH - 1) as f32) as f32;
		}

		//This loop executes once per subsquare on the plane, and pushes the indices of the two triangles that comprise said subsquare into the indices Vec
		for i in 0..SUBSQUARE_COUNT {
			let xpos = i % (TERRAIN_WIDTH-1);
			let ypos = i / (TERRAIN_WIDTH-1);

			//Push indices for bottom-left triangle
			indices.push((xpos + ypos * TERRAIN_WIDTH) as u16);
			indices.push((xpos + ypos * TERRAIN_WIDTH + TERRAIN_WIDTH) as u16);
			indices.push((xpos + ypos * TERRAIN_WIDTH + 1) as u16);
			
			//Push indices for top-right triangle
			indices.push((xpos + ypos * TERRAIN_WIDTH + 1) as u16);
			indices.push((xpos + ypos * TERRAIN_WIDTH + TERRAIN_WIDTH) as u16);
			indices.push((xpos + ypos * TERRAIN_WIDTH + TERRAIN_WIDTH + 1) as u16);
		}

		//The ith vertex will be shared by each surface in vertex_surface_map[i]
		let mut vertex_surface_map = Vec::with_capacity(vertices.len() / ELEMENT_STRIDE);
		for _ in 0..(vertices.len() / ELEMENT_STRIDE) {
			vertex_surface_map.push(Vec::new());
		}

		//Calculate surface normals
		surface_normals = {
			const INDICES_PER_TRIANGLE: usize = 3;
			let mut norms = Vec::with_capacity(indices.len() / INDICES_PER_TRIANGLE);

			//This loop executes once per triangle in the mesh
			for i in (0..indices.len()).step_by(INDICES_PER_TRIANGLE) {
				let mut tri_verts = [glm::zero(); INDICES_PER_TRIANGLE];

				for j in 0..INDICES_PER_TRIANGLE {
					let index = indices[i + j];
					tri_verts[j] = glm::vec4(vertices[index as usize * ELEMENT_STRIDE],
											 vertices[index as usize * ELEMENT_STRIDE + 1],
											 vertices[index as usize * ELEMENT_STRIDE + 2],
											 1.0);

					vertex_surface_map[index as usize].push(i / INDICES_PER_TRIANGLE);
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

		println!("The generated surface contains {} vertices", vertices.len() / ELEMENT_STRIDE);
		let vao = create_vertex_array_object(&vertices, &indices, &[3, 3, 2]);
		let model_matrix = glm::scaling(&glm::vec3(TERRAIN_SCALE, TERRAIN_AMPLITUDE, TERRAIN_SCALE));
		meshes.insert(Mesh::new(vao, model_matrix, load_texture("textures/grass.jpg"), vec![0, indices.len() as GLsizei], None))
	};

	//This counter is used for both trees and grass
	let mut halton_counter = 1;

	//Plant trees
	const TREE_COUNT: usize = 500;
	let (trees_vao, trees_geo_boundaries, trees_mats) = unsafe {
		let model_data = load_wavefront_obj("models/tree1.obj").unwrap();

		let attribute_offsets = [3, 3, 2];
		let vao = create_vertex_array_object(&model_data.vertices, &model_data.indices, &attribute_offsets);
		let mut model_matrices = [0.0f32; TREE_COUNT * 16];

		//Populate the buffer
		for i in 0..TREE_COUNT {
			let xpos = TERRAIN_SCALE * (halton_sequence(halton_counter as f32, 2.0) - 0.5);
			let zpos = TERRAIN_SCALE * (halton_sequence(halton_counter as f32, 3.0) - 0.5);
			halton_counter += 1;
			
			//Get height from simplex noise generator
			let ypos = get_terrain_height(xpos, zpos, simplex_generator, TERRAIN_AMPLITUDE, TERRAIN_SCALE, SIMPLEX_SCALE);

			//Determine which floor triangle this tree is on
			let (moved_xpos, moved_zpos) = (xpos + (TERRAIN_SCALE / 2.0), zpos + (TERRAIN_SCALE / 2.0));			
			let (subsquare_x, subsquare_z) = (f32::floor(moved_xpos * ((TERRAIN_WIDTH - 1) as f32 / TERRAIN_SCALE)) as usize,
											  f32::floor(moved_zpos * ((TERRAIN_WIDTH - 1) as f32 / TERRAIN_SCALE)) as usize);
			let subsquare_index = subsquare_x + subsquare_z * (TERRAIN_WIDTH - 1);
			let (norm_x, norm_z) = (moved_xpos / (TERRAIN_WIDTH - 1) as f32 + subsquare_x as f32 * TERRAIN_SCALE / (TERRAIN_WIDTH - 1) as f32,
						  			moved_zpos / (TERRAIN_WIDTH - 1) as f32 + subsquare_z as f32 * TERRAIN_SCALE / (TERRAIN_WIDTH - 1) as f32);
			let normal_index = if norm_x + norm_z <= 1.0 {
				subsquare_index * 2
			} else {
				subsquare_index * 2 + 1
			};
			
			let rotation_vector = glm::cross::<f32, glm::U3>(&glm::vec3(0.0, 1.0, 0.0), &surface_normals[normal_index]);
			let rotation_magnitude = f32::acos(glm::dot(&glm::vec3(0.0, 1.0, 0.0), &surface_normals[normal_index]));
			let matrix = glm::translation(&glm::vec3(xpos, ypos, zpos)) * glm::rotation(rotation_magnitude*0.2, &rotation_vector);

			//Write this matrix to the buffer
			let mut count = 0;
			for j in glm::value_ptr(&matrix) {
				model_matrices[i * 16 + count] = *j;
				count += 1;
			}
		}

		let position_buffer = gl_gen_buffer();
		gl::BindBuffer(gl::ARRAY_BUFFER, position_buffer);
		gl::BufferData(gl::ARRAY_BUFFER, (TREE_COUNT * 16 * mem::size_of::<GLfloat>()) as GLsizeiptr, &model_matrices[0] as *const GLfloat as *const c_void, gl::STATIC_DRAW);
		gl::BindVertexArray(vao);

		for i in 0..4 {
			let current_attribute = (attribute_offsets.len() + i) as GLuint;
			gl::VertexAttribPointer(current_attribute, 4, gl::FLOAT, gl::FALSE, 16 * mem::size_of::<GLfloat>() as GLsizei, (4 * i * mem::size_of::<GLfloat>()) as *const c_void);
			gl::VertexAttribDivisor(current_attribute, 1);
			gl::EnableVertexAttribArray(current_attribute);
		}

		(vao, model_data.geo_boundaries, model_data.materials)
	};

	//Plant grass
	const GRASS_COUNT: usize = 100000;
	let grass_texture = unsafe { load_texture("textures/billboardgrass.png") };

	//Calculate the model_matrices for the grass billboards
	let (grass_vao, grass_indices_count) = unsafe {
		let vertices = [
			//Position				Normals						Tex coords
			-0.5, 0.0, 0.0,			0.0, 0.0, 1.0,				0.0, 1.0,
			0.5, 0.0, 0.0,			0.0, 0.0, 1.0,				1.0, 1.0,
			-0.5, 1.0, 0.0,			0.0, 0.0, 1.0,				0.0, 0.0,
			0.5, 1.0, 0.0,			0.0, 0.0, 1.0,				1.0, 0.0,
			0.0, 0.0, -0.5,			0.0, 0.0, 1.0,				0.0, 1.0,
			0.0, 0.0, 0.5,			0.0, 0.0, 1.0,				1.0, 1.0,
			0.0, 1.0, -0.5,			0.0, 0.0, 1.0,				0.0, 0.0,
			0.0, 1.0, 0.5,			0.0, 0.0, 1.0,				1.0, 0.0
		];
		let indices = [
			0u16, 1, 2,
			3, 2, 1,
			4, 5, 6,
			7, 6, 5
		];

		let mut model_matrices = vec![0.0f32; GRASS_COUNT * 16];

		//Populate the buffer
		for i in 0..GRASS_COUNT {
			let xpos = TERRAIN_SCALE * (halton_sequence(halton_counter as f32, 2.0) - 0.5);
			let zpos = TERRAIN_SCALE * (halton_sequence(halton_counter as f32, 3.0) - 0.5);
			halton_counter += 1;
			
			//Get height from simplex noise generator
			let ypos = get_terrain_height(xpos, zpos, simplex_generator, TERRAIN_AMPLITUDE, TERRAIN_SCALE, SIMPLEX_SCALE);

			//Determine which floor triangle this tree is on
			let (moved_xpos, moved_zpos) = (xpos + (TERRAIN_SCALE / 2.0), zpos + (TERRAIN_SCALE / 2.0));			
			let (subsquare_x, subsquare_z) = (f32::floor(moved_xpos * ((TERRAIN_WIDTH - 1) as f32 / TERRAIN_SCALE)) as usize,
											  f32::floor(moved_zpos * ((TERRAIN_WIDTH - 1) as f32 / TERRAIN_SCALE)) as usize);
			let subsquare_index = subsquare_x + subsquare_z * (TERRAIN_WIDTH - 1);
			let (norm_x, norm_z) = (moved_xpos / (TERRAIN_WIDTH - 1) as f32 + subsquare_x as f32 * TERRAIN_SCALE / (TERRAIN_WIDTH - 1) as f32,
						  			moved_zpos / (TERRAIN_WIDTH - 1) as f32 + subsquare_z as f32 * TERRAIN_SCALE / (TERRAIN_WIDTH - 1) as f32);
			let normal_index = if norm_x + norm_z <= 1.0 {
				subsquare_index * 2
			} else {
				subsquare_index * 2 + 1
			};
			
			let rotation_vector = glm::cross::<f32, glm::U3>(&glm::vec3(0.0, 1.0, 0.0), &surface_normals[normal_index]);
			let rotation_magnitude = f32::acos(glm::dot(&glm::vec3(0.0, 1.0, 0.0), &surface_normals[normal_index]));
			let matrix = glm::translation(&glm::vec3(xpos, ypos, zpos)) * glm::rotation(rotation_magnitude*0.2, &rotation_vector) * uniform_scale(0.5);

			//Write this matrix to the buffer
			let mut count = 0;
			for j in glm::value_ptr(&matrix) {
				model_matrices[i * 16 + count] = *j;
				count += 1;
			}
		}

		let matrices_buffer = gl_gen_buffer();
		let attribute_offsets = [3, 3, 2];
		let vao = create_vertex_array_object(&vertices, &indices, &attribute_offsets);
		gl::BindBuffer(gl::ARRAY_BUFFER, matrices_buffer);
		gl::BufferData(gl::ARRAY_BUFFER, (GRASS_COUNT * 16 * mem::size_of::<GLfloat>()) as GLsizeiptr, &model_matrices[0] as *const GLfloat as *const c_void, gl::STATIC_DRAW);
		gl::BindVertexArray(vao);

		for i in 0..4 {
			let current_attribute = (attribute_offsets.len() + i) as GLuint;
			gl::VertexAttribPointer(current_attribute, 4, gl::FLOAT, gl::FALSE, 16 * mem::size_of::<GLfloat>() as GLsizei, (4 * i * mem::size_of::<GLfloat>()) as *const c_void);
			gl::VertexAttribDivisor(current_attribute, 1);
			gl::EnableVertexAttribArray(current_attribute);
		}

		(vao, indices.len())
	};

	//Variables to keep track of the loaded models
	let model_texture = unsafe { load_texture("textures/checkerboard.jpg") };
	let model_bounding_sphere_radius = 0.20;
	let mut bound_controller_indices = Vec::new();
	let mut model_indices = Vec::new();
	let mut model_to_controller_matrices = Vec::new();

	//Initialize the struct of arrays containing controller related state
	let mut controllers = Controllers::new();

	//Index of the HMD mesh in meshes
	let mut hmd_mesh_index = None;

	//Camera state
	let mut camera = Camera::new(glm::vec3(0.0, -1.0, -1.0));

	let mut last_rbutton_state = window.get_mouse_button(MouseButton::Button2);

	//The instant recorded at the beginning of last frame
	let mut last_frame_instant = Instant::now();

	//Tracking space position information
	let mut tracking_to_world: glm::TMat4<f32> = glm::identity();
	let mut tracking_position: glm::TVec4<f32> = glm::zero();
	let mut is_flying = false;

	//Play background music
	let mut is_muted = false;
	let mut bgm_sink = match rodio::default_output_device() {
		Some(device) => {
			let sink = rodio::Sink::new(&device);
			let source = rodio::Decoder::new(BufReader::new(File::open("audio/dark_ruins.mp3").unwrap())).unwrap();
			sink.append(source);
			sink.set_volume(0.25);
			sink.play();
			Some(sink)
		}
		None => {
			println!("Unable to find audio device.");
			None
		}
	};

	//Flags
	let mut is_wireframe = false;
	let mut is_lighting = true;

	//Unit vector pointing towards the sun
	let light_direction = glm::normalize(&glm::vec4(0.8, 1.0, 1.0, 0.0));

	//Shadow map data
	let shadow_map_resolution = 10240;
	let projection_size = 10.0;
	let shadow_viewprojection = glm::ortho(-projection_size, projection_size, -projection_size, projection_size, -projection_size, 5.0 * projection_size) *
								glm::look_at(&glm::vec4_to_vec3(&(light_direction * 4.0)), &glm::vec3(0.0, 0.0, 0.0), &glm::vec3(0.0, 1.0, 0.0));
	let (shadow_buffer, shadow_map) = unsafe {
		let mut framebuffer = 0;
		gl::GenFramebuffers(1, &mut framebuffer);

		let mut depth_texture = 0;
		gl::GenTextures(1, &mut depth_texture);
		gl::BindTexture(gl::TEXTURE_2D, depth_texture);
		gl::TexImage2D(gl::TEXTURE_2D, 0, gl::DEPTH_COMPONENT as i32, shadow_map_resolution, shadow_map_resolution, 0, gl::DEPTH_COMPONENT, gl::FLOAT, ptr::null());
		gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
		gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
		gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::REPEAT as i32);
		gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::REPEAT as i32);

		gl::BindFramebuffer(gl::FRAMEBUFFER, framebuffer);
		gl::FramebufferTexture2D(gl::FRAMEBUFFER, gl::DEPTH_ATTACHMENT, gl::TEXTURE_2D, depth_texture, 0);

		if gl::CheckFramebufferStatus(gl::FRAMEBUFFER) != gl::FRAMEBUFFER_COMPLETE {
			panic!("Shadow map framebuffer didn't complete");
		}
		gl::BindFramebuffer(gl::FRAMEBUFFER, 0);

		(framebuffer, depth_texture)
	};

	//Main loop
	while !window.should_close() {
		//Calculate time since the last frame started
		let seconds_elapsed = {
			let frame_instant = Instant::now();
			let dur = frame_instant.duration_since(last_frame_instant);
			last_frame_instant = frame_instant;

			//There's an underlying assumption here that frames will always take less than one second to complete
			(dur.subsec_millis() as f32 / 1000.0) + (dur.subsec_micros() as f32 / 1_000_000.0)
		};

		//Find controllers if we haven't already
		if let Some(ref sys) = openvr_system {
			for i in 0..controllers.device_indices.len() {
				if let None = controllers.device_indices[i] {
					const ROLES: [TrackedControllerRole; 2] = [TrackedControllerRole::LeftHand, TrackedControllerRole::RightHand];
					controllers.device_indices[i] = sys.tracked_device_index_for_controller_role(ROLES[i]);
				}
			}
		}

		//Load controller meshes if we haven't already
		if None == controllers.mesh_indices[0] || None == controllers.mesh_indices[1] {
			if let Some(index) = controllers.device_indices[0] {
				if let Some(mesh) = load_openvr_mesh(&openvr_system, &openvr_rendermodels, index) {
					controllers.mesh_indices[0] = Some(meshes.insert(mesh.clone()));
					controllers.mesh_indices[1] = Some(meshes.insert(mesh));
				}
			}
		}

		//Load HMD mesh if we haven't already
		if let None = hmd_mesh_index {
			if let Some(mut mesh) = load_openvr_mesh(&openvr_system, &openvr_rendermodels, 0) {
				mesh.render_pass_visibilities = [false, false, !camera.attached_to_hmd];
				mesh.specular_coefficient = 128.0;
				hmd_mesh_index = Some(meshes.insert(mesh));
			}
		}

		//Get VR pose data
		let render_poses = match &openvr_compositor {
			Some(comp) => {	Some(comp.wait_get_poses().unwrap().render)	}
			None => { None }
		};

		//Get controller state structs
		for i in 0..Controllers::NUMBER_OF_CONTROLLERS {
			if let (Some(index), Some(sys)) = (controllers.device_indices[i], &openvr_system) {
				controllers.states[i] = sys.controller_state(index);
			}
		}

		//Check if the worker thread has new results for us
		if let Ok(work_result) = result_rx.try_recv() {
			match work_result {
				WorkResult::Model(option_mesh) => {
					if let Some(package) = option_mesh {
						let vao = unsafe { create_vertex_array_object(&package.vertices, &package.indices, &[3, 3, 2]) };
						let mesh = Mesh::new(vao, uniform_scale(0.3), model_texture, package.geo_boundaries, Some(package.materials));
						model_indices.push(Some(meshes.insert(mesh)));
						bound_controller_indices.push(None);
						model_to_controller_matrices.push(glm::identity());
					}
				}
			}
		}

		//Handle window and keyboard events
		for (_, event) in glfw::flush_messages(&events) {
			match event {
				WindowEvent::Close => {	window.set_should_close(true); }
				WindowEvent::FramebufferSize(width, height) => {
					window_size = (width as u32, height as u32);
					aspect_ratio = window_size.0 as f32 / window_size.1 as f32;
				}
				WindowEvent::Key(key, _, Action::Press, ..) => {
					match key {
						Key::Escape => { window.set_should_close(true); }
						Key::W => { camera.velocity = glm::vec3(0.0, 0.0, camera.speed); }
						Key::S => { camera.velocity = glm::vec3(0.0, 0.0, -camera.speed); }
						Key::A => { camera.velocity = glm::vec3(camera.speed, 0.0, 0.0); }
						Key::D => { camera.velocity = glm::vec3(-camera.speed, 0.0, 0.0); }
						Key::I => { camera.fov = 90.0; }
						Key::G => { is_lighting = !is_lighting; }
						Key::LeftShift => { camera.velocity *= 15.0; }
						Key::L => {		
							handle_result(order_tx.send(WorkOrder::Model));
						}
						Key::M => {
							if let Some(sink) = &mut bgm_sink {
								sink.set_volume(1.0 * is_muted as u32 as f32);
								is_muted = !is_muted;
							}
						}
						Key::Q => {
							if is_wireframe {
								unsafe { gl::PolygonMode(gl::FRONT_AND_BACK, gl::FILL); }
							} else {
								unsafe { gl::PolygonMode(gl::FRONT_AND_BACK, gl::LINE); }
							}
							is_wireframe = !is_wireframe;
						}
						Key::Space => {
							camera.attached_to_hmd = !camera.attached_to_hmd;
							if let Some(mesh) = meshes.get_element(hmd_mesh_index) {
								mesh.render_pass_visibilities[2] = !camera.attached_to_hmd;
							}
						}
						_ => { println!("You pressed the unbound key: {:?}", key); }
					}
				}
				WindowEvent::Key(key, _, Action::Release, ..) => {
					match key {
						Key::A | Key::D => { camera.velocity.x = 0.0; }
						Key::W | Key::S => { camera.velocity.z = 0.0; }
						Key::LeftShift => { camera.velocity /= 5.0; }
						_ => {}
					}
				}
				_ => {}
			}
		}

		//Handle mouse input
		let rbutton_state = window.get_mouse_button(MouseButton::Button2);
		let cursor_pos = window.get_cursor_pos();
		let cursor_delta = [cursor_pos.0 - window_size.0 as f64 / 2.0, cursor_pos.1 - window_size.1 as f64 / 2.0];

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

		//Check if the right mouse button has been clicked
		if last_rbutton_state == Action::Press && rbutton_state == Action::Release {
			if window.get_cursor_mode() == CursorMode::Normal {
				window.set_cursor_mode(CursorMode::Disabled);
			} else {
				window.set_cursor_mode(CursorMode::Normal);
			}

			//Reset cursor to center of screen
			window.set_cursor_pos(window_size.0 as f64 / 2.0, window_size.1 as f64 / 2.0);
		}
		last_rbutton_state = rbutton_state;

		//Handle controller input
		for i in 0..Controllers::NUMBER_OF_CONTROLLERS {
			if let (Some(device_index),
					Some(mesh_index),
					Some(state),
					Some(_sys),
					Some(poses)) = (  controllers.device_indices[i],
									  controllers.mesh_indices[i],
									  controllers.states[i],
									  &openvr_system,
									  render_poses) {

				controllers.collided_with[i].clear();
				for j in 0..model_indices.len() {
					//If the trigger was pulled this frame, grab the object the controller is currently touching, if there is one
					if let Some(loaded_index) = model_indices[j] {
						let is_colliding = glm::distance(&get_mesh_origin(&meshes[mesh_index as usize]), &get_mesh_origin(&meshes[loaded_index])) < model_bounding_sphere_radius;

						if is_colliding {
							controllers.colliding_with[i].push(loaded_index);
						}

						if controllers.pressed_this_frame(i, button_id::GRIP) && is_colliding {
							//Set the controller's mesh as the mesh the cube mesh is "bound" to
							bound_controller_indices[j] = Some(i);

							//Calculate the loaded_mesh-space to controller-space matrix aka inverse(controller.model_matrix) * loaded_mesh.model_matrix
							if let (Some(cont_mesh), Some(loaded_mesh)) = (&meshes[mesh_index], &meshes[loaded_index]) {
								model_to_controller_matrices[j] = glm::affine_inverse(cont_mesh.model_matrix) * loaded_mesh.model_matrix;
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

				let scale = 0.05;		
				let yvel = if i == 0 {
					scale * glm::vec4(0.0, -glm::clamp_scalar(state.axis[1].x * 4.0, 0.0, 1.0), 0.0, 0.0)
				} else {
					scale * glm::vec4(0.0, glm::clamp_scalar(state.axis[1].x * 4.0, 0.0, 1.0), 0.0, 0.0)
				};
				
				let mut movement_vector = yvel;

				//Handle left-hand controls
				if i == 0 {
					let controller_to_tracking = openvr_to_mat4(*poses[device_index as usize].device_to_absolute_tracking());

					//We check to make sure at least one axis isn't zero in order to ensure no division by zero
					if state.axis[0].x != 0.0 || state.axis[0].y != 0.0 {
						let mut temp = tracking_to_world * controller_to_tracking * glm::vec4(state.axis[0].x, 0.0, -state.axis[0].y, 0.0);
						let len = glm::length(&temp);
						temp.y = 0.0;
						temp *= len / glm::length(&temp);
						movement_vector += scale * temp;
					}

					if controllers.holding_button(i, button_id::STEAM_VR_TOUCHPAD) {
						movement_vector *= 5.0;
					}
				}

				//Check if the user toggled flying controls
				if i == 1 && controllers.released_this_frame(i, button_id::APPLICATION_MENU) {
					is_flying = !is_flying;
				}
				
				tracking_position += movement_vector;

				if !is_flying {
					tracking_position.y = get_terrain_height(tracking_position.x, tracking_position.z, simplex_generator, TERRAIN_AMPLITUDE, TERRAIN_SCALE, SIMPLEX_SCALE);
				}
			}
			tracking_to_world = glm::translation(&glm::vec4_to_vec3(&tracking_position));

			//Clear collided_with and move all of the elements from colliding_with into it
			controllers.collided_with[i].clear();
			for index in controllers.colliding_with[i].drain(0..controllers.colliding_with[i].len()) {
				controllers.collided_with[i].push(index);
			}
		}
		controllers.previous_states = controllers.states;

		//Get view matrices
		let v_matrices = match (&openvr_system, &render_poses) {
			(Some(sys), Some(poses)) => {
					let hmd_to_tracking = openvr_to_mat4(*poses[0].device_to_absolute_tracking());
					let left_eye_to_hmd = openvr_to_mat4(sys.eye_to_head_transform(Eye::Left));
					let right_eye_to_hmd = openvr_to_mat4(sys.eye_to_head_transform(Eye::Right));

					let companion_v_mat = if camera.attached_to_hmd { 
						glm::affine_inverse(tracking_to_world * hmd_to_tracking)
					} else {
						camera.freecam_matrix()
					};

					//Need to return inverse(tracking_to_world * hmd_to_tracking * eye_to_hmd)
					[glm::affine_inverse(tracking_to_world * hmd_to_tracking * left_eye_to_hmd),
					 glm::affine_inverse(tracking_to_world * hmd_to_tracking * right_eye_to_hmd),
					 companion_v_mat]
			}
			_ => { [glm::identity(), glm::identity(), camera.freecam_matrix()] }
		};

		//Get projection matrices
		let p_mat = glm::perspective(aspect_ratio, f32::to_radians(camera.fov), NEAR_Z, FAR_Z);
		let p_matrices = match &openvr_system {
			Some(sys) => { [get_projection_matrix(sys, Eye::Left), get_projection_matrix(sys, Eye::Right), p_mat] }
			None => { [glm::identity(), glm::identity(), p_mat] }
		};

		//Update the camera
		//The process here is, for the camera's velocity vector: View Space -> World Space -> scale by seconds_elapsed -> Make into vec4
		camera.position += glm::vec4_to_vec3(&(seconds_elapsed * (glm::affine_inverse(v_matrices[2]) * glm::vec3_to_vec4(&camera.velocity))));
		camera.fov += camera.fov_delta * seconds_elapsed;
		
		//Update the OpenVR meshes
		if let Some(poses) = render_poses {
			update_openvr_mesh(&mut meshes, &poses, &tracking_to_world, 0, hmd_mesh_index);
			for i in 0..Controllers::NUMBER_OF_CONTROLLERS {
				if let Some(index) = &controllers.device_indices[i] {
					update_openvr_mesh(&mut meshes, &poses, &tracking_to_world, *index as usize, controllers.mesh_indices[i]);
				}
			}
		}

		//If a model is being grabbed, update its model matrix
		for i in 0..bound_controller_indices.len() {
			if let Some(index) = bound_controller_indices[i] {
				if let (Some(mesh_index), Some(load_index)) = (controllers.mesh_indices[index], model_indices[i]) {				
					if let (Some(loaded), Some(controller)) = meshes.two_mut_refs(load_index, mesh_index) {
						loaded.model_matrix = controller.model_matrix * model_to_controller_matrices[i];
					}
				}
			}
		}

		//Rendering code
		let framebuffers = [vr_render_target, vr_render_target, 0];
		let eyes = [Some(Eye::Left), Some(Eye::Right), None];
		let sizes = [render_target_size, render_target_size, window_size];
		let render_context = RenderContext::new(&p_matrices, &v_matrices, &light_direction, shadow_map, &shadow_viewprojection, is_lighting);
		unsafe {
			gl::Enable(gl::DEPTH_TEST);	//Enable depth testing
			gl::DepthFunc(gl::LEQUAL);	//Pass the fragment with the smallest z-value

			//Set clear color
			gl::ClearColor(0.53, 0.81, 0.92, 1.0);

			//Render the shadow map
			gl::BindFramebuffer(gl::FRAMEBUFFER, shadow_buffer);
			gl::Viewport(0, 0, shadow_map_resolution, shadow_map_resolution);
			gl::DrawBuffer(gl::NONE);
			gl::ReadBuffer(gl::NONE);
			gl::Clear(gl::DEPTH_BUFFER_BIT);

			//Render meshes into shadow map
			gl::UseProgram(shadow_map_shader);
			for option_mesh in meshes.iter() {
				if let Some(mesh) = option_mesh {
					let mvp = shadow_viewprojection * mesh.model_matrix;
					gl::UniformMatrix4fv(get_uniform_location(shadow_map_shader, "shadowMVP"), 1, gl::FALSE, &flatten_glm(&mvp) as *const GLfloat);
					gl::BindVertexArray(mesh.vao);
					for i in 0..mesh.geo_boundaries.len()-1 {
						gl::DrawElements(gl::TRIANGLES, mesh.geo_boundaries[i + 1] - mesh.geo_boundaries[i], gl::UNSIGNED_SHORT, (mem::size_of::<GLshort>() as i32 * mesh.geo_boundaries[i]) as *const c_void);
					}
				}
			}

			//Rendering instanced meshes
			gl::UseProgram(instanced_shadow_map_shader);
			gl::UniformMatrix4fv(get_uniform_location(instanced_shadow_map_shader, "shadowVP"), 1, gl::FALSE, &flatten_glm(&shadow_viewprojection) as *const GLfloat);

			//Render instanced trees into shadow map
			gl::BindVertexArray(trees_vao);
			for i in 0..trees_geo_boundaries.len()-1 {
				gl::DrawElementsInstanced(gl::TRIANGLES, trees_geo_boundaries[i + 1] - trees_geo_boundaries[i], gl::UNSIGNED_SHORT, (mem::size_of::<GLshort>() as i32 * trees_geo_boundaries[i]) as *const c_void, TREE_COUNT as GLsizei);
			}

			//Render instanced grass into shadow map
			//gl::BindVertexArray(grass_vao);
			//gl::DrawElementsInstanced(gl::TRIANGLES, grass_indices_count as GLint, gl::UNSIGNED_SHORT, ptr::null(), GRASS_COUNT as GLsizei);

			//Turn the color buffer back on now that we're done rendering the shadow map
			gl::DrawBuffer(gl::BACK);

			//Render once per framebuffer (Left eye, Right eye, Companion window)
			for i in 0..framebuffers.len() {
				//Set up render target
				gl::BindFramebuffer(gl::FRAMEBUFFER, framebuffers[i]);
				gl::Viewport(0, 0, sizes[i].0 as GLsizei, sizes[i].1 as GLsizei);

				//Clear the framebuffer's color buffer and depth buffer
				gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

				//Render the regular meshes
				gl::Enable(gl::CULL_FACE);
				render_meshes(&meshes, model_shader, i, &render_context);

				//Render the trees with instanced rendering
				gl::UseProgram(instanced_model_shader);
				
				bind_uniforms(instanced_model_shader,
							  &["view_projection", "shadow_vp"],
							  &[&(p_matrices[i] * v_matrices[i]), &shadow_viewprojection],
							  &["view_position", "light_direction"],
							  &[&render_context.view_positions[i], &light_direction],
							  &["using_material", "lighting", "shadow_map", "tex", "shadow_map"],
							  &[1, is_lighting as GLint, 0, 0, 1]);

				gl::ActiveTexture(gl::TEXTURE1);
				gl::BindTexture(gl::TEXTURE_2D, shadow_map);
				gl::BindVertexArray(trees_vao);

				//Draw calls
				for j in 0..trees_geo_boundaries.len()-1 {
					bind_material(instanced_model_shader, &trees_mats[j]);
					gl::DrawElementsInstanced(gl::TRIANGLES, trees_geo_boundaries[j + 1] - trees_geo_boundaries[j], gl::UNSIGNED_SHORT, (mem::size_of::<GLshort>() as i32 * trees_geo_boundaries[j]) as *const c_void, TREE_COUNT as GLsizei);
				}

				//Render the grass billboards with instanced rendering
				gl::ActiveTexture(gl::TEXTURE0);
				gl::BindTexture(gl::TEXTURE_2D, grass_texture);
				gl::Uniform1i(get_uniform_location(instanced_model_shader, "using_material"), 0);
				gl::BindVertexArray(grass_vao);

				//Disable backface culling before the draw call because we want the grass to be double-sided
				gl::Disable(gl::CULL_FACE);

				gl::DrawElementsInstanced(gl::TRIANGLES, grass_indices_count as GLint, gl::UNSIGNED_SHORT, ptr::null(), GRASS_COUNT as GLsizei);

				//Draw the skybox last to take advantage of early depth testing
				//Don't draw the skybox in wireframe mode
				if !is_wireframe {
					//Compute the view-projection matrix for the skybox (the conversion functions are just there to nullify the translation component of the view matrix)
					//The skybox vertices should obviously be rotated along with the camera, but shouldn't be translated in order to maintain the illusion
					//that the sky is infinitely far away
					let skybox_view_projection = p_matrices[i] * glm::mat3_to_mat4(&glm::mat4_to_mat3(&v_matrices[i]));

					//Render the skybox
					gl::UseProgram(skybox_shader);
					gl::UniformMatrix4fv(get_uniform_location(skybox_shader, "view_projection"), 1, gl::FALSE, &flatten_glm(&skybox_view_projection) as *const GLfloat);
					gl::BindTexture(gl::TEXTURE_CUBE_MAP, skybox_cubemap);
					gl::BindVertexArray(skybox_vao);
					gl::DrawElements(gl::TRIANGLES, 36, gl::UNSIGNED_SHORT, ptr::null());
				}

				//Submit render to HMD
				submit_to_hmd(eyes[i], &openvr_compositor, &openvr_texture_handle);
			}
		}

		window.render_context().swap_buffers();
		glfw.poll_events();
	}

	//Shut down the worker thread
	handle_result(order_tx.send(WorkOrder::Quit));

	worker_handle.join().unwrap();

	//Shut down OpenVR
	if let Some(ctxt) = openvr_context {
		unsafe { ctxt.shutdown(); }
	}
}
