extern crate gl;
extern crate nalgebra_glm as glm;
use glfw::{Action, Context, CursorMode, Key, MouseButton, WindowMode, WindowEvent};
use openvr::{ApplicationType, button_id, ControllerState, Eye, System, RenderModels, TrackedControllerRole, TrackedDevicePose};
use openvr::compositor::texture::{ColorSpace, Handle, Texture};
use nfd::Response;
use std::fs::File;
use std::io::BufReader;
use std::thread;
use std::time::Instant;
use std::sync::mpsc;
use std::ptr;
use obj;
use rand::random;
use crate::structs::*;
use crate::glutil::*;
use self::gl::types::*;

mod structs;
mod glutil;

const NEAR_Z: f32 = 0.25;
const FAR_Z: f32 = 50.0;

type MeshArrays = (Vec<f32>, Vec<u16>);

fn openvr_to_mat4(mat: [[f32; 4]; 3]) -> glm::TMat4<f32> {
	glm::mat4(
			mat[0][0], mat[0][1], mat[0][2], mat[0][3],
			mat[1][0], mat[1][1], mat[1][2], mat[1][3],
			mat[2][0], mat[2][1], mat[2][2], mat[2][3],
			0.0, 0.0, 0.0, 1.0
		)
}

fn flatten_glm(mat: &glm::TMat4<f32>) -> [f32; 16] {
	let slice = glm::value_ptr(mat);

	let mut result = [0.0; 16];
	for i in 0..16 {
		result[i] = slice[i];
	}
	result
}

fn get_projection_matrix(sys: &System, eye: Eye) -> glm::TMat4<f32> {
	let t_matrix = sys.projection_matrix(eye, NEAR_Z, FAR_Z);

	glm::mat4(
			t_matrix[0][0], t_matrix[0][1], t_matrix[0][2], t_matrix[0][3],
			t_matrix[1][0], t_matrix[1][1], t_matrix[1][2], t_matrix[1][3],
			t_matrix[2][0], t_matrix[2][1], t_matrix[2][2], t_matrix[2][3],
			t_matrix[3][0], t_matrix[3][1], t_matrix[3][2], t_matrix[3][3]
		)
}

fn attach_mesh_to_controller(meshes: &mut OptionVec<Mesh>, poses: &[TrackedDevicePose], controller_index: &Option<u32>, mesh_index: Option<usize>) {
	if let Some(index) = controller_index {
		let controller_model_matrix = openvr_to_mat4(*poses[*index as usize].device_to_absolute_tracking());
		if let Some(mesh) = get_mesh(meshes, mesh_index) {
			mesh.model_matrix = controller_model_matrix;
		}
	}
}

fn load_controller_meshes<'a>(openvr_system: &Option<System>, openvr_rendermodels: &Option<RenderModels>, meshes: &mut OptionVec<Mesh>, index: u32) -> [Option<usize>; 2] {
	let mut result = [None; 2];
	if let (Some(ref sys), Some(ref ren_mod)) = (&openvr_system, &openvr_rendermodels) {
		let name = sys.string_tracked_device_property(index, openvr::property::RenderModelName_String).unwrap();
		if let Some(model) = ren_mod.load_render_model(&name).unwrap() {
			//Flatten each vertex into a simple &[f32]
			const ELEMENT_STRIDE: usize = 8;
			let mut vertices = Vec::with_capacity(ELEMENT_STRIDE * model.vertices().len());
			for vertex in model.vertices() {
				vertices.push(vertex.position[0]);
				vertices.push(vertex.position[1]);
				vertices.push(vertex.position[2]);
				vertices.push(vertex.normal[0]);
				vertices.push(vertex.normal[1]);
				vertices.push(vertex.normal[2]);
				vertices.push(vertex.texture_coord[0]);
				vertices.push(vertex.texture_coord[1]);
			}

			//Create vao
			let vao = unsafe { create_vertex_array_object(&vertices, model.indices()) };

			let mesh = Mesh::new(vao, glm::translation(&glm::vec3(0.0, -1.0, 0.0)), 0, model.indices().len() as i32);
			let left_index = Some(meshes.insert(mesh));

			let mesh = Mesh::new(vao, glm::translation(&glm::vec3(0.0, -1.0, 0.0)), 0, model.indices().len() as i32);
			let right_index = Some(meshes.insert(mesh));

			result = [left_index, right_index];
		}
	}
	result
}

fn pressed_this_frame(state: &ControllerState, p_state: &ControllerState, flag: u32) -> bool {
	state.button_pressed & (1 as u64) << flag != 0 && p_state.button_pressed & (1 as u64) << flag == 0
}

fn get_frame_origin(model_matrix: &glm::TMat4<f32>) -> glm::TVec4<f32> {
	model_matrix * glm::vec4(0.0, 0.0, 0.0, 1.0)
}

fn get_mesh_origin(mesh: &Option<Mesh>) -> glm::TVec4<f32> {
	match mesh {
		Some(mesh) => {
			get_frame_origin(&mesh.model_matrix)
		}
		None => {
			println!("Couldn't return mesh origin cause it was \"None\"");
			glm::vec4(0.0, 0.0, 0.0, 1.0)
		}
	}
}

fn load_wavefront_obj(path: &str) -> Option<MeshArrays> {
	let model: obj::Obj = match obj::load_obj(BufReader::new(File::open(path).unwrap())) {
		Ok(m) => {
			m
		}
		Err(e) => {
			println!("{:?}", e);
			return None;
		}
	};

	//Take loaded model and create a Mesh
	const ELEMENT_STRIDE: usize = 8;
	let mut vert_data = Vec::with_capacity(ELEMENT_STRIDE * model.vertices.len());
	for v in model.vertices {
		vert_data.push(v.position[0]);
		vert_data.push(v.position[1]);
		vert_data.push(v.position[2]);
		vert_data.push(v.normal[0]);
		vert_data.push(v.normal[1]);
		vert_data.push(v.normal[2]);
		vert_data.push(random::<f32>());
		vert_data.push(random::<f32>());
	}
	Some((vert_data, model.indices))
}

fn get_mesh(meshes: &mut OptionVec<Mesh>, index: Option<usize>) -> Option<&mut Mesh> {
	match index {
		Some(i) => {
			match &mut meshes[i] {
				Some(mesh) => { Some(mesh) }
				None => { None }
			}
		}
		None => { None }
	}
}

fn main() {
	//Initialize OpenVR
	let openvr_context = unsafe {
		match openvr::init(ApplicationType::Scene) {
			Ok(ctxt) => {
				Some(ctxt)
			}
			Err(e) => {
				println!("ERROR: {}", e);
				None
			}
		}
	};

	//Get the OpenVR submodules
	let (openvr_system, openvr_compositor, openvr_rendermodels) = {
		if let Some(ref ctxt) = openvr_context {
			(Some(ctxt.system().unwrap()), Some(ctxt.compositor().unwrap()), Some(ctxt.render_models().unwrap()))
		} else {
			(None, None, None)
		}
	};

	//Calculate render target size
	let render_target_size = match openvr_system {
		Some(ref sys) => {
			sys.recommended_render_target_size()
		}
		None => {
			(1280, 720)
		}
	};

	//Init glfw
	let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

	//Using OpenGL 3.3 core, but that could change
	glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
	glfw.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));

	//Disable window resizing
	glfw.window_hint(glfw::WindowHint::Resizable(false));

	//Create window
	let window_size = (1280, 720);
	let (mut window, events) = {
		glfw.create_window(window_size.0,
						   window_size.1,
						   "Model viewer",
							WindowMode::Windowed).unwrap()
	};

	//Calculate window's aspect ratio
	let aspect_ratio = window_size.0 as f32 / window_size.1 as f32;

	//Configure window
	window.set_key_polling(true);
	window.set_framebuffer_size_polling(true);

	//Load all OpenGL function pointers
	gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);

	//Compile shader program
	let nonluminous_shader = unsafe { compile_program_from_files("shaders/nonluminous_vertex.glsl", "shaders/nonluminous_fragment.glsl") };

	//Get locations of program uniforms
	let mvp_location = unsafe { get_uniform_location(nonluminous_shader, "mvp") };
	let model_matrix_location = unsafe { get_uniform_location(nonluminous_shader, "model_matrix") };
	let light_position_location = unsafe { get_uniform_location(nonluminous_shader, "light_position") };
	let view_position_location = unsafe { get_uniform_location(nonluminous_shader, "view_position") };

	//Setup the VR rendering target
	let vr_render_target = unsafe { create_vr_render_target(&render_target_size) };
	let openvr_texture_handle = Texture {
		handle: Handle::OpenGLTexture(vr_render_target as usize),
		color_space: ColorSpace::Auto
	};

	//Enable and configure depth testing and enable backface culling
	unsafe {
		gl::Enable(gl::DEPTH_TEST);
		gl::DepthFunc(gl::LESS);
		gl::Enable(gl::CULL_FACE);
	}

	//Texture loading channel
	let (texture_tx, texture_rx) = mpsc::channel::<ImageData>();

	//The channel for sending 3D models between threads
	let (load_tx, load_rx) = mpsc::channel::<Option<MeshArrays>>();

	//Spawn thread to load brick texture
	let tx = texture_tx.clone();
	thread::spawn( move || {
		tx.send(image_data_from_path("textures/bricks.jpg")).unwrap();
	});

	//Textures
	let checkerboard_texture = unsafe { load_texture("textures/checkerboard.jpg") };
	let mut brick_texture = 0;

	//OptionVec of meshes
	let mut meshes = OptionVec::with_capacity(5);

	//Create the floor
	unsafe {
		let vertices = [
			//Positions					//Normals							//Tex coords
			-0.5f32, 0.0, -0.5,			0.0, 1.0, 0.0,						0.0, 0.0,
			-0.5, 0.0, 0.5,				0.0, 1.0, 0.0,						0.0, 4.0,
			0.5, 0.0, -0.5,				0.0, 1.0, 0.0,						4.0, 0.0,
			0.5, 0.0, 0.5,				0.0, 1.0, 0.0,						4.0, 4.0
		];
		let indices = [
			0u16, 1, 2,
			1, 3, 2
		];
		let vao = create_vertex_array_object(&vertices, &indices);
		let mesh = Mesh::new(vao, glm::scaling(&glm::vec3(5.0, 5.0, 5.0)), checkerboard_texture, indices.len() as i32);
		meshes.insert(mesh);
	}

	//Create the sphere that represents the light source
	let mut light_position = glm::vec4(0.0, 1.0, 0.0, 1.0);
	let sphere_mesh_index = unsafe {
		match load_wavefront_obj("models/sphere.obj") {
			Some(obj) => {
				let vao = create_vertex_array_object(&obj.0, &obj.1);
				let t = glm::vec4_to_vec3(&light_position);
				let mesh = Mesh::new(vao, glm::translation(&t) * glm::scaling(&glm::vec3(0.1, 0.1, 0.1)), 0, obj.1.len() as i32);
				Some(meshes.insert(mesh))
			}
			None => {
				None
			}
		}
	};

	//Variables for the mesh loaded from a file	
	let loaded_sphere_radius = 0.20;
	let mut loaded_bound_controller_index = None;
	let mut loaded_mesh_index = None;
	let mut loaded_space_to_controller_space = glm::identity();

	//Thread listening flags
	let mut loading_model_flag = false;
	let mut loading_brick_texture_flag = true;

	//Initialize the struct of arrays containing controller related state
	let mut controllers = Controllers::new();

	//Gameplay state
	let mut ticks = 0.0;

	//Camera state
	let mut manual_camera = false;
	let mut camera = Camera::new(glm::vec3(0.0, -1.0, -1.0));

	/*
	let mut camera.position = glm::vec3(0.0, -1.0, -1.0);
	let mut camera.velocity = glm::vec3(0.0, 0.0, 0.0);
	let mut camera.yaw = 0.0;
	let mut camera.pitch = 0.0;
	let mut camera.fov = 90.0;
	let mut camera.fov_delta = 0.0;
	let Camera::SPEED = 2.0;
	let Camera::FOV_SPEED = 5.0;
	*/

	let mut locked_cursor = false;
	let mut last_mouse_action = window.get_mouse_button(MouseButton::Button1);

	//The instant recorded at the beginning of last frame
	let mut last_frame_instant = Instant::now();

	//Set up rendering data for later
	let framebuffers = [vr_render_target, vr_render_target, 0];
	let sizes = [render_target_size, render_target_size, window_size];
	let eyes = [Some(Eye::Left), Some(Eye::Right), None];

	//Main loop
	while !window.should_close() {
		//Frame rate independence variables
		let frame_instant = Instant::now();
		let seconds_elapsed = { 
			let dur = frame_instant.duration_since(last_frame_instant);
			(dur.subsec_millis() as f32 / 1000.0) + (dur.subsec_micros() as f32 / 1_000_000.0)
		};

		//Find controllers if we haven't already
		if let Some(ref sys) = openvr_system {
			for i in 0..controllers.indices.len() {
				if let None = controllers.indices[i] {
					const ROLES: [TrackedControllerRole; 2] = [TrackedControllerRole::LeftHand, TrackedControllerRole::RightHand];
					controllers.indices[i] = sys.tracked_device_index_for_controller_role(ROLES[i]);
				}
			}
		}

		//Load controller meshes if we haven't already
		if let None = controllers.mesh_indices[0] {
			for i in 0..controllers.indices.len() {
				if let Some(index) = controllers.indices[i] {
					controllers.mesh_indices = load_controller_meshes(&openvr_system,
														  &openvr_rendermodels,
														  &mut meshes,
														  index);

					//We break here because the models only need to be loaded once, but we still want to check both controller indices if necessary
					break;
				}
			}
		}

		//Get VR pose data
		let render_poses = match openvr_compositor {
			Some(ref comp) => {
				Some(comp.wait_get_poses().unwrap().render)
			}
			None => {
				None
			}
		};

		//Get controller state structs
		for i in 0..Controllers::NUMBER_OF_CONTROLLERS {
			if let (Some(index), Some(sys)) = (controllers.indices[i], &openvr_system) {
				controllers.states[i] = sys.controller_state(index);
			}
		}

		//Check if a new model has been loaded
		if loading_model_flag {
			if let Ok(package) = load_rx.try_recv() {
				if let Some(pack) = package {
					let vao = unsafe { create_vertex_array_object(&pack.0, &pack.1) };
					let mesh = Mesh::new(vao, glm::translation(&glm::vec3(0.0, 0.8, 0.0)) * glm::scaling(&glm::vec3(0.1, 0.1, 0.1)), brick_texture, pack.1.len() as i32);

					//Delete old mesh if there is one
					if let Some(i) = loaded_mesh_index {
						meshes[i] = None;
					}
					loaded_mesh_index = Some(meshes.insert(mesh));
				}				
				loading_model_flag = false;
			}
		}

		//Check if the cube's texture has been loaded
		if loading_brick_texture_flag {
			//Check if the cube's texture is loaded yet
			if let Ok((data, width, height)) = texture_rx.try_recv() {
				let image_data = (data, width, height);
				brick_texture = unsafe { load_texture_from_data(image_data) };

				let mesh_indices = [loaded_mesh_index, sphere_mesh_index];

				for index in &mesh_indices {
					if let Some(mesh) = get_mesh(&mut meshes, *index) {
						mesh.texture = brick_texture;
					}
				}

				loading_brick_texture_flag = false;
			}
		}

		//Handle window and keyboard events
		for (_, event) in glfw::flush_messages(&events) {
			match event {
				WindowEvent::Close => {
					window.set_should_close(true);
				}
				WindowEvent::Key(key, _, Action::Press, ..) => {
					match key {
						Key::W => {
							camera.velocity.z = Camera::SPEED;
						}
						Key::S => {
							camera.velocity.z = -Camera::SPEED;
						}
						Key::A => {
							camera.velocity.x = Camera::SPEED;
						}
						Key::D => {
							camera.velocity.x = -Camera::SPEED;
						}
						Key::O => {
							camera.fov_delta = -Camera::FOV_SPEED;
						}
						Key::P => {
							camera.fov_delta = Camera::FOV_SPEED;
						}
						Key::L => {
							let tx = load_tx.clone();
							thread::spawn( move || {
								//Invoke file selection dialogue
								let path = match nfd::open_file_dialog(None, None).unwrap() {
									Response::Okay(filename) => { filename }
									_ => { return }
								};

								//Send model data back to the main thread
								tx.send(load_wavefront_obj(&path)).unwrap();
							});
							loading_model_flag = true;
						}
						Key::Space => {
							manual_camera = !manual_camera;
						}
						Key::Escape => {
							window.set_should_close(true);
						}
						_ => {
							println!("You pressed the unbound key: {:?}", key);
						}
					}
				}
				WindowEvent::Key(key, _, Action::Release, ..) => {
					match key {
						Key::A | Key::D => {
							camera.velocity.x = 0.0;
						}
						Key::W | Key::S => {
							camera.velocity.z = 0.0;
						}
						Key::O | Key::P => {
							camera.fov_delta = 0.0;
							println!("fov is now {}", camera.fov);
						}
						_ => {}
					}
				}
				_ => {}
			}
		}

		//Handle mouse input
		let mouse_action = window.get_mouse_button(MouseButton::Button1);
		let cursor_pos = window.get_cursor_pos();
		let cursor_delta = (cursor_pos.0 - window_size.0 as f64 / 2.0, cursor_pos.1 - window_size.1 as f64 / 2.0);

		if locked_cursor {
			const MOUSE_SENSITIVITY: f32 = 0.001;
			camera.yaw += cursor_delta.0 as f32 * MOUSE_SENSITIVITY;
			camera.pitch += cursor_delta.1 as f32 * MOUSE_SENSITIVITY;

			if camera.pitch > glm::half_pi() {
				camera.pitch = glm::half_pi();
			} else if camera.pitch < -glm::half_pi::<f32>() {
				camera.pitch = -glm::half_pi::<f32>();
			}

			//Reset cursor to center of screen
			window.set_cursor_pos(window_size.0 as f64 / 2.0, window_size.1 as f64 / 2.0);
		}

		//Check if the mouse has been clicked
		if last_mouse_action == Action::Press && mouse_action == Action::Release {
			locked_cursor = !locked_cursor;

			if locked_cursor {
				window.set_cursor_mode(CursorMode::Disabled);
			} else {
				window.set_cursor_mode(CursorMode::Normal);
			}

			//Reset cursor to center of screen
			window.set_cursor_pos(window_size.0 as f64 / 2.0, window_size.1 as f64 / 2.0);
		}
		last_mouse_action = mouse_action;

		//Handle controller input
		for i in 0..Controllers::NUMBER_OF_CONTROLLERS {
			if let (Some(mesh_index),
					Some(loaded_index),
					Some(state),
					Some(p_state)) = (controllers.mesh_indices[i],
									  loaded_mesh_index,
									  controllers.states[i],
									  controllers.previous_states[i]) {

				//If the trigger was pulled this frame, grab the object the controller is currently touching, if there is one
				if pressed_this_frame(&state, &p_state, button_id::STEAM_VR_TRIGGER) {
					let controller_origin = get_mesh_origin(&meshes[mesh_index as usize]);
					let loaded_origin = get_mesh_origin(&meshes[loaded_index]);

					//Check for collision
					if glm::distance(&controller_origin, &loaded_origin) < loaded_sphere_radius {
						//Set the controller's mesh as the mesh the cube mesh is "bound" to
						loaded_bound_controller_index = Some(i);

						//Calculate the cube-space to controller-space matrix aka inverse(controller.model_matrix) * cube.model_matrix
						if let (Some(cont_mesh), Some(loaded_mesh)) = (&meshes[mesh_index], &meshes[loaded_index]) {
							loaded_space_to_controller_space = glm::affine_inverse(cont_mesh.model_matrix) * loaded_mesh.model_matrix;
						}
					}
				}

				//If the trigger was released this frame
				if state.button_pressed & (1 as u64) << button_id::STEAM_VR_TRIGGER == 0 &&
				   p_state.button_pressed & (1 as u64) << button_id::STEAM_VR_TRIGGER != 0 {
				   	if let Some(idx) = loaded_bound_controller_index {
				   		if idx == i {
				   			loaded_bound_controller_index = None;
				   		}
				   	}					
				}
			}
		}

		//Get view matrices
		let v_matrices = match openvr_system {
			Some(ref sys) => {
				match render_poses {
					Some(poses) => {
						let hmd_to_absolute = openvr_to_mat4(*poses[0].device_to_absolute_tracking());
						let left_eye_to_hmd = openvr_to_mat4(sys.eye_to_head_transform(Eye::Left));
						let right_eye_to_hmd = openvr_to_mat4(sys.eye_to_head_transform(Eye::Right));

						let companion_v_mat = if !manual_camera { 
							glm::affine_inverse(hmd_to_absolute)
						} else {
							//glm::translation(&camera.position) * glm::rotation(camera.yaw, &glm::vec3(0.0, 1.0, 0.0))
							glm::rotation(camera.pitch, &glm::vec3(1.0, 0.0, 0.0)) *
							glm::rotation(camera.yaw, &glm::vec3(0.0, 1.0, 0.0)) *
							glm::translation(&camera.position)
						};

						//Need to return inverse(hmd_to_absolute * eye_to_hmd)
						[glm::affine_inverse(hmd_to_absolute * left_eye_to_hmd),
						 glm::affine_inverse(hmd_to_absolute * right_eye_to_hmd),
						 companion_v_mat]
					}
					None => {						
						//Create a matrix that gets a decent view of the scene
						let view_matrix = glm::translation(&camera.position) * glm::rotation(camera.yaw, &glm::vec3(0.0, 1.0, 0.0));
						[glm::identity(), glm::identity(), view_matrix]
					}
				}
			}
			None => {
				//Create a matrix that gets a decent view of the scene
				let view_matrix = glm::translation(&camera.position) * glm::rotation(camera.yaw, &glm::vec3(0.0, 1.0, 0.0));
				[glm::identity(), glm::identity(), view_matrix]
			}
		};

		//Get view positions
		let view_positions = {
			let mut temp = Vec::with_capacity(v_matrices.len());
			for matrix in &v_matrices {
				temp.push(glm::affine_inverse(*matrix) * glm::vec4(0.0, 0.0, 0.0, 1.0));
			}
			temp
		};

		//Get projection matrices
		let p_mat = glm::perspective(aspect_ratio, f32::to_radians(camera.fov), NEAR_Z, FAR_Z);
		let p_matrices = match openvr_system {
			Some(ref sys) => {
				[get_projection_matrix(sys, Eye::Left), get_projection_matrix(sys, Eye::Right), p_mat]
			}
			None => {
				[glm::identity(), glm::identity(), p_mat]
			}
		};

		//Update simulation
		ticks += 2.0 * seconds_elapsed;

		//Update the camera
		camera.position += camera.velocity * seconds_elapsed;
		camera.fov += camera.fov_delta * seconds_elapsed;

		//Ensure controller meshes are drawn at each controller's position
		if let Some(poses) = render_poses {
			for i in 0..Controllers::NUMBER_OF_CONTROLLERS {
				attach_mesh_to_controller(&mut meshes, &poses, &controllers.indices[i], controllers.mesh_indices[i]);
			}
		}

		//If the loaded mesh is currently being grabbed, draw it at the grabbing controller's position
		if let Some(index) = loaded_bound_controller_index {
			if let (Some(mesh_index), Some(load_index)) = (controllers.mesh_indices[index], loaded_mesh_index) {
				let indices = meshes.two_mut_refs(load_index, mesh_index);
				if let (Some(loaded), Some(controller)) = indices {
					loaded.model_matrix = controller.model_matrix * loaded_space_to_controller_space;
				}
			}
		}

		//Make the light bob up and down
		if let Some(mesh) = get_mesh(&mut meshes, sphere_mesh_index) {
			mesh.model_matrix = glm::translation(&glm::vec3(0.0, 0.5*f32::sin(ticks*0.2) + 0.8, 0.0)) * glm::scaling(&glm::vec3(0.1, 0.1, 0.1));
			light_position = get_frame_origin(&mesh.model_matrix);
		}

		//End of frame updates
		controllers.previous_states = controllers.states;
		last_frame_instant = frame_instant;

		//Rendering code
		unsafe {
			//Set clear color
			gl::ClearColor(0.53, 0.81, 0.92, 1.0);
						
			//Bind the program that will render the meshes
			gl::UseProgram(nonluminous_shader);

			//Render once per framebuffer (Left eye, Right eye, Companion window)
			for i in 0..framebuffers.len() {
				//Set up render target
				gl::BindFramebuffer(gl::FRAMEBUFFER, framebuffers[i]);
				gl::Viewport(0, 0, sizes[i].0 as GLsizei, sizes[i].1 as GLsizei);

				//Render the scene
				gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
				for option_mesh in meshes.iter() {
					if let Some(mesh) = option_mesh {
						//Compute the model-view-projection matrix
						let mvp = p_matrices[i] * v_matrices[i] * mesh.model_matrix;

						//Send matrix uniforms to GPU
						let mat_locs = [mvp_location, model_matrix_location];
						let mats = [mvp, mesh.model_matrix];
						for i in 0..mat_locs.len() {
							gl::UniformMatrix4fv(mat_locs[i], 1, gl::FALSE, &flatten_glm(&mats[i]) as *const GLfloat);
						}

						//Send vector uniforms to GPU
						let vec_locs = [light_position_location, view_position_location];
						let vecs = [light_position, view_positions[i]];
						for i in 0..vec_locs.len() {
							let pos = [vecs[i].x, vecs[i].y, vecs[i].z, 1.0];
							gl::Uniform4fv(vec_locs[i], 1, &pos as *const GLfloat);
						}

						//Bind the mesh's texture
						gl::BindTexture(gl::TEXTURE_2D, mesh.texture);

						//Bind the mesh's vertex array object
						gl::BindVertexArray(mesh.vao);

						//Draw call
						gl::DrawElements(gl::TRIANGLES, mesh.indices_count, gl::UNSIGNED_SHORT, ptr::null());
					}
				}

				//Submit render to HMD
				submit_to_hmd(eyes[i], &openvr_compositor, &openvr_texture_handle);
			}
		}

		window.render_context().swap_buffers();
		glfw.poll_events();
	}

	//Shut down OpenVR
	if let Some(ctxt) = openvr_context {
		unsafe {
			ctxt.shutdown();
		}
	}
}
