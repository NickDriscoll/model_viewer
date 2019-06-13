extern crate gl;
extern crate nalgebra_glm as glm;
use glfw::{Action, Context, Key, WindowMode, WindowEvent};
use openvr::{ApplicationType, button_id, ControllerState, Eye, System, RenderModels, TrackedControllerRole, TrackedDevicePose};
use openvr::compositor::texture::{ColorSpace, Handle, Texture};
use nfd::Response;
use std::os::raw::c_void;
use std::fs::File;
use std::io::BufReader;
use std::thread;
use std::sync::mpsc;
use obj;
use rand::random;
use crate::structs::*;
use crate::glutil::*;
use self::gl::types::*;

mod structs;
mod glutil;

const NEAR_Z: f32 = 0.25;
const FAR_Z: f32 = 50.0;

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
		if let Some(i) = mesh_index {
			if let Some(mesh) = &mut meshes[i] {
				mesh.model_matrix = controller_model_matrix;
			}
		}
	}
}

fn load_controller_meshes<'a>(openvr_system: &Option<System>, openvr_rendermodels: &Option<RenderModels>, meshes: &mut OptionVec<Mesh>, index: u32, program: GLuint) -> [Option<usize>; 2] {
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

			let mesh = Mesh::new(vao, glm::translation(&glm::vec3(0.0, -1.0, 0.0)), program, None, model.indices().len() as i32);
			let left_index = Some(meshes.insert(mesh));

			let mesh = Mesh::new(vao, glm::translation(&glm::vec3(0.0, -1.0, 0.0)), program, None, model.indices().len() as i32);
			let right_index = Some(meshes.insert(mesh));

			result = [left_index, right_index];
		}
	}
	result
}

fn pressed_this_frame(state: &ControllerState, p_state: &ControllerState, flag: u32) -> bool {
	state.button_pressed & (1 as u64) << flag != 0 && p_state.button_pressed & (1 as u64) << flag == 0
}

fn get_mesh_origin(mesh: &Option<Mesh>) -> glm::TVec4<f32> {
	match mesh {
		Some(mesh) => {
			mesh.model_matrix * glm::vec4(0.0, 0.0, 0.0, 1.0)
		}
		None => {
			println!("Couldn't return mesh origin cause it was \"None\"");
			glm::vec4(0.0, 0.0, 0.0, 1.0)
		}
	}
}

fn main() {
	//Get the OpenVR context
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

	//Using OpenGL 3.3 core. but that could change
	glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
	glfw.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));

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
	let texture_program = unsafe { compile_program_from_files("shaders/vertex_texture.glsl", "shaders/fragment_texture.glsl") };

	//Get mvp uniform location
	let mvp_location = unsafe { get_uniform_location(texture_program, "mvp") };
	let model_matrix_location = unsafe { get_uniform_location(texture_program, "model_matrix") };

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

	//Spawn thread to load brick texture
	let tx = texture_tx.clone();
	thread::spawn( move || {
		tx.send(image_data_from_path("textures/bricks.jpg")).unwrap();
	});

	//Textures
	let checkerboard_texture = unsafe { load_texture("textures/checkerboard.jpg") };
	let mut brick_texture = None;

	//OptionVec of meshes
	let mut meshes: OptionVec<Mesh> = OptionVec::with_capacity(5);

	//Create the floor
	let floor_mesh_index = unsafe {
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
		let mesh = Mesh::new(vao, glm::scaling(&glm::vec3(5.0, 5.0, 5.0)),
							 texture_program, Some(checkerboard_texture), indices.len() as i32);
		meshes.insert(mesh)
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

	let mut camera_position = glm::vec3(0.0, -1.0, -1.0);
	let mut camera_velocity = glm::vec3(0.0, 0.0, 0.0);
	let mut camera_fov = 90.0;
	let mut camera_fov_delta = 0.0;
	let camera_speed = 0.05;

	type ModelLoadPacket = (Vec<f32>, Vec<u16>);
	let (load_tx, load_rx) = mpsc::channel::<ModelLoadPacket>();

	//Main loop
	while !window.should_close() {
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
														  index,
														  texture_program);

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
			if let Ok(pack) = load_rx.try_recv() {
				let vao = unsafe { create_vertex_array_object(&pack.0, &pack.1) };
				let mesh = Mesh::new(vao, glm::translation(&glm::vec3(0.0, 0.8, 0.0)) * glm::scaling(&glm::vec3(0.1, 0.1, 0.1)), texture_program, brick_texture, pack.1.len() as i32);

				//Delete old mesh if there is one
				if let Some(i) = loaded_mesh_index {
					meshes[i] = None;
				}
				loaded_mesh_index = Some(meshes.insert(mesh));
				loading_model_flag = false;
			}
		}

		//Check if the cube's texture has been loaded
		if loading_brick_texture_flag {
			//Check if the cube's texture is loaded yet
			if let Ok((data, width, height)) = texture_rx.try_recv() {
				let image_data = (data, width, height);
				brick_texture = unsafe { Some(load_texture_from_data(image_data)) };

				let mesh_indices = [loaded_mesh_index];

				for index in &mesh_indices {
					if let Some(i) = index {					
						if let Some(ref mut mesh) = &mut meshes[*i] {
							mesh.texture = brick_texture;
						}						
					}
				}

				loading_brick_texture_flag = false;
			}
		}

		//Handle window events
		for (_, event) in glfw::flush_messages(&events) {
			match event {
				WindowEvent::Close => {
					window.set_should_close(true);
				}
				WindowEvent::Key(key, _, Action::Press, ..) => {
					match key {
						Key::W => {
							camera_velocity.z = camera_speed;
						}
						Key::S => {
							camera_velocity.z = -camera_speed;
						}
						Key::A => {
							camera_velocity.x = camera_speed;
						}
						Key::D => {
							camera_velocity.x = -camera_speed;
						}
						Key::O => {
							camera_fov_delta = -1.0;
						}
						Key::P => {
							camera_fov_delta = 1.0;
						}
						Key::L => {
							let tx = load_tx.clone();
							thread::spawn( move || {
								//Invoke file selection dialogue
								let path = match nfd::open_file_dialog(None, None).unwrap() {
									Response::Okay(filename) => {
										filename
									}
									_ => { return }
								};

								let model: obj::Obj = match obj::load_obj(BufReader::new(File::open(path).unwrap())) {
									Ok(m) => {
										m
									}
									Err(e) => {
										println!("{:?}", e);
										return
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

								tx.send((vert_data, model.indices)).unwrap();
							});
							loading_model_flag = true;
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
							camera_velocity.x = 0.0;
						}
						Key::W | Key::S => {
							camera_velocity.z = 0.0;
						}
						Key::O | Key::P => {
							camera_fov_delta = 0.0;
							println!("fov is now {}", camera_fov);
						}
						_ => {}
					}
				}
				_ => {}
			}
		}

		//Handle controller input
		for i in 0..Controllers::NUMBER_OF_CONTROLLERS {
			if let (Some(mesh_index),
					Some(loaded_index),
					Some(state),
					Some(p_state)) = (controllers.mesh_indices[i],
									  loaded_mesh_index,
									  controllers.states[i],
									  controllers.previous_states[i]) {

				//If the trigger was pulled this frame
				if pressed_this_frame(&state, &p_state, button_id::STEAM_VR_TRIGGER) {
					//Grab the object the controller is currently touching, if there is one

					let controller_origin = get_mesh_origin(&meshes[mesh_index as usize]);
					let loaded_origin = get_mesh_origin(&meshes[loaded_index]);

					//Get distance from controller_origin to loaded_origin
					let dist = f32::sqrt(f32::powi(controller_origin.x - loaded_origin.x, 2) +
										 f32::powi(controller_origin.y - loaded_origin.y, 2) +
										 f32::powi(controller_origin.z - loaded_origin.z, 2));

					if dist < loaded_sphere_radius {
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
					loaded_bound_controller_index = None;
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

						//Need to return inverse(hmd_to_absolute * eye_to_hmd)
						(glm::affine_inverse(hmd_to_absolute * left_eye_to_hmd),
						 glm::affine_inverse(hmd_to_absolute * right_eye_to_hmd),
						 glm::affine_inverse(hmd_to_absolute))
					}
					None => {						
						//Create a matrix that gets a decent view of the scene
						let view_matrix = glm::translation(&camera_position);
						(glm::identity(), glm::identity(), view_matrix)
					}
				}
			}
			None => {
				//Create a matrix that gets a decent view of the scene
				let view_matrix = glm::translation(&camera_position);
				(glm::identity(), glm::identity(), view_matrix)
			}
		};

		//Get projection matrices
		let p_mat = glm::perspective(aspect_ratio, f32::to_radians(camera_fov), NEAR_Z, FAR_Z);
		let p_matrices = match openvr_system {
			Some(ref sys) => {
				(get_projection_matrix(sys, Eye::Left), get_projection_matrix(sys, Eye::Right), p_mat)
			}
			None => {
				(glm::identity(), glm::identity(), p_mat)
			}
		};

		//Update simulation
		ticks += 0.02;

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

		//Update the camera
		camera_position += camera_velocity;
		camera_fov += camera_fov_delta;

		controllers.previous_states = controllers.states;

		//Rendering code
		unsafe {
			//Set up to render on texture
			gl::BindFramebuffer(gl::FRAMEBUFFER, vr_render_target);
			gl::Viewport(0, 0, render_target_size.0 as GLsizei, render_target_size.1 as GLsizei);

			//Set clear color
			gl::ClearColor(0.53, 0.81, 0.92, 1.0);

			//Render left eye
			render_scene(&mut meshes, p_matrices.0, v_matrices.0, mvp_location, model_matrix_location);

			//Send to HMD
			submit_to_hmd(Eye::Left, &openvr_compositor, &openvr_texture_handle);

			//Render right eye
			render_scene(&mut meshes, p_matrices.1, v_matrices.1, mvp_location, model_matrix_location);

			//Send to HMD
			submit_to_hmd(Eye::Right, &openvr_compositor, &openvr_texture_handle);

			//Unbind the vr render target so that we can draw to the window
			gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
			gl::Viewport(0, 0, window_size.0 as GLsizei, window_size.1 as GLsizei);

			//Draw companion view
			render_scene(&mut meshes, p_matrices.2, v_matrices.2, mvp_location, model_matrix_location);
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
