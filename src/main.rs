extern crate gl;
extern crate nalgebra_glm as glm;
use glfw::Action;
use glfw::Context;
use glfw::Key;
use glfw::WindowMode;
use glfw::WindowEvent;
use self::gl::types::*;
use openvr::ApplicationType;
use openvr::button_id;
use openvr::ControllerState;
use openvr::Eye;
use openvr::System;
use openvr::compositor::texture::{ColorSpace, Handle, Texture};
use openvr::RenderModels;
use openvr::TrackedControllerRole;
use openvr::TrackedDevicePose;
use nfd::Response;
use std::os::raw::c_void;
use std::fs::File;
use std::io::BufReader;
use std::thread;
use std::sync::mpsc;
use obj::*;
use rand::random;
use crate::structs::*;
use crate::glutil::*;

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
			const ELEMENT_STRIDE: usize = 5;
			let mut vertices = Vec::with_capacity(ELEMENT_STRIDE * model.vertices().len());
			for vertex in model.vertices() {
				vertices.push(vertex.position[0]);
				vertices.push(vertex.position[1]);
				vertices.push(vertex.position[2]);
				vertices.push(vertex.texture_coord[0]);
				vertices.push(vertex.texture_coord[1]);
			}

			//Create vao
			let vao = unsafe { create_vertex_array_object(&vertices, model.indices()) };

			let mesh = Mesh::new(vao, glm::translation(&glm::vec3(0.0, -1.0, 0.0)), program, None, model.indices().len() as i32);
			let left_index = Some(meshes.insert(mesh));

			let mesh = Mesh::new(vao, glm::translation(&glm::vec3(0.0, -1.0, 0.0)), program, None, model.indices().len() as i32);
			let right_index = Some(meshes.insert(mesh));

			println!("Loaded controller mesh");

			result = [left_index, right_index];
		}
	}
	result
}

fn pressed_this_frame(state: &ControllerState, p_state: &ControllerState, flag: u32) -> bool {
	state.button_pressed & (1 as u64) << flag != 0 && p_state.button_pressed & (1 as u64) << flag == 0
}

fn main() {
	//Init OpenVR
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

	//Get the OpenVR system and compositor
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

	//Setup the VR rendering target
	let vr_render_target = unsafe {
		let mut render_target = 0;
		gl::GenFramebuffers(1, &mut render_target);
		gl::BindFramebuffer(gl::FRAMEBUFFER, render_target);

		//Create the texture that will be rendered to
		let mut vr_render_texture = 0;
		gl::GenTextures(1, &mut vr_render_texture);
		gl::BindTexture(gl::TEXTURE_2D, vr_render_texture);
		gl::TexImage2D(gl::TEXTURE_2D, 0,
						   gl::RGB as i32,
						   render_target_size.0 as GLsizei,
						   render_target_size.1 as GLsizei,
						   0,
						   gl::RGB,
						   gl::UNSIGNED_BYTE,
						   0 as *const c_void);
		gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
		gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);

		//Create depth buffer
		let mut depth_buffer = 0;
		gl::GenRenderbuffers(1, &mut depth_buffer);
		gl::BindRenderbuffer(gl::RENDERBUFFER, depth_buffer);
		gl::RenderbufferStorage(gl::RENDERBUFFER,
								gl::DEPTH_COMPONENT,
								render_target_size.0 as GLsizei,
								render_target_size.1 as GLsizei);
		gl::FramebufferRenderbuffer(gl::FRAMEBUFFER, gl::DEPTH_ATTACHMENT, gl::RENDERBUFFER, depth_buffer);

		//Configure framebuffer
		gl::FramebufferTexture(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, vr_render_texture, 0);
		let drawbuffers = [gl::COLOR_ATTACHMENT0];
		gl::DrawBuffers(1, &drawbuffers as *const u32);
		if gl::CheckFramebufferStatus(gl::FRAMEBUFFER) != gl::FRAMEBUFFER_COMPLETE {
			println!("Framebuffer wasn't complete");
		}
		gl::BindRenderbuffer(gl::RENDERBUFFER, 0);
		gl::BindTexture(gl::TEXTURE_2D, 0);
		gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
		render_target
	};
	let openvr_texture_handle = {
		Texture {
			handle: Handle::OpenGLTexture(vr_render_target as usize),
			color_space: ColorSpace::Auto
		}
	};

	unsafe {
		gl::Enable(gl::DEPTH_TEST);
		gl::DepthFunc(gl::LESS);
		gl::Enable(gl::CULL_FACE);
	}

	//OptionVec of meshes
	let mut meshes: OptionVec<Mesh> = OptionVec::with_capacity(5);

	//Create the floor
	let floor_mesh_index = unsafe {
		let vertices = [
			//Positions					//Tex coords
			-0.5f32, 0.0, -0.5,			0.0, 0.0,
			-0.5, 0.0, 0.5,				0.0, 4.0,
			0.5, 0.0, -0.5,				4.0, 0.0,
			0.5, 0.0, 0.5,				4.0, 4.0
		];
		let indices = [
			0u16, 1, 2,
			1, 3, 2
		];
		let vao = create_vertex_array_object(&vertices, &indices);
		let mesh = Mesh::new(vao, glm::scaling(&glm::vec3(5.0, 5.0, 5.0)),
							 texture_program, Some(load_texture("textures/checkerboard.jpg")), indices.len() as i32);
		meshes.insert(mesh)
	};

	//Create the cube's mesh
	let cube_mesh_index = unsafe {
		let vertices = [
			//Position data 				//Tex coords
			-0.5f32, -0.5, 0.5,				0.0, 1.0,
			-0.5, 0.5, 0.5,					-1.0, 1.0,
			0.5, 0.5, 0.5,					2.0, 1.0,
			0.5, -0.5, 0.5,					1.0, 1.0,
			-0.5, -0.5, -0.5,				0.0, 0.0,
			-0.5, 0.5, -0.5,				-1.0, 0.0,
			0.5, 0.5, -0.5,					1.0, -1.0,
			0.5, -0.5, -0.5,				1.0, 0.0
		];
		let indices = [
			1u16, 0, 3,
			3, 2, 1,
			2, 3, 7,
			7, 6, 2,
			3, 0, 4,
			4, 7, 3,
			6, 5, 1,
			1, 2, 6,
			4, 5, 6,
			6, 7, 4,
			5, 4, 0,
			0, 1, 5
		];
		let vao = create_vertex_array_object(&vertices, &indices);
		let mesh = Mesh::new(vao, glm::translation(&glm::vec3(0.0, 1.0, 0.0)) * glm::scaling(&glm::vec3(0.25, 0.25, 0.25)), texture_program, Some(load_texture("textures/bricks.jpg")), indices.len() as i32);
		meshes.insert(mesh)
	};

	//Create the cube's bounding sphere
	let cube_sphere_radius = 0.20;
	let mut cube_bound_controller_mesh = None;

	//Controller related variables
	let mut controllers = Controllers::new();

	/*
	const Controllers::NUMBER_OF_CONTROLLERS: usize = 2;
	let mut controller_indices = [None; Controllers::NUMBER_OF_CONTROLLERS];
	let mut controller_mesh_indices = [None; Controllers::NUMBER_OF_CONTROLLERS];
	let mut controller_states = [None; Controllers::NUMBER_OF_CONTROLLERS];
	let mut previous_controller_states: [Option<ControllerState>; Controllers::NUMBER_OF_CONTROLLERS] = [None; Controllers::NUMBER_OF_CONTROLLERS];
	*/

	//Gameplay state
	let mut ticks = 0.0;
	let mut camera_position = glm::vec3(0.0, -1.0, -1.0);
	let mut camera_velocity = glm::vec3(0.0, 0.0, 0.0);
	let mut camera_fov = 90.0;
	let mut camera_fov_delta = 0.0;
	let camera_speed = 0.05;

	let mut loaded_mesh_index: Option<usize> = None;
	let mut loading_model_flag = false;

	type ModelLoadPacket = (Vec<f32>, Vec<u16>);
	let (load_tx, load_rx) = mpsc::channel::<ModelLoadPacket>();

	//Main loop
	while !window.should_close() {
		//Find controllers if we haven't already
		if let Some(ref sys) = openvr_system {
			for i in 0..controllers.controller_indices.len() {
				if let None = controllers.controller_indices[i] {
					const ROLES: [TrackedControllerRole; 2] = [TrackedControllerRole::LeftHand, TrackedControllerRole::RightHand];
					controllers.controller_indices[i] = sys.tracked_device_index_for_controller_role(ROLES[i]);
				}
			}
		}

		//Load controller meshes if we haven't already
		if let None = controllers.controller_mesh_indices[0] {
			for i in 0..controllers.controller_indices.len() {
				if let Some(index) = controllers.controller_indices[i] {
					controllers.controller_mesh_indices = load_controller_meshes(&openvr_system,
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
			if let (Some(index), Some(sys)) = (controllers.controller_indices[i], &openvr_system) {
				controllers.controller_states[i] = sys.controller_state(index);
			}
		}

		//Check if a new model has been loaded
		if loading_model_flag {
			if let Ok(pack) = load_rx.try_recv() {
				let vao = unsafe { create_vertex_array_object(&pack.0, &pack.1) };
				let mesh = Mesh::new(vao, glm::scaling(&glm::vec3(0.2, 0.2, 0.2)), texture_program, None, pack.1.len() as i32);

				//Delete old mesh if there is one
				if let Some(i) = loaded_mesh_index {
					meshes[i] = None;
				}
				loaded_mesh_index = Some(meshes.insert(mesh));
				loading_model_flag = false;
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
									Response::OkayMultiple(_) => {
										return
									}
									Response::Cancel => { return }
								};

								let model: Obj = match load_obj(BufReader::new(File::open(path).unwrap())) {
									Ok(m) => {
										m
									}
									Err(e) => {
										println!("{:?}", e);
										return
									}
								};

								//Take loaded model and create a Mesh
								const ELEMENT_STRIDE: usize = 6;
								let mut vert_data = Vec::with_capacity(ELEMENT_STRIDE * model.vertices.len());
								for v in model.vertices {
									vert_data.push(v.position[0]);
									vert_data.push(v.position[1]);
									vert_data.push(v.position[2]);
									vert_data.push(random::<f32>());
									vert_data.push(random::<f32>());
								}

								let pack = (vert_data, model.indices);
								tx.send(pack).unwrap();
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
			if let (Some(mesh_index), Some(state), Some(p_state)) = (controllers.controller_mesh_indices[i], controllers.controller_states[i], controllers.previous_controller_states[i]) {
				if pressed_this_frame(&state, &p_state, button_id::STEAM_VR_TRIGGER) {
					//Grab the object the controller is currently touching, if there is one
					let controller_point = match &meshes[mesh_index as usize] {
						Some(mesh) => {
							mesh.model_matrix * glm::vec4(0.0, 0.0, 0.0, 1.0)
						}
						_ => {
							glm::vec4(0.0, 0.0, 0.0, 1.0)
						}
					};

					let cube_center = match &meshes[cube_mesh_index] {
						Some(mesh) => {
							mesh.model_matrix * glm::vec4(0.0, 0.0, 0.0, 1.0)
						}
						None => {
							println!("ERROR: Controller mesh not found despite controller being registered");
							glm::vec4(0.0, 0.0, 0.0, 1.0)
						}
					};

					//Get distance from controller_point to cube_center
					let dist = f32::sqrt(f32::powi(controller_point.x - cube_center.x, 2) +
										 f32::powi(controller_point.y - cube_center.y, 2) +
										 f32::powi(controller_point.z - cube_center.z, 2));

					if dist < cube_sphere_radius {
						cube_bound_controller_mesh = Some(mesh_index);
					}
				}

				if state.button_pressed & (1 as u64) << button_id::STEAM_VR_TRIGGER == 0 &&
				   p_state.button_pressed & (1 as u64) << button_id::STEAM_VR_TRIGGER != 0 {
					cube_bound_controller_mesh = None;
				}
			}
		}

		//Get view matrices for each eye
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
				attach_mesh_to_controller(&mut meshes, &poses, &controllers.controller_indices[i], controllers.controller_mesh_indices[i]);
			}
		}

		//If the cube is currently being grabbed, draw it at the grabbing controller's position
		if let Some(mesh_index) = cube_bound_controller_mesh {
			let indices = meshes.two_mut_refs(cube_mesh_index, mesh_index);
			if let (Some(cube), Some(controller)) = indices {
				cube.model_matrix = controller.model_matrix * glm::scaling(&glm::vec3(0.25, 0.25, 0.25));
			}
		}

		//Update the loaded mesh
		if let Some(index) = loaded_mesh_index {
			if let Some(mesh) = &mut meshes[index] {
				mesh.model_matrix = glm::rotation(ticks*0.5, &glm::vec3(0.0, 1.0, 0.0)) *
											 glm::scaling(&glm::vec3(0.2, 0.2, 0.2));				
			}
		}

		camera_position += camera_velocity;
		camera_fov += camera_fov_delta;

		controllers.previous_controller_states = controllers.controller_states;

		//Rendering code
		unsafe {
			//Set up to render on texture
			gl::BindFramebuffer(gl::FRAMEBUFFER, vr_render_target);
			gl::Viewport(0, 0, render_target_size.0 as GLsizei, render_target_size.1 as GLsizei);

			//Set clear color
			gl::ClearColor(0.53, 0.81, 0.92, 1.0);

			//Render left eye
			render_scene(&mut meshes, p_matrices.0, v_matrices.0, mvp_location);

			//Send to HMD
			submit_to_hmd(Eye::Left, &openvr_compositor, &openvr_texture_handle);

			//Render right eye
			render_scene(&mut meshes, p_matrices.1, v_matrices.1, mvp_location);

			//Send to HMD
			submit_to_hmd(Eye::Right, &openvr_compositor, &openvr_texture_handle);

			//Unbind the vr render target so that we can draw to the window
			gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
			gl::Viewport(0, 0, window_size.0 as GLsizei, window_size.1 as GLsizei);

			//Draw companion view
			render_scene(&mut meshes, p_matrices.2, v_matrices.2, mvp_location);
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
