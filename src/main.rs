extern crate gl;
extern crate nalgebra_glm as glm;
use glfw::{Action, Context, CursorMode, Key, MouseButton, WindowMode, WindowEvent};
use openvr::{ApplicationType, button_id, ControllerState, Eye, System, RenderModels, TrackedControllerRole};
use openvr::compositor::texture::{ColorSpace, Handle, Texture};
use nfd::Response;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::sync::mpsc;
use std::ptr;
use obj;
use rand::random;
use noise::{NoiseFn, OpenSimplex, Seedable};
use crate::structs::*;
use crate::glutil::*;
use self::gl::types::*;

mod structs;
mod glutil;

//The distances of the near and far clipping planes from the origin
const NEAR_Z: f32 = 0.05;
const FAR_Z: f32 = 50.0;

type MeshArrays = (Vec<f32>, Vec<u16>);

enum WorkOrder<'a> {
	Image(&'a str)
}

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

//This returns the Mesh struct associated with the OpenVR model
fn load_openvr_mesh(openvr_system: &Option<System>, openvr_rendermodels: &Option<RenderModels>, index: u32) -> Option<Mesh> {
	let mut result = None;
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
			let vao = unsafe { create_vertex_array_object(&vertices, model.indices(), &[3, 3, 2]) };

			//Create texture
			let t = unsafe { load_texture_from_data(([128, 128, 128].to_vec(), 1, 1, gl::RGB)) };

			let mut mesh = Mesh::new(vao, glm::identity(), "", model.indices().len() as GLsizei);
			mesh.texture = t;

			result = Some(mesh);
		}
	}
	result
}

fn pressed_this_frame(state: &ControllerState, p_state: &ControllerState, flag: u32) -> bool {
	state.button_pressed & (1 as u64) << flag != 0 && p_state.button_pressed & (1 as u64) << flag == 0
}

fn released_this_frame(state: &ControllerState, p_state: &ControllerState, flag: u32) -> bool {
	state.button_pressed & (1 as u64) << flag == 0 && p_state.button_pressed & (1 as u64) << flag != 0
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

fn uniform_scale(scale: f32) -> glm::TMat4<f32> {
	glm::scaling(&glm::vec3(scale, scale, scale))
}

fn main() {
	//Initialize OpenVR
	let openvr_context = unsafe {
		match openvr::init(ApplicationType::Scene) {
			Ok(ctxt) => {
				Some(ctxt)
			}
			Err(e) => {
				println!("OpenVR initialization error: {}", e);
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
		Some(ref sys) => { sys.recommended_render_target_size() }
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

	//Configure window
	window.set_key_polling(true);
	window.set_framebuffer_size_polling(true);

	//Load all OpenGL function pointers
	gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);

	//Enable and configure depth testing and enable backface culling
	unsafe {
		gl::Enable(gl::DEPTH_TEST);
		gl::DepthFunc(gl::LESS);
		gl::Enable(gl::CULL_FACE);
		gl::PolygonMode(gl::FRONT_AND_BACK, gl::LINE);
	}

	//Compile shader program
	let nonluminous_shader = unsafe { compile_program_from_files("shaders/nonluminous_vertex.glsl", "shaders/nonluminous_fragment.glsl") };

	//Get locations of program uniforms
	let mvp_location = unsafe { get_uniform_location(nonluminous_shader, "mvp") };
	let model_matrix_location = unsafe { get_uniform_location(nonluminous_shader, "model_matrix") };
	let light_position_location = unsafe { get_uniform_location(nonluminous_shader, "light_position") };
	let view_position_location = unsafe { get_uniform_location(nonluminous_shader, "view_position") };
	let shininess_location = unsafe { get_uniform_location(nonluminous_shader, "shininess") };

	//Setup the VR rendering target
	let vr_render_target = unsafe { create_vr_render_target(&render_target_size) };
	let openvr_texture_handle = Texture {
		handle: Handle::OpenGLTexture(vr_render_target as usize),
		color_space: ColorSpace::Auto
	};

	//Texture loading channel
	let (texture_tx, texture_rx) = mpsc::channel::<(&str, ImageData)>();
	let (order_tx, order_rx) = mpsc::channel::<WorkOrder>();

	//The channel for sending 3D models between threads
	let (load_tx, load_rx) = mpsc::channel::<Option<MeshArrays>>();

	//Map of texture paths to texture ids
	let mut textures: HashMap<String, GLuint> = HashMap::new();

	//Spawn thread to load textures
	thread::spawn( move || {
		loop {
			match order_rx.recv() {
				Ok(WorkOrder::Image(path)) => {
					println!("[Worker]\tLoading image: {}...", path);
					texture_tx.send((path, image_data_from_path(path))).unwrap();
					println!("[Worker]\tDone!");
				}
				Err(_) => {
					return;
				}
			}
		}
	});
	order_tx.send(WorkOrder::Image("textures/checkerboard.jpg")).unwrap();
	order_tx.send(WorkOrder::Image("textures/bricks.jpg")).unwrap();

	//OptionVec of meshes
	let mut meshes = OptionVec::with_capacity(10);

	//Set up the simplex noise generator
	let mut simplex_generator = OpenSimplex::new();
	let simplex_seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() * 1000;
	println!("Seed used for simplex noise: {}", simplex_seed);
	simplex_generator = simplex_generator.set_seed(simplex_seed as u32);

	//Create the large tessellated plane
	unsafe {
		const ELEMENT_STRIDE: usize = 8;
		//const WIDTH: usize = 100;
		const WIDTH: usize = 8;
		const TRIS: usize = (WIDTH - 1) * (WIDTH - 1) * 2;
		const AMPLITUDE: f32 = 20.0;

		//Buffers to be filled
		let mut verts = [0.0; WIDTH*WIDTH*ELEMENT_STRIDE];
		let mut indices: Vec<u16> = Vec::with_capacity(TRIS * 3);

		for i in (0..verts.len()).step_by(ELEMENT_STRIDE) {
			let xpos: usize = (i / ELEMENT_STRIDE) % WIDTH;
			let ypos: usize = (i / ELEMENT_STRIDE) / WIDTH;

			verts[i] = (xpos as f32 / (WIDTH - 1) as f32) as f32 - 0.5;
			verts[i + 2] = (ypos as f32 / (WIDTH - 1) as f32) as f32 - 0.5;
			//verts[i + 1] = AMPLITUDE * simplex_generator.get([verts[i] as f64, verts[i + 2] as f64]) as f32;			
			verts[i + 3] = 0.0;
			verts[i + 4] = 1.0;
			verts[i + 5] = 0.0;
			verts[i + 6] = WIDTH as f32 * (xpos as f32 / (WIDTH - 1) as f32) as f32;
			verts[i + 7] = WIDTH as f32 * (ypos as f32 / (WIDTH - 1) as f32) as f32;
		}

		//This loop executes once per subsquare on the plane
		for i in 0..((WIDTH-1)*(WIDTH-1)) {
			let xpos = i % (WIDTH-1);
			let ypos = i / (WIDTH-1);

			//Push indices for bottom-left triangle
			indices.push((xpos + ypos * WIDTH) as u16);
			indices.push((xpos + ypos * WIDTH + WIDTH) as u16);
			indices.push((xpos + ypos * WIDTH + 1) as u16);
			
			//Push indices for top-right triangle
			indices.push((xpos + ypos * WIDTH + 1) as u16);
			indices.push((xpos + ypos * WIDTH + WIDTH) as u16);
			indices.push((xpos + ypos * WIDTH + WIDTH + 1) as u16);
		}

		let vao = create_vertex_array_object(&verts, &indices, &[3, 3, 2]);
		let model_matrix = glm::scaling(&glm::vec3(WIDTH as f32, 1.0, WIDTH as f32));
		meshes.insert(Mesh::new(vao, model_matrix, "textures/checkerboard.jpg", indices.len() as i32));
	};

	//Create the sphere that represents the light source
	let mut light_position = glm::vec4(0.0, 1.0, 0.0, 1.0);
	let sphere_index = unsafe {
		match load_wavefront_obj("models/sphere.obj") {
			Some(obj) => {
				let vao = create_vertex_array_object(&obj.0, &obj.1, &[3, 3, 2]);
				let t = glm::vec4_to_vec3(&light_position);
				let mesh = Mesh::new(vao, glm::translation(&t) * uniform_scale(0.01), "textures/bricks.jpg", obj.1.len() as i32);
				Some(meshes.insert(mesh))
			}
			None => { None }
		}
	};

	let model_bounding_sphere_radius = 0.20;
	let mut bound_controller_indices = Vec::new();
	let mut model_indices = Vec::new();
	let mut model_to_controller_matrices = Vec::new();

	//Initialize the struct of arrays containing controller related state
	let mut controllers = Controllers::new();

	let mut hmd_mesh_index = None;

	//Gameplay state
	let mut ticks = 0.0;

	//Camera state
	let mut camera = Camera::new(glm::vec3(0.0, -1.0, -1.0));

	let mut last_rbutton_state = window.get_mouse_button(MouseButton::Button2);

	//The instant recorded at the beginning of last frame
	let mut last_frame_instant = Instant::now();

	let mut tracking_to_world: glm::TMat4<f32> = glm::identity();

	//Set up data for each render pass
	let framebuffers = [vr_render_target, vr_render_target, 0];
	let eyes = [Some(Eye::Left), Some(Eye::Right), None];
	let mut sizes = [render_target_size, render_target_size, window_size];

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
		if let None = controllers.mesh_indices[0] {
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
				mesh.shininess = 128.0;
				hmd_mesh_index = Some(meshes.insert(mesh));
			}
		}

		//Get VR pose data
		let render_poses = match &openvr_compositor {
			Some(comp) => {
				Some(comp.wait_get_poses().unwrap().render)
			}
			None => {
				None
			}
		};

		//Get controller state structs
		for i in 0..Controllers::NUMBER_OF_CONTROLLERS {
			if let (Some(index), Some(sys)) = (controllers.device_indices[i], &openvr_system) {
				controllers.states[i] = sys.controller_state(index);
			}
		}

		//Check if a new model has been loaded
		if let Ok(Some(package)) = load_rx.try_recv() {
			let vao = unsafe { create_vertex_array_object(&package.0, &package.1, &[3, 3, 2]) };
			let mesh = Mesh::new(vao, glm::translation(&glm::vec3(0.0, 0.8, 0.0)) * uniform_scale(0.1), "textures/bricks.jpg", package.1.len() as i32);
			model_indices.push(Some(meshes.insert(mesh)));
			bound_controller_indices.push(None);
			model_to_controller_matrices.push(glm::identity());
		}

		//Check if there are any textures to be received from the worker thread
		if let Ok(texture_data) = texture_rx.try_recv() {
			let tex_id = unsafe { load_texture_from_data(texture_data.1) };
			textures.insert(texture_data.0.to_string(), tex_id);
		}

		//Connect all meshes to their textures
		for option_mesh in meshes.iter_mut() {
			if let Some(mesh) = option_mesh {
				if mesh.texture == 0 {
					if let Some(tex_id) = textures.get(&mesh.texture_path) {
						mesh.texture = *tex_id;
					}
				}
			}
		}

		//Handle window and keyboard events
		for (_, event) in glfw::flush_messages(&events) {
			match event {
				WindowEvent::Close => {
					window.set_should_close(true);
				}
				WindowEvent::FramebufferSize(width, height) => {
					window_size = (width as u32, height as u32);
					sizes[2] = window_size;
					aspect_ratio = window_size.0 as f32 / window_size.1 as f32;
				}
				WindowEvent::Key(key, _, Action::Press, ..) => {
					match key {
						Key::W => { camera.velocity = glm::vec3(0.0, 0.0, Camera::SPEED); }
						Key::S => { camera.velocity = glm::vec3(0.0, 0.0, -Camera::SPEED); }
						Key::A => { camera.velocity = glm::vec3(Camera::SPEED, 0.0, 0.0); }
						Key::D => { camera.velocity = glm::vec3(-Camera::SPEED, 0.0, 0.0); }
						Key::O => { camera.fov_delta = -Camera::FOV_SPEED; }
						Key::P => { camera.fov_delta = Camera::FOV_SPEED; }
						Key::I => { camera.fov = 90.0; }
						Key::Escape => { window.set_should_close(true); }
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
						}
						Key::Space => {
							camera.attached_to_hmd = !camera.attached_to_hmd;

							if let Some(mesh) = get_mesh(&mut meshes, hmd_mesh_index) {
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
						Key::O | Key::P => {
							camera.fov_delta = 0.0;
							println!("Field of view is now {} degrees", camera.fov);
						}
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

		//If the cursor is currently captured
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
					Some(p_state),
					Some(sys),
					Some(poses)) = (  controllers.device_indices[i],
									  controllers.mesh_indices[i],
									  controllers.states[i],
									  controllers.previous_states[i],
									  &openvr_system,
									  render_poses) {

				controllers.collided_with[i].clear();
				for j in 0..model_indices.len() {
					//If the trigger was pulled this frame, grab the object the controller is currently touching, if there is one
					if let Some(loaded_index) = model_indices[j] {
						//Make controller vibrate if it collides with something
						let is_colliding = glm::distance(&get_mesh_origin(&meshes[mesh_index as usize]), &get_mesh_origin(&meshes[loaded_index])) < model_bounding_sphere_radius;

						if is_colliding {
							controllers.colliding_with[i].push(loaded_index);

							//Check if controller was colliding with that model last frame
							if !controllers.collided_with[i].contains(&loaded_index) {
								println!("Triggering pulse");
								//sys.trigger_haptic_pulse(device_index, 0, 3999);
							}
						}

						if pressed_this_frame(&state, &p_state, button_id::GRIP) && is_colliding {
							//Set the controller's mesh as the mesh the cube mesh is "bound" to
							bound_controller_indices[j] = Some(i);

							//Calculate the cube-space to controller-space matrix aka inverse(controller.model_matrix) * cube.model_matrix
							if let (Some(cont_mesh), Some(loaded_mesh)) = (&meshes[mesh_index], &meshes[loaded_index]) {
								model_to_controller_matrices[j] = glm::affine_inverse(cont_mesh.model_matrix) * loaded_mesh.model_matrix;
							}
						}
					}
					
					//If the trigger was released this frame
					if released_this_frame(&state, &p_state, button_id::GRIP) {
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

				//Handle left-hand/movement controls
				if i == 0 {
					let controller_to_tracking = openvr_to_mat4(*poses[device_index as usize].device_to_absolute_tracking());

					if state.axis[0].x != 0.0 || state.axis[0].y != 0.0 {
						let mut temp = tracking_to_world * controller_to_tracking * glm::vec4(state.axis[0].x, 0.0, -state.axis[0].y, 0.0);
						let len = glm::length(&temp);
						temp.y = 0.0;
						temp *= len / glm::length(&temp);
						movement_vector += scale * temp;
					}
				}
				
				tracking_to_world = glm::translation(&glm::vec4_to_vec3(&movement_vector)) * tracking_to_world;
			}

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
						camera.get_freecam_matrix()
					};

					//Need to return inverse(tracking_to_world * hmd_to_tracking * eye_to_hmd)
					[glm::affine_inverse(tracking_to_world * hmd_to_tracking * left_eye_to_hmd),
					 glm::affine_inverse(tracking_to_world * hmd_to_tracking * right_eye_to_hmd),
					 companion_v_mat]
			}
			_ => {				
				//Create a matrix that gets a decent view of the scene
				[glm::identity(), glm::identity(), camera.get_freecam_matrix()]
			}
		};

		//Get view positions
		let view_positions = {
			let mut temp = Vec::with_capacity(v_matrices.len());
			for matrix in &v_matrices {
				temp.push(get_frame_origin(&glm::affine_inverse(*matrix)));
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

		//Just a scaled time value
		ticks += 2.0 * seconds_elapsed;

		//Update the camera
		camera.position += glm::vec4_to_vec3(&(seconds_elapsed * (glm::affine_inverse(v_matrices[2]) * glm::vec3_to_vec4(&camera.velocity))));
		camera.fov += camera.fov_delta * seconds_elapsed;

		//Update HMD Mesh's model matrix
		if let Some(poses) = render_poses {
			let hmd_to_absolute = openvr_to_mat4(*poses[0].device_to_absolute_tracking());
			if let Some(mesh) = get_mesh(&mut meshes, hmd_mesh_index) {
				mesh.model_matrix = tracking_to_world * hmd_to_absolute;
			}
		}

		//Update the controller's model matrices
		if let Some(poses) = render_poses {
			for i in 0..Controllers::NUMBER_OF_CONTROLLERS {
				if let Some(index) = &controllers.device_indices[i] {
					let controller_model_matrix = openvr_to_mat4(*poses[*index as usize].device_to_absolute_tracking());
					if let Some(mesh) = get_mesh(&mut meshes, controllers.mesh_indices[i]) {
						mesh.model_matrix = tracking_to_world * controller_model_matrix;
					}
				}
			}
		}

		//If a model is being grabbed, place it in the right spot
		for i in 0..bound_controller_indices.len() {
			if let Some(index) = bound_controller_indices[i] {
				if let (Some(mesh_index), Some(load_index)) = (controllers.mesh_indices[index], model_indices[i]) {				
					if let (Some(loaded), Some(controller)) = meshes.two_mut_refs(load_index, mesh_index) {
						loaded.model_matrix = controller.model_matrix * model_to_controller_matrices[i];
					}
				}
			}
		}

		//Make the light bob up and down
		if let Some(mesh) = get_mesh(&mut meshes, sphere_index) {
			mesh.model_matrix = glm::translation(&glm::vec3(0.0, 0.5*f32::sin(ticks*0.2) + 0.8, 0.0)) * uniform_scale(0.1);
			light_position = get_frame_origin(&mesh.model_matrix);
		}

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
						if mesh.render_pass_visibilities[i] && mesh.texture != 0 {
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

							//Send float uniform to GPU
							gl::Uniform1f(shininess_location, mesh.shininess);

							//Bind the mesh's texture
							gl::BindTexture(gl::TEXTURE_2D, mesh.texture);

							//Bind the mesh's vertex array object
							gl::BindVertexArray(mesh.vao);

							//Draw call
							gl::DrawElements(gl::TRIANGLES, mesh.indices_count, gl::UNSIGNED_SHORT, ptr::null());
						}
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
