extern crate gl;
extern crate nalgebra_glm as glm;
use glfw::{Action, Context, CursorMode, Key, MouseButton, WindowMode, WindowEvent};
use openvr::{ApplicationType, button_id, Eye, System, RenderModels, TrackedControllerRole, TrackedDevicePose};
use openvr::compositor::texture::{ColorSpace, Handle, Texture};
use nfd::Response;
use std::collections::HashMap;
use std::fs::{File, read_to_string};
use std::io::BufReader;
use std::os::raw::c_void;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::sync::mpsc;
use std::ptr;
use wavefront_obj::{mtl, obj};
use noise::{NoiseFn, OpenSimplex, Seedable};
use crate::structs::*;
use crate::glutil::*;
use self::gl::types::*;

//Including the other source files
mod structs;
mod glutil;

//The distances of the near and far clipping planes from the origin
const NEAR_Z: f32 = 0.05;
const FAR_Z: f32 = 800.0;

//Left eye, Right eye, Companion window
const RENDER_PASSES: usize = 3;

type MeshData = (Vec<f32>, Vec<u16>, Vec<GLsizei>, Vec<Option<mtl::Material>>);

//Things you can request the worker thread to do
enum WorkOrder<'a> {
	Image(&'a str),
	Model,
	Quit
}

//Things the worker thread can send back to the main thread
enum WorkResult<'a> {
	Image(&'a str, ImageData),
	Model(Option<MeshData>)
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
			
			let vao = unsafe { create_vertex_array_object(&vertices, model.indices(), &[3, 3, 2]) };
			let t = unsafe { load_texture_from_data(([25, 140, 15].to_vec(), 1, 1, gl::RGB)) };
			let mut mesh = Mesh::new(vao, glm::identity(), "", vec![0, model.indices().len() as GLsizei], None);
			mesh.texture = t;

			result = Some(mesh);
		}
	}
	result
}

fn get_frame_origin(something_to_world: &glm::TMat4<f32>) -> glm::TVec4<f32> {
	something_to_world * glm::vec4(0.0, 0.0, 0.0, 1.0)
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

fn load_wavefront_obj(path: &str) -> Option<MeshData> {
	//Gracefully exit if this is not an obj
	if path.split_at(path.len() - 3).1 != "obj" {
		println!("{} is not an obj file, dude.", path);
		return None;
	}

	//Load .obj file's contents as a string
	let obj_contents = match read_to_string(path) {
		Ok(s) => { s }
		Err(e) => {
			println!("{}", e);
			return None;
		}
	};

	//Load .mtl file's contents as a string
	let mtl_contents = match read_to_string(format!("{}.mtl", path.split_at(path.len() - 4).0)) {
		Ok(s) => { s }
		Err(e) => {
			println!("{}", e);
			return None;
		}
	};
	
	//Parse the Objects from the file
	let obj_set = match obj::parse(obj_contents) {
		Ok(m) => { m }
		Err(e) => {
			println!("{:?}", e);
			return None;
		}
	};

	//Parse the Materials from the file
	let mtl_set = match mtl::parse(mtl_contents) {
		Ok(m) => { m }
		Err(e) => {
			println!("{:?}", e);
			return None;
		}
	};

	//Transform the object into something the engine can actually use
	const BUFFER_SIZE: usize = 500;
	let mut index_map = HashMap::new();
	let mut vertices = Vec::with_capacity(BUFFER_SIZE);
	let mut indices = Vec::with_capacity(BUFFER_SIZE);
	let mut geometry_boundaries = Vec::with_capacity(BUFFER_SIZE);
	let mut materials_in_order = Vec::with_capacity(BUFFER_SIZE);
	let mut current_index = 0u16;
	let mut index_offset = 0;
	for object in obj_set.objects {
		for geo in &object.geometry {
			geometry_boundaries.push(indices.len() as GLsizei);

			//Copy the current material into materials_in_order
			match &geo.material_name {
				Some(name) => {
					for material in &mtl_set.materials {
						if *name == material.name {
							materials_in_order.push(Some(material.clone()));
							break;
						}
					}
				}
				None => {
					materials_in_order.push(None);
				}
			}

			for shape in &geo.shapes {
				match shape.primitive {
					obj::Primitive::Triangle(a, b, c) => {
						let verts = [a, b, c];
						for v in &verts {
							let pair = (v.0 + index_offset, v.2, v.1);
							match index_map.get(&pair) {
								Some(i) => {
									//This vertex has already been accounted for, so we can just push the index into indices
									indices.push(*i);
								}
								None => {
									//This vertex is not accounted for, and so now we must add its data to vertices

									//We add the position data to vertices
									vertices.push(object.vertices[pair.0 - index_offset].x as f32);
									vertices.push(object.vertices[pair.0 - index_offset].y as f32);
									vertices.push(object.vertices[pair.0 - index_offset].z as f32);

									//Push the normal vector data if there is any
									match pair.1 {
										Some(i) => {
											let coords = [object.normals[i].x as f32, object.normals[i].y as f32, object.normals[i].z as f32];
											for c in &coords {
												vertices.push(*c);
											}
										}
										None => {
											for _ in 0..3 {
												vertices.push(0.0);
											}
										}
									}

									//Push the texture coordinate data if there is any
									match pair.2 {
										Some(i) => {
											vertices.push(object.tex_vertices[i].u as f32);
											vertices.push(object.tex_vertices[i].v as f32);
										}
										None => {
											vertices.push(0.0);
											vertices.push(0.0);
										}
									}

									//Add the index to indices
									indices.push(current_index);

									//Then we declare that this vertex will appear in vertices at current_index
									index_map.insert(pair, current_index);
									current_index += 1;
								}
							}
						}
					}
					_ => {
						println!("Only triangle meshes are supported.");
						return None;
					}
				}
			}
		}
		index_offset += current_index as usize;
	}
	geometry_boundaries.push(indices.len() as GLsizei);
	Some((vertices, indices, geometry_boundaries, materials_in_order))
}

fn uniform_scale(scale: f32) -> glm::TMat4<f32> {
	glm::scaling(&glm::vec3(scale, scale, scale))
}

fn update_openvr_mesh(meshes: &mut OptionVec<Mesh>, poses: &[TrackedDevicePose], tracking_to_world: &glm::TMat4<f32>, device_index: usize, mesh_index: Option<usize>) {
	let device_to_absolute = openvr_to_mat4(*poses[device_index].device_to_absolute_tracking());
	if let Some(mesh) = meshes.get_element(mesh_index) {
		mesh.model_matrix = tracking_to_world * device_to_absolute;
	}
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

	//Calculate render target size
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
	let skybox_shader = unsafe { compile_program_from_files("shaders/skybox_vertex.glsl", "shaders/skybox_fragment.glsl") };

	//Setup the VR rendering target
	let vr_render_target = unsafe { create_vr_render_target(&render_target_size) };
	let openvr_texture_handle = Texture {
		handle: Handle::OpenGLTexture(vr_render_target as usize),
		color_space: ColorSpace::Auto
	};

	//Create channels for communication with the worker thread
	let (order_tx, order_rx) = mpsc::channel::<WorkOrder>();
	let (result_tx, result_rx) = mpsc::channel::<WorkResult>();

	//Map of texture paths to texture ids
	let mut textures: HashMap<String, GLuint> = HashMap::new();

	//Spawn thread to do work
	let worker_handle = thread::spawn(move || {
		loop {
			match order_rx.recv() {
				Ok(WorkOrder::Image(path)) => {
					if let Err(e) = result_tx.send(WorkResult::Image(path, image_data_from_path(path))) {
						println!("{}", e);
					}
				}
				Ok(WorkOrder::Model) => {
					//Invoke file selection dialogue
					let path = match nfd::open_file_dialog(None, None).unwrap() {
						Response::Okay(filename) => { Some(filename) }
						_ => { None }
					};

					//Send model data back to the main thread
					if let Some(p) = path {
						if let Err(e) = result_tx.send(WorkResult::Model(load_wavefront_obj(&p))) {
							println!("{}", e);
						}
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
		let seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() * 1000;
		println!("Seed used for terrain generation: {}", seed as u32);
		OpenSimplex::new().set_seed(seed as u32)
	};

	//Create the large tessellated surface
	const SIMPLEX_SCALE: f64 = 3.0;
	const TERRAIN_SCALE: f32 = 500.0;
	const TERRAIN_AMPLITUDE: f32 = TERRAIN_SCALE / 10.0;
	let surface_normals;
	unsafe {
		const ELEMENT_STRIDE: usize = 8;
		const WIDTH: usize = 100; //Width (and height) in vertices
		const TRIS: usize = (WIDTH - 1) * (WIDTH - 1) * 2;

		//Buffers to be filled
		let mut vertices = vec![0.0; WIDTH * WIDTH * ELEMENT_STRIDE];
		let mut indices: Vec<u16> = Vec::with_capacity(TRIS * 3);

		//Calculate the positions and tex coords for each vertex
		for i in (0..vertices.len()).step_by(ELEMENT_STRIDE) {
			let xpos: usize = (i / ELEMENT_STRIDE) % WIDTH;
			let ypos: usize = (i / ELEMENT_STRIDE) / WIDTH;

			//Calculate vertex position
			vertices[i] = (xpos as f32 / (WIDTH - 1) as f32) as f32 - 0.5;
			vertices[i + 2] = (ypos as f32 / (WIDTH - 1) as f32) as f32 - 0.5;

			vertices[i + 1] = simplex_generator.get([vertices[i] as f64 * SIMPLEX_SCALE, vertices[i + 2] as f64 * SIMPLEX_SCALE]) as f32;

			//Calculate texture coordinates
			vertices[i + 6] = TERRAIN_SCALE * (xpos as f32 / (WIDTH - 1) as f32) as f32;
			vertices[i + 7] = TERRAIN_SCALE * (ypos as f32 / (WIDTH - 1) as f32) as f32;
		}

		//This loop executes once per subsquare on the plane, and pushes the indices of the two triangles that comprise said subsquare into the indices Vec
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

		//The ith vertex will be shared by each surface in vertex_surface_map[i]
		let mut vertex_surface_map = Vec::with_capacity(vertices.len() / ELEMENT_STRIDE);
		for _ in 0..(vertices.len() / ELEMENT_STRIDE) {
			vertex_surface_map.push(Vec::new());
		}

		//Calculate surface normals
		surface_normals = {
			const INDICES_PER_TRIANGLE: usize = 3;
			let mut norms = Vec::with_capacity(indices.len() / INDICES_PER_TRIANGLE);

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

				//The cross product of two edges of a surface must be normal to that surface
				norms.push(glm::cross::<f32, glm::U3>(&u, &v));
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
			vertices[i + 3] = averaged_vector.data[0];
			vertices[i + 4] = averaged_vector.data[1];
			vertices[i + 5] = averaged_vector.data[2];
		}

		println!("The generated surface contains {} vertices", vertices.len() / ELEMENT_STRIDE);
		let vao = create_vertex_array_object(&vertices, &indices, &[3, 3, 2]);
		let model_matrix = glm::scaling(&glm::vec3(TERRAIN_SCALE, TERRAIN_AMPLITUDE, TERRAIN_SCALE));
		order_tx.send(WorkOrder::Image("textures/grass.jpg")).unwrap();
		meshes.insert(Mesh::new(vao, model_matrix, "textures/grass.jpg", vec![0, indices.len() as GLsizei], None))
	};

	//Variables to keep track of the loaded models
	let model_bounding_sphere_radius = 0.20;
	let mut bound_controller_indices = Vec::new();
	let mut model_indices = Vec::new();
	let mut model_to_controller_matrices = Vec::new();

	//Initialize the struct of arrays containing controller related state
	let mut controllers = Controllers::new();

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
			let source = rodio::Decoder::new(BufReader::new(File::open("audio/woodlands.mp3").unwrap())).unwrap();
			sink.append(source);
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

	let light_direction = glm::normalize(&glm::vec4(1.0, 1.0, 0.0, 0.0));

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
				WorkResult::Image(path, tex_data) => {
					let tex_id = unsafe { load_texture_from_data(tex_data) };
					textures.insert(path.to_string(), tex_id);
				}
				WorkResult::Model(option_mesh) => {
					if let Some(package) = option_mesh {
						let vao = unsafe { create_vertex_array_object(&package.0, &package.1, &[3, 3, 2]) };
						let mesh = Mesh::new(vao, glm::translation(&glm::vec3(0.0, 0.8, 0.0)) * uniform_scale(0.3), "textures/bricks.jpg", package.2, Some(package.3));
						model_indices.push(Some(meshes.insert(mesh)));
						bound_controller_indices.push(None);
						model_to_controller_matrices.push(glm::identity());
					}
				}
			}
		}

		//Connect all meshes to their textures
		//TODO: Change so that when a texture is loaded, all meshes are checked, and when a mesh is created, there's a check for the texture it needs
		//instead of this check that just happens every frame indiscriminantly
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
						Key::O => { camera.fov_delta = -Camera::FOV_SPEED; }
						Key::P => { camera.fov_delta = Camera::FOV_SPEED; }
						Key::I => { camera.fov = 90.0; }
						Key::G => { is_lighting = !is_lighting; }
						Key::LeftShift => { camera.velocity *= 15.0; }
						Key::L => {							
							if let Err(e) = order_tx.send(WorkOrder::Model) {
								println!("{}", e);
							}							
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

				//Handle right hand controls
				if i == 1 {
					//Check if the user toggled flying controls
					if controllers.released_this_frame(i, button_id::APPLICATION_MENU) {
						is_flying = !is_flying;
					}
				}
				
				//tracking_to_world = glm::translation(&glm::vec4_to_vec3(&movement_vector)) * tracking_to_world;
				tracking_position += movement_vector;

				if !is_flying {
					tracking_position.y = TERRAIN_AMPLITUDE * simplex_generator.get([tracking_position.x as f64 * SIMPLEX_SCALE / TERRAIN_SCALE as f64,
																			 		 tracking_position.z as f64 * SIMPLEX_SCALE / TERRAIN_SCALE as f64]) as f32;
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

		//Set up render data
		let framebuffers = [vr_render_target, vr_render_target, 0];
		let eyes = [Some(Eye::Left), Some(Eye::Right), None];
		let sizes = [render_target_size, render_target_size, window_size];

		//Rendering code
		let render_context = RenderContext::new(&p_matrices, &v_matrices, light_direction, is_lighting);
		unsafe {
			gl::Enable(gl::DEPTH_TEST);	//Enable depth testing
			gl::DepthFunc(gl::LESS);	//Pass the fragment with the smallest z-value

			//Set clear color
			gl::ClearColor(0.53, 0.81, 0.92, 1.0);

			//Render once per framebuffer (Left eye, Right eye, Companion window)
			for i in 0..framebuffers.len() {
				//Set up render target
				gl::BindFramebuffer(gl::FRAMEBUFFER, framebuffers[i]);
				gl::Viewport(0, 0, sizes[i].0 as GLsizei, sizes[i].1 as GLsizei);

				//Clear the framebuffer's color buffer and depth buffer
				gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

				//Compute the view-projection matrix for the skybox (the conversion functions are just there to nullify the translation component of the view matrix)
				//The skybox vertices should obviously be rotated along with the camera, but shouldn't be translated in order to maintain the illusion
				//that the sky is infinitely far away
				let view_projection = p_matrices[i] * glm::mat3_to_mat4(&glm::mat4_to_mat3(&v_matrices[i]));

				//Render the skybox
				gl::UseProgram(skybox_shader);
				gl::UniformMatrix4fv(get_uniform_location(skybox_shader, "view_projection"), 1, gl::FALSE, &flatten_glm(&(view_projection * uniform_scale(400.0))) as *const GLfloat);
				gl::BindTexture(gl::TEXTURE_CUBE_MAP, skybox_cubemap);
				gl::BindVertexArray(skybox_vao);
				gl::DrawElements(gl::TRIANGLES, 36, gl::UNSIGNED_SHORT, ptr::null());

				//Render the regular meshes
				gl::Enable(gl::CULL_FACE);
				render_meshes(&meshes, model_shader, i, &render_context);

				//Submit render to HMD
				submit_to_hmd(eyes[i], &openvr_compositor, &openvr_texture_handle);
			}
		}

		window.render_context().swap_buffers();
		glfw.poll_events();
	}

	//Shut down the worker thread
	order_tx.send(WorkOrder::Quit).unwrap();
	worker_handle.join().unwrap();

	//Shut down OpenVR
	if let Some(ctxt) = openvr_context {
		unsafe { ctxt.shutdown(); }
	}
}
