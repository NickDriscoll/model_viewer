extern crate gl;
extern crate nalgebra_glm as glm;
use glfw::Action;
use glfw::Context;
use glfw::Key;
use glfw::WindowMode;
use glfw::WindowEvent;
use self::gl::types::*;
use openvr::ApplicationType;
use openvr::Eye;
use openvr::System;
use openvr::Compositor;
use openvr::compositor::texture::Handle;
use openvr::compositor::texture::ColorSpace;
use openvr::compositor::texture::Texture;
use openvr::RenderModels;
use openvr::TrackedControllerRole;
use openvr::TrackedDevicePose;
use image::GenericImageView;
use std::ffi::CString;
use std::fs::File;
use std::io::Read;
use std::str;
use std::path::Path;
use std::ptr;
use std::mem;
use std::string::String;
use std::os::raw::c_void;
use crate::structs::Mesh;
use crate::structs::GLProgram;

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

unsafe fn compile_shader(shadertype: GLenum, source: &str) -> GLuint {
	let shader = gl::CreateShader(shadertype);
	let cstr_vert = CString::new(source.as_bytes()).unwrap();
	gl::ShaderSource(shader, 1, &cstr_vert.as_ptr(), ptr::null());
	gl::CompileShader(shader);

	//Check for errors
	let mut success = gl::FALSE as GLint;
	const INFO_LOG_SIZE: usize = 512;
	let mut infolog = Vec::with_capacity(INFO_LOG_SIZE);
	infolog.set_len(INFO_LOG_SIZE - 1);
	gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);
	if success != gl::TRUE as GLint {
		gl::GetShaderInfoLog(shader, INFO_LOG_SIZE as i32, ptr::null_mut(), infolog.as_mut_ptr() as *mut GLchar);
		panic!("\n{}\n", str::from_utf8(&infolog).unwrap());
	}
	shader
}

unsafe fn compile_shader_from_file(shadertype: GLenum, path: &str) -> GLuint {
	let mut source = String::new();
	File::open(path).unwrap().read_to_string(&mut source).unwrap();
	compile_shader(shadertype, &source)
}

unsafe fn compile_program_from_files(vertex_path: &str, fragment_path: &str) -> GLuint {
	let vertexshader = compile_shader_from_file(gl::VERTEX_SHADER, vertex_path);
	let fragmentshader = compile_shader_from_file(gl::FRAGMENT_SHADER, fragment_path);

	let mut success = gl::FALSE as GLint;
	const INFO_LOG_SIZE: usize = 512;
	let mut infolog = Vec::with_capacity(INFO_LOG_SIZE);

	//Link shaders
	let shader_progam = gl::CreateProgram();
	gl::AttachShader(shader_progam, vertexshader);
	gl::AttachShader(shader_progam, fragmentshader);
	gl::LinkProgram(shader_progam);

	//Check for errors
	gl::GetProgramiv(shader_progam, gl::LINK_STATUS, &mut success);
	if success != gl::TRUE as GLint {
		gl::GetProgramInfoLog(shader_progam, INFO_LOG_SIZE as i32, ptr::null_mut(), infolog.as_mut_ptr() as *mut GLchar);
		panic!("\n--------SHADER COMPILATION ERROR--------\n{}", str::from_utf8(&infolog).unwrap());
	}

	gl::DeleteShader(vertexshader);
	gl::DeleteShader(fragmentshader);
	shader_progam
}

unsafe fn bind_program_and_uniforms(program: &GLProgram, matrix_values: &[glm::TMat4<f32>], vector_values: &[glm::TVec3<f32>]) {
	//For now, assert that locations equal values
	if program.matrix_locations.len() != matrix_values.len() ||
		program.vector_locations.len() != vector_values.len() {
			panic!("ERROR!\nmatrix_locations: {}\nmatrix_values: {}\nvector_locations: {}\nvector_values: {}",
				program.matrix_locations.len(),
				matrix_values.len(),
				program.vector_locations.len(),
				vector_values.len());
	}

	gl::UseProgram(program.name);

	for i in 0..program.matrix_locations.len() {
		gl::UniformMatrix4fv(program.matrix_locations[i], 1, gl::FALSE, &flatten_glm(&matrix_values[i]) as *const GLfloat);
	}

	for i in 0..program.vector_locations.len() {
		let v = [vector_values[i].x, vector_values[i].y, vector_values[i].z];
		gl::Uniform3fv(program.vector_locations[i], 1, &v as *const GLfloat);
	}
}

unsafe fn render_mesh(mesh: &Mesh) {
	bind_program_and_uniforms(&mesh.program, &mesh.matrix_values, &mesh.vector_values);

	if let Some(tex) = mesh.texture {
		gl::BindTexture(gl::TEXTURE_2D, tex);
	}

	gl::BindVertexArray(mesh.vao);
	gl::DrawElements(gl::TRIANGLES, mesh.indices_count, gl::UNSIGNED_SHORT, ptr::null());
}

unsafe fn render_scene(meshes: &mut [Mesh], p_matrix: glm::TMat4<f32>, v_matrix: glm::TMat4<f32>) {
	gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

	for mesh in meshes.iter_mut() {
		mesh.matrix_values[0] = p_matrix * v_matrix * mesh.model_matrix;
	}

	for mesh in meshes.iter() {
		render_mesh(&mesh);
	}
}

unsafe fn submit_to_hmd(eye: Eye, openvr_compositor: &Option<Compositor>, target_handle: &Texture) {
	if let Some(ref comp) = openvr_compositor {
		comp.submit(eye, target_handle, None, None).unwrap();
	}
}

unsafe fn create_vertex_array_object(vertices: &[f32], indices: &[u16], vertex_stride: i32, attribute_sizes: &[usize]) -> GLuint {
	let (mut vbo, mut vao, mut ebo) = (0, 0, 0);
	gl::GenVertexArrays(1, &mut vao);
	gl::GenBuffers(1, &mut vbo);
	gl::GenBuffers(1, &mut ebo);

	gl::BindVertexArray(vao);

	gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
	gl::BufferData(gl::ARRAY_BUFFER,
				   (vertices.len() * mem::size_of::<GLfloat>()) as GLsizeiptr,
				   &vertices[0] as *const f32 as *const c_void,
				   gl::STATIC_DRAW);

	gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
	gl::BufferData(gl::ELEMENT_ARRAY_BUFFER,
				   (indices.len() * mem::size_of::<GLushort>()) as GLsizeiptr,
				   &indices[0] as *const u16 as *const c_void,
				   gl::STATIC_DRAW);

	let byte_stride = vertex_stride * mem::size_of::<GLfloat>() as GLsizei;

	//Configure and enable the vertex attributes
	let mut accumulated_size = 0;
	for i in 0..attribute_sizes.len() {
		gl::VertexAttribPointer(i as u32, attribute_sizes[i] as i32, gl::FLOAT, gl::FALSE, byte_stride, (accumulated_size * mem::size_of::<GLfloat>()) as *const c_void);
		gl::EnableVertexAttribArray(i as u32);
		accumulated_size += attribute_sizes[i];
	}

	vao
}

unsafe fn load_texture(path: &str) -> GLuint {
	let image = match image::open(&Path::new(path)) {
		Ok(im) => {
			im
		}
		Err(_) => {
			panic!("Unable to open {}", path);
		}
	};
	let data = image.raw_pixels();

	let mut tex = 0;
	gl::GenTextures(1, &mut tex);
	gl::BindTexture(gl::TEXTURE_2D, tex);
	gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::REPEAT as i32);
	gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::REPEAT as i32);
	gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
	gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);

	gl::TexImage2D(gl::TEXTURE_2D,
				   0,
				   gl::RGB as i32,
				   image.width() as i32,
				   image.height() as i32,
				   0,
				   gl::RGB,
				   gl::UNSIGNED_BYTE,
				   &data[0] as *const u8 as *const c_void);
	gl::GenerateMipmap(gl::TEXTURE_2D);
	tex
}

fn attach_mesh_to_controller(meshes: &mut [Mesh], poses: &[TrackedDevicePose], controller_index: &Option<u32>, mesh_index: Option<usize>) {
	if let Some(index) = controller_index {
		let controller_model_matrix = openvr_to_mat4(*poses[*index as usize].device_to_absolute_tracking());
		if let Some(i) = mesh_index {
			meshes[i].model_matrix = controller_model_matrix;
		}
	}
}

fn load_controller_meshes<'a>(openvr_system: &Option<System>, openvr_rendermodels: &Option<RenderModels>, meshes: &mut Vec<Mesh<'a>>, index: u32, program: &'a GLProgram) -> (Option<usize>, Option<usize>) {
	let mut result = (None, None);
	if let (Some(ref sys), Some(ref ren_mod)) = (&openvr_system, &openvr_rendermodels) {
		let name = sys.string_tracked_device_property(index, openvr::property::RenderModelName_String).unwrap();
		if let Some(model) = ren_mod.load_render_model(&name).unwrap() {
			//Flatten each vertex into a simple &[f32]
			const ELEMENT_STRIDE: usize = 6;
			let mut vertices = Vec::with_capacity(ELEMENT_STRIDE * model.vertices().len());
			for vertex in model.vertices() {
				vertices.push(vertex.position[0]);
				vertices.push(vertex.position[1]);
				vertices.push(vertex.position[2]);
				vertices.push(0.0);
				vertices.push(0.0);
				vertices.push(0.0);
			}

			//Create vao
			let vao = unsafe { create_vertex_array_object(&vertices, model.indices(), ELEMENT_STRIDE as i32, &[3, 3]) };

			let mesh = Mesh::new(vao, glm::translation(&glm::vec3(0.0, -1.0, 0.0)), program, None, model.indices().len() as i32);
			meshes.push(mesh);
			let left_index = Some(meshes.len() - 1);

			let mesh = Mesh::new(vao, glm::translation(&glm::vec3(0.0, -1.0, 0.0)), program, None, model.indices().len() as i32);
			meshes.push(mesh);
			let right_index = Some(meshes.len() - 1);

			println!("Loaded controller mesh");

			result = (left_index, right_index);
		}
	}
	result
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

	//Compile shader programs
	let texture_program = unsafe { GLProgram::from_files("shaders/vertex_texture.glsl", "shaders/fragment_texture.glsl", &["mvp"], &[]) };
	let color_program = unsafe { GLProgram::from_files("shaders/vertex_color.glsl", "shaders/fragment_color.glsl", &["mvp", "model_matrix"], &["light_pos"]) };

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
		let handle = Handle::OpenGLTexture(vr_render_target as usize);
		Texture {
			handle,
			color_space: ColorSpace::Auto
		}
	};

	unsafe {
		gl::Enable(gl::DEPTH_TEST);
		gl::DepthFunc(gl::LESS);
		gl::Enable(gl::CULL_FACE);
	}

	//Vec of meshes
	let mut meshes = Vec::with_capacity(3);

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
		let vao = create_vertex_array_object(&vertices, &indices, 5, &[3, 2]);
		let mesh = Mesh::new(vao, glm::scaling(&glm::vec3(5.0, 5.0, 5.0)),
							 &texture_program, Some(load_texture("textures/checkerboard.jpg")), indices.len() as i32);
		meshes.push(mesh);
		meshes.len() - 1
	};

	//Create the cube
	let cube_mesh_index = unsafe {
		let vertices = [
			//Position data 				//Color values
			-0.5f32, -0.5, 0.5,				0.0, 0.0, 0.0,
			-0.5, 0.5, 0.5,					1.0, 0.0, 0.0,
			0.5, 0.5, 0.5,					1.0, 1.0, 0.0,
			0.5, -0.5, 0.5,					0.0, 1.0, 0.0,
			-0.5, -0.5, -0.5,				0.0, 0.0, 1.0,
			-0.5, 0.5, -0.5,				1.0, 0.0, 1.0,
			0.5, 0.5, -0.5,					1.0, 1.0, 1.0,
			0.5, -0.5, -0.5,				0.0, 1.0, 1.0
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
		let vao = create_vertex_array_object(&vertices, &indices, 6, &[3, 3]);
		let mesh = Mesh::new(vao, glm::identity(), &color_program, None, indices.len() as i32);
		meshes.push(mesh);
		meshes.len() - 1
	};

	let light_pos = glm::vec3::<f32>(1.0, 1.0, 1.0);

	//Controller related variables
	let mut controller_indices = match openvr_system {
		Some(ref sys) => {
			let left_index = sys.tracked_device_index_for_controller_role(TrackedControllerRole::LeftHand);
			let right_index = sys.tracked_device_index_for_controller_role(TrackedControllerRole::RightHand);

			(left_index, right_index)
		}
		None => {
			(None, None)
		}
	};
	let mut controller_mesh_indices = (None, None);

	//Gameplay state
	let mut ticks = 0.0;
	let mut camera_position = glm::vec3(0.0, -1.0, -1.0);
	let mut camera_velocity = glm::vec3(0.0, 0.0, 0.0);
	let mut camera_fov = 65.0;
	let mut camera_fov_delta = 0.0;

	//Main loop
	while !window.should_close() {
		//Find controllers if we haven't already
		if let Some(ref sys) = openvr_system {
			if let (any, None)  = controller_indices {
				controller_indices = (any, sys.tracked_device_index_for_controller_role(TrackedControllerRole::RightHand));
			}
			if let (None, any) = controller_indices {
				controller_indices = (sys.tracked_device_index_for_controller_role(TrackedControllerRole::LeftHand), any);
			}
		}

		//Load controller model if we haven't already
		if let (None, None) = controller_mesh_indices {
			controller_mesh_indices = {
				match controller_indices {
					(Some(index), _) |
					(_, Some(index)) => {
						load_controller_meshes(&openvr_system,
											   &openvr_rendermodels,
											   &mut meshes,
											   index,
											   &color_program)
					}
					_ => {
						(None, None)
					}
				}
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
							camera_velocity.z = 0.1;
						}
						Key::S => {
							camera_velocity.z = -0.1;
						}
						Key::A => {
							camera_velocity.x = 0.1;
						}
						Key::D => {
							camera_velocity.x = -0.1;
						}
						Key::O => {
							camera_fov_delta = -1.0;
						}
						Key::P => {
							camera_fov_delta = 1.0;
						}
						Key::Escape => {
							window.set_should_close(true);
						}
						_ => {}
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

		//Get VR pose data
		let render_poses = match openvr_compositor {
			Some(ref comp) => {
				Some(comp.wait_get_poses().unwrap().render)
			}
			None => {
				None
			}
		};

		//Attach a mesh to the controllers
		if let Some(poses) = render_poses {
			let (ref left, ref right) = controller_indices;
			attach_mesh_to_controller(&mut meshes, &poses, left, controller_mesh_indices.0);
			attach_mesh_to_controller(&mut meshes, &poses, right, controller_mesh_indices.1);
		}

		//Get view matrices for each eye
		let v_matrices = match openvr_system {
			Some(ref sys) => {
				if let Some(poses) = render_poses {
					let hmd_to_absolute = openvr_to_mat4(*poses[0].device_to_absolute_tracking());
					let left_eye_to_hmd = openvr_to_mat4(sys.eye_to_head_transform(Eye::Left));
					let right_eye_to_hmd = openvr_to_mat4(sys.eye_to_head_transform(Eye::Right));

					//Need to return inverse(hmd_to_absolute * eye_to_hmd)
					(glm::affine_inverse(hmd_to_absolute * left_eye_to_hmd),
					 glm::affine_inverse(hmd_to_absolute * right_eye_to_hmd),
					 glm::affine_inverse(hmd_to_absolute))
				} else {
					//Create a matrix that gets a decent view of the scene
					let view_matrix = glm::translation(&camera_position);
					(glm::identity(), glm::identity(), view_matrix)
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

		meshes[cube_mesh_index].model_matrix = glm::translation(&glm::vec3(0.0, 1.0, 0.0)) *
								   glm::rotation(ticks*0.5, &glm::vec3(1.0, 0.0, 0.0)) *
								   glm::rotation(ticks*0.5, &glm::vec3(0.0, 1.0, 0.0)) *
								   glm::scaling(&glm::vec3(0.25, 0.25, 0.25));
		meshes[cube_mesh_index].matrix_values[1] = meshes[cube_mesh_index].model_matrix;
		meshes[cube_mesh_index].vector_values[0] = light_pos;

		camera_position += camera_velocity;
		camera_fov += camera_fov_delta;

		//Rendering code
		unsafe {
			//Set up to render on texture
			gl::BindFramebuffer(gl::FRAMEBUFFER, vr_render_target);
			gl::Viewport(0, 0, render_target_size.0 as GLsizei, render_target_size.1 as GLsizei);

			//Set clear color
			gl::ClearColor(0.53, 0.81, 0.92, 1.0);

			//Render left eye
			render_scene(&mut meshes, p_matrices.0, v_matrices.0);

			//Send to HMD
			submit_to_hmd(Eye::Left, &openvr_compositor, &openvr_texture_handle);

			//Render right eye
			render_scene(&mut meshes, p_matrices.1, v_matrices.1);

			//Send to HMD
			submit_to_hmd(Eye::Right, &openvr_compositor, &openvr_texture_handle);

			//Unbind the vr render target so that we can draw to the window
			gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
			gl::Viewport(0, 0, window_size.0 as GLsizei, window_size.1 as GLsizei);

			render_scene(&mut meshes, p_matrices.2, v_matrices.2);
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
