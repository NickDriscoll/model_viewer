use gl::types::*;
use std::ffi::CString;
use std::str;
use std::io::Read;
use std::fs::File;
use std::ptr;
use std::mem;
use std::path::Path;
use std::os::raw::c_void;
use image::GenericImageView;
use openvr::Eye;
use openvr::Compositor;
use crate::structs::*;
use crate::flatten_glm;

pub type ImageData = (Vec<u8>, u32, u32);
const INFO_LOG_SIZE: usize = 512;

pub unsafe fn compile_shader(shadertype: GLenum, source: &str) -> GLuint {
	let shader = gl::CreateShader(shadertype);
	let cstr_vert = CString::new(source.as_bytes()).unwrap();
	gl::ShaderSource(shader, 1, &cstr_vert.as_ptr(), ptr::null());
	gl::CompileShader(shader);

	//Check for errors
	let mut success = gl::FALSE as GLint;
	let mut infolog = Vec::with_capacity(INFO_LOG_SIZE);

	gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);
	if success != gl::TRUE as GLint {
		gl::GetShaderInfoLog(shader, INFO_LOG_SIZE as i32, ptr::null_mut(), infolog.as_mut_ptr() as *mut GLchar);
		shader_compilation_error(infolog);
	}
	shader
}

pub unsafe fn compile_shader_from_file(shadertype: GLenum, path: &str) -> GLuint {
	let mut source = String::new();
	File::open(path).unwrap().read_to_string(&mut source).unwrap();
	compile_shader(shadertype, &source)
}

pub unsafe fn compile_program_from_files(vertex_path: &str, fragment_path: &str) -> GLuint {
	let vertexshader = compile_shader_from_file(gl::VERTEX_SHADER, vertex_path);
	let fragmentshader = compile_shader_from_file(gl::FRAGMENT_SHADER, fragment_path);

	let mut success = gl::FALSE as GLint;
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
		shader_compilation_error(infolog);
	}

	gl::DeleteShader(vertexshader);
	gl::DeleteShader(fragmentshader);
	shader_progam
}

pub fn shader_compilation_error(infolog: Vec<u8>) {
	let error_message = match str::from_utf8(&infolog) {
		Ok(message) => { message }
		Err(e) => {
			let sized_log = &infolog[0..e.valid_up_to()];
			str::from_utf8(sized_log).unwrap()
		}
	};
	panic!("\n--------SHADER COMPILATION ERROR--------\n{}", error_message);
}

pub unsafe fn get_uniform_location(program: GLuint, name: &str) -> GLint {
	let cstring = CString::new(name.as_bytes()).unwrap();
	gl::GetUniformLocation(program, cstring.as_ptr())
}

pub unsafe fn render_mesh(mesh: &Mesh, p_matrix: &glm::TMat4<f32>, v_matrix: &glm::TMat4<f32>, mvp_location: GLint) {
	gl::UseProgram(mesh.program);

	//Send the model-view-projection matrix to the GPU
	let mvp = p_matrix * v_matrix * mesh.model_matrix;
	gl::UniformMatrix4fv(mvp_location, 1, gl::FALSE, &flatten_glm(&mvp) as *const GLfloat);

	let tex = match mesh.texture {
		Some(t) => {
			t
		}
		None => {
			//Color every fragment black if there's no texture
			0
		}
	};
	gl::BindTexture(gl::TEXTURE_2D, tex);

	gl::BindVertexArray(mesh.vao);
	gl::DrawElements(gl::TRIANGLES, mesh.indices_count, gl::UNSIGNED_SHORT, ptr::null());
}

pub unsafe fn render_scene(meshes: &mut OptionVec<Mesh>, p_matrix: glm::TMat4<f32>, v_matrix: glm::TMat4<f32>, mvp_location: GLint) {
	gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

	for option_mesh in meshes.iter() {
		if let Some(mesh) = option_mesh {
			render_mesh(&mesh, &p_matrix, &v_matrix, mvp_location);
		}
	}
}

pub unsafe fn submit_to_hmd(eye: Eye, openvr_compositor: &Option<Compositor>, target_handle: &openvr::compositor::texture::Texture) {
	if let Some(ref comp) = openvr_compositor {
		comp.submit(eye, target_handle, None, None).unwrap();
	}
}

pub unsafe fn create_vertex_array_object(vertices: &[f32], indices: &[u16]) -> GLuint {
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
	

	const ATTRIBUTE_STRIDE: i32 = 8;
	let byte_stride = ATTRIBUTE_STRIDE * mem::size_of::<GLfloat>() as i32;

	//Configure and enable the vertex attributes	
	gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, byte_stride, 0 as *const c_void);
	gl::EnableVertexAttribArray(0);

	gl::VertexAttribPointer(1, 3, gl::FLOAT, gl::FALSE, byte_stride, (3 * mem::size_of::<GLfloat>()) as *const c_void);
	gl::EnableVertexAttribArray(1);
	
	gl::VertexAttribPointer(2, 2, gl::FLOAT, gl::FALSE, byte_stride, (6 * mem::size_of::<GLfloat>()) as *const c_void);
	gl::EnableVertexAttribArray(2);

	vao
}

pub unsafe fn load_texture(path: &str) -> GLuint {
	load_texture_from_data(image_data_from_path(path))
}

pub fn image_data_from_path(path: &str) -> ImageData {
	let image = match image::open(&Path::new(path)) {
		Ok(im) => {
			im
		}
		Err(_) => {
			panic!("Unable to open {}", path);
		}
	};
	(image.raw_pixels(), image.width(), image.height())
}

pub unsafe fn load_texture_from_data(image_data: ImageData) -> GLuint {
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
				   image_data.1 as i32,
				   image_data.2 as i32,
				   0,
				   gl::RGB,
				   gl::UNSIGNED_BYTE,
				   &image_data.0[0] as *const u8 as *const c_void);
	gl::GenerateMipmap(gl::TEXTURE_2D);
	tex
}