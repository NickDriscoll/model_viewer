use gl::types::*;
use std::ffi::CString;
use std::str;
use std::io::Read;
use std::fs::File;
use std::{mem, process, ptr};
use std::path::Path;
use std::os::raw::c_void;
use image::GenericImageView;
use openvr::{Compositor, Eye};

//pub type ImageData = (Vec<u8>, u32, u32);
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
	infolog.set_len(INFO_LOG_SIZE - 1);
	gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);
	if success != gl::TRUE as GLint {
		gl::GetShaderInfoLog(shader, INFO_LOG_SIZE as i32, ptr::null_mut(), infolog.as_mut_ptr() as *mut GLchar);
		shader_compilation_error(&infolog);
	}
	shader
}

pub unsafe fn compile_shader_from_file(shadertype: GLenum, path: &str) -> GLuint {
	let mut source = String::new();

	match File::open(path) {
		Ok(mut file) => {
			file.read_to_string(&mut source).unwrap();
		}
		Err(e) => {
			println!("{}\npath: \"{}\"", e, path);
			process::exit(-1);
		}
	}
	compile_shader(shadertype, &source)
}

pub unsafe fn compile_program_from_files(vertex_path: &str, fragment_path: &str) -> GLuint {
	let vertexshader = compile_shader_from_file(gl::VERTEX_SHADER, vertex_path);
	let fragmentshader = compile_shader_from_file(gl::FRAGMENT_SHADER, fragment_path);

	//Link shaders
	let shader_progam = gl::CreateProgram();
	gl::AttachShader(shader_progam, vertexshader);
	gl::AttachShader(shader_progam, fragmentshader);
	gl::LinkProgram(shader_progam);

	//Check for errors
	let mut success = gl::FALSE as GLint;
	let mut infolog = Vec::with_capacity(INFO_LOG_SIZE);
	gl::GetProgramiv(shader_progam, gl::LINK_STATUS, &mut success);
	if success != gl::TRUE as GLint {
		gl::GetProgramInfoLog(shader_progam, INFO_LOG_SIZE as i32, ptr::null_mut(), infolog.as_mut_ptr() as *mut GLchar);
		shader_compilation_error(&infolog);
	}

	gl::DeleteShader(vertexshader);
	gl::DeleteShader(fragmentshader);
	shader_progam
}

pub fn shader_compilation_error(infolog: &[u8]) {
	let error_message = match str::from_utf8(infolog) {
		Ok(message) => { message }
		Err(e) => {
			let sized_log = &infolog[0..e.valid_up_to()];
			str::from_utf8(&sized_log).unwrap()
		}
	};
	panic!("\n--------SHADER COMPILATION ERROR--------\n{}", error_message);
}

pub unsafe fn get_uniform_location(program: GLuint, name: &str) -> GLint {
	let cstring = CString::new(name.as_bytes()).unwrap();
	gl::GetUniformLocation(program, cstring.as_ptr())
}

pub unsafe fn submit_to_hmd(eye: Option<Eye>, openvr_compositor: &Option<Compositor>, target_handle: &openvr::compositor::texture::Texture) {
	if let (Some(ref comp), Some(e)) = (openvr_compositor, eye) {
		if let Err(err) = comp.submit(e, target_handle, None, None) {
			println!("{}", err);
		}
	}
}

pub unsafe fn create_vertex_array_object(vertices: &[f32], indices: &[u16], attribute_strides: &[i32]) -> GLuint {
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

	let byte_stride = {
		let mut sum = 0;
		for stride in attribute_strides {
			sum += stride;
		}
		sum * mem::size_of::<GLfloat>() as i32
	};

	//Configure and enable the vertex attributes
	let mut cumulative_size = 0;
	for i in 0..attribute_strides.len() {
		gl::VertexAttribPointer(i as u32, attribute_strides[i], gl::FLOAT, gl::FALSE, byte_stride, (cumulative_size * mem::size_of::<GLfloat>() as u32) as *const c_void);
		gl::EnableVertexAttribArray(i as u32);
		cumulative_size += attribute_strides[i] as u32;
	}

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

pub unsafe fn create_vr_render_target(render_target_size: &(u32, u32)) -> GLuint {
	let mut render_target = 0;
	gl::GenFramebuffers(1, &mut render_target);
	gl::BindFramebuffer(gl::FRAMEBUFFER, render_target);

	//Create the texture that will be rendered to
	let mut vr_render_texture = 0;
	gl::GenTextures(1, &mut vr_render_texture);
	gl::BindTexture(gl::TEXTURE_2D, vr_render_texture);
	gl::TexImage2D(gl::TEXTURE_2D,
					   0,
					   gl::RGBA as i32,
					   render_target_size.0 as GLsizei,
					   render_target_size.1 as GLsizei,
					   0,
					   gl::RGBA,
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
}