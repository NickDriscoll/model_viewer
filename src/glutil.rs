use gl::types::*;
use std::ffi::CString;
use std::str;
use std::io::Read;
use std::fs::File;
use std::{mem, process, ptr};
use std::os::raw::c_void;
use image::DynamicImage;
use openvr::{Compositor, Eye};
use crate::*;

const INFO_LOG_SIZE: usize = 2048;

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

const SHADER_PATH: &str = "shaders";
pub unsafe fn compile_program_from_files(vertex_name: &str, fragment_name: &str) -> GLuint {
	let vertexshader = compile_shader_from_file(gl::VERTEX_SHADER, &format!("{}/vertex/{}", SHADER_PATH, vertex_name));
	let fragmentshader = compile_shader_from_file(gl::FRAGMENT_SHADER, &format!("{}/fragment/{}", SHADER_PATH, fragment_name));

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

pub unsafe fn uniform_location(program: GLuint, name: &str) -> GLint {
	let cstring = CString::new(name.as_bytes()).unwrap();
	gl::GetUniformLocation(program, cstring.as_ptr())
}

pub unsafe fn gl_gen_buffer() -> GLuint {
	let mut buffer = 0;
	gl::GenBuffers(1, &mut buffer);
	buffer
}

pub unsafe fn submit_to_hmd(eye: Option<Eye>, openvr_compositor: &Option<Compositor>, target_handle: &openvr::compositor::texture::Texture) {
	if let (Some(ref comp), Some(e)) = (openvr_compositor, eye) {
		handle_result(comp.submit(e, target_handle, None, None));
	}
}

//Input: array of vertex data, an array of indices, and an array representing the number of elements per vertex attribute
//Output: A vertex array object with the vertex data bound as a GL_ARRAY_BUFFER, and the index data bound as a GL_ELEMENT_ARRAY_BUFFER
pub unsafe fn create_vertex_array_object(vertices: &[f32], indices: &[u16], attribute_strides: &[i32]) -> GLuint {
	let mut vao = 0;
	gl::GenVertexArrays(1, &mut vao);
	let vbo = gl_gen_buffer();
	let ebo = gl_gen_buffer();

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
		gl::VertexAttribPointer(i as GLuint,
								attribute_strides[i],
								gl::FLOAT,
								gl::FALSE,
								byte_stride,
								(cumulative_size * mem::size_of::<GLfloat>() as GLuint) as *const c_void);
		
		gl::EnableVertexAttribArray(i as GLuint);
		cumulative_size += attribute_strides[i] as u32;
	}

	vao
}

pub unsafe fn load_texture(path: &str, parameters: &[(GLenum, GLenum)]) -> GLuint {
	load_texture_from_data(image_data_from_path(path), parameters)
}

pub fn image_data_from_path(path: &str) -> ImageData {
	match image::open(path) {
		Ok(DynamicImage::ImageRgb8(im)) => {
			let width = im.width();
			let height = im.height();
			let raw = im.into_raw();
			ImageData {
				data: raw,
				width: width as GLint,
				height: height as GLint,
				format: gl::RGB,
				internal_format: gl::SRGB
			}
		}
		Ok(DynamicImage::ImageRgba8(im)) => {
			let width = im.width();
			let height = im.height();
			let raw = im.into_raw();
			ImageData {
				data: raw,
				width: width as GLint,
				height: height as GLint,
				format: gl::RGBA,
				internal_format: gl::SRGB8_ALPHA8
			}
		}
		Ok(_) => {
			panic!("{:?} is of unsupported image type", path);
		}
		Err(e) => {
			panic!("Unable to open {}: {}", path, e);
		}
	}
}

pub unsafe fn load_texture_from_data(image_data: ImageData, parameters: &[(GLenum, GLenum)]) -> GLuint {
	let mut tex = 0;
	gl::GenTextures(1, &mut tex);
	gl::BindTexture(gl::TEXTURE_2D, tex);
	for param in parameters {
		gl::TexParameteri(gl::TEXTURE_2D, param.0, param.1 as GLint);
	}

	gl::TexImage2D(gl::TEXTURE_2D,
				   0,
				   image_data.internal_format as i32,
				   image_data.width,
				   image_data.height,
				   0,
				   image_data.format,
				   gl::UNSIGNED_BYTE,
				   &image_data.data[0] as *const u8 as *const c_void);
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
					   gl::SRGB8_ALPHA8 as i32,
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
		println!("Framebuffer wasn't complete: VR render target");
	}
	render_target
}

pub unsafe fn bind_matrix4(program: GLuint, name: &str, matrix: &glm::TMat4<f32>) {
	gl::UniformMatrix4fv(uniform_location(program, name), 1, gl::FALSE, &flatten_glm(matrix) as *const GLfloat);
}

pub unsafe fn bind_uniforms(program: GLuint, mat_uniforms: &[&str], mats: &[&glm::TMat4<f32>], vec_uniforms: &[&str], vecs: &[&glm::TVec4<f32>], byte_uniforms: &[&str], bytes: &[GLint]) {
	//Send matrix uniforms to GPU
	for i in 0..mat_uniforms.len() {
		bind_matrix4(program, mat_uniforms[i], mats[i]);
	}

	//Send vector uniforms to GPU
	for i in 0..vec_uniforms.len() {
		gl::Uniform4fv(uniform_location(program, vec_uniforms[i]), 1, &[vecs[i].x, vecs[i].y, vecs[i].z, vecs[i].w] as *const GLfloat);
	}

	//Send byte uniforms to the GPU
	for i in 0..byte_uniforms.len() {
		gl::Uniform1i(uniform_location(program, byte_uniforms[i]), bytes[i]);
	}
}

pub unsafe fn bind_material(program: GLuint, material: &Option<mtl::Material>) {
	const NAMES: [&str; 3] = ["ambient_material", "diffuse_material", "specular_material"];
	let (colors, specular_coefficient) = match material {
		Some(material) => {
			let amb = [material.color_ambient.r as f32, material.color_ambient.g as f32, material.color_ambient.b as f32];
			let diff = [material.color_diffuse.r as f32, material.color_diffuse.g as f32, material.color_diffuse.b as f32];
			let spec = [material.color_specular.r as f32, material.color_specular.g as f32, material.color_specular.b as f32];
			let spec_co = material.specular_coefficient as f32;
			([amb, diff, spec], spec_co)
		}
		None => {
			panic!("This really should be unreachable");
		}
	};
	for i in 0..NAMES.len() {
		gl::Uniform3fv(uniform_location(program, NAMES[i]), 1, &colors[i] as *const GLfloat);
	}
	gl::Uniform1f(uniform_location(program, "specular_coefficient"), specular_coefficient);
}

//Binds "buffer" to the next available attribute of vao as an array of mat4
pub unsafe fn bind_instanced_matrices(vao: GLuint, attribute_count: usize, buffer: &[f32], count: usize) {
	let matrices_buffer = gl_gen_buffer();
	gl::BindBuffer(gl::ARRAY_BUFFER, matrices_buffer);
	gl::BufferData(gl::ARRAY_BUFFER, (count * 16 * mem::size_of::<GLfloat>()) as GLsizeiptr, &buffer[0] as *const GLfloat as *const c_void, gl::STATIC_DRAW);
	gl::BindVertexArray(vao);

	for i in 0..4 {
		let current_attribute = (attribute_count + i) as GLuint;
		gl::VertexAttribPointer(current_attribute, 4, gl::FLOAT, gl::FALSE, 16 * mem::size_of::<GLfloat>() as GLsizei, (4 * i * mem::size_of::<GLfloat>()) as *const c_void);
		gl::VertexAttribDivisor(current_attribute, 1);
		gl::EnableVertexAttribArray(current_attribute);
	}
}