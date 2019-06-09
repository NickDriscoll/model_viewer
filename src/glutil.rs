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

pub unsafe fn compile_shader(shadertype: GLenum, source: &str) -> GLuint {
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

pub unsafe fn compile_shader_from_file(shadertype: GLenum, path: &str) -> GLuint {
	let mut source = String::new();
	File::open(path).unwrap().read_to_string(&mut source).unwrap();
	compile_shader(shadertype, &source)
}

pub unsafe fn compile_program_from_files(vertex_path: &str, fragment_path: &str) -> GLuint {
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

pub unsafe fn get_uniform_location(program: GLuint, name: &str) -> GLint {
	let cstring = CString::new(name.as_bytes()).unwrap();
	gl::GetUniformLocation(program, cstring.as_ptr())
}

/*
pub unsafe fn bind_program_and_uniforms(program: &GLProgram, matrix_values: &[glm::TMat4<f32>], vector_values: &[glm::TVec3<f32>]) {
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
*/

pub unsafe fn render_mesh(mesh: &Mesh, p_matrix: &glm::TMat4<f32>, v_matrix: &glm::TMat4<f32>, mvp_location: GLint) {
	//bind_program_and_uniforms(&mesh.program, &mesh.matrix_values, &mesh.vector_values);
	gl::UseProgram(mesh.program);

	let mvp = p_matrix * v_matrix * mesh.model_matrix;
	gl::UniformMatrix4fv(mvp_location, 1, gl::FALSE, &flatten_glm(&mvp) as *const GLfloat);

	if let Some(tex) = mesh.texture {
		gl::BindTexture(gl::TEXTURE_2D, tex);
	}

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

pub unsafe fn create_vertex_array_object(vertices: &[f32], indices: &[u16], attribute_sizes: &[usize]) -> GLuint {
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

	let mut vertex_stride = 0;
	for size in attribute_sizes {
		vertex_stride += size;
	}

	let byte_stride = vertex_stride as i32 * mem::size_of::<GLfloat>() as i32;

	//Configure and enable the vertex attributes
	let mut accumulated_size = 0;
	for i in 0..attribute_sizes.len() {
		gl::VertexAttribPointer(i as u32, attribute_sizes[i] as i32, gl::FLOAT, gl::FALSE, byte_stride, (accumulated_size * mem::size_of::<GLfloat>()) as *const c_void);
		gl::EnableVertexAttribArray(i as u32);
		accumulated_size += attribute_sizes[i];
	}

	vao
}

pub unsafe fn load_texture(path: &str) -> GLuint {
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