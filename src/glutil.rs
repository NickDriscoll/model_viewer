use gl::types::*;
use std::ffi::CString;
use std::str;
use std::io::Read;
use std::fs::File;
use std::ptr;

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
	let mvp_str = CString::new(name.as_bytes()).unwrap();
	gl::GetUniformLocation(program, mvp_str.as_ptr())
}