use gl::types::*;
use std::ffi::CString;

pub unsafe fn get_uniform_location(program: GLuint, name: &str) -> GLint {
	let mvp_str = CString::new(name.as_bytes()).unwrap();
	gl::GetUniformLocation(program, mvp_str.as_ptr())
}