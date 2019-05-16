use gl::types::*;
use crate::glutil::get_uniform_location;
use crate::compile_program_from_files;

//A renderable 3D thing
#[derive(Clone)]
pub struct Mesh<'a> {
	pub vao: GLuint, //Vertex array object
	pub model_matrix: glm::TMat4<f32>, //Matrix that transforms points in model space to world space
	pub program: &'a GLProgram, //GLSL program to be rendered with
	pub texture: Option<GLuint>, //Texture
	pub indices_count: GLsizei, //Number of indices in index array
	pub matrix_values: Vec<glm::TMat4<f32>>,
	pub vector_values: Vec<glm::TVec3<f32>>
}

impl<'a> Mesh<'a> {
	pub fn new(vao: GLuint, model_matrix: glm::TMat4<f32>, glprogram: &'a GLProgram, texture: Option<GLuint>, indices_count: GLsizei) -> Self {
		let mut matrix_values = Vec::new();
		let mut vector_values = Vec::new();

		for _ in 0..glprogram.matrix_locations.len() {
			matrix_values.push(glm::identity());
		}

		for _ in 0..glprogram.vector_locations.len() {
			vector_values.push(glm::vec3(0.0, 0.0, 0.0));
		}

		Mesh {
			vao,
			model_matrix,
			program: glprogram,
			texture,
			indices_count,
			matrix_values,
			vector_values
		}
	}
}

pub struct GLProgram {
	pub name: GLuint,
	pub matrix_locations: Vec<GLint>,
	pub vector_locations: Vec<GLint>
}

impl GLProgram {
	pub fn new(name: GLuint) -> Self {
		GLProgram {
			name,
			matrix_locations: Vec::new(),
			vector_locations: Vec::new()
		}
	}

	pub unsafe fn from_files(vertex_file: &str, fragment_file: &str, matrices: &[&str], vectors: &[&str]) -> Self {
		let name = compile_program_from_files(vertex_file, fragment_file);
		let mut program = GLProgram::new(name);

		for uniform in matrices {
			let loc = get_uniform_location(name, uniform);
			program.matrix_locations.push(loc);
		}

		for uniform in vectors {
			let loc = get_uniform_location(name, uniform);
			program.vector_locations.push(loc);
		}

		program
	}
}
