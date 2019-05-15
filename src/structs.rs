use gl::types::*;

//A renderable 3D thing
#[derive(Clone)]
pub struct Mesh {
	pub vao: GLuint, //Vertex array object
	pub model_matrix: glm::TMat4<f32>, //Matrix that transforms points in model space to world space
	pub program: GLuint, //GLSL program to be rendered with
	pub texture: Option<GLuint>, //Texture
	pub indices_count: GLsizei, //Number of indices in index array
	pub matrix_values: Vec<glm::TMat4<f32>>,
	pub vector_values: Vec<glm::TVec3<f32>>
}

impl Mesh {
	pub fn new(vao: GLuint, model_matrix: glm::TMat4<f32>, program: GLuint, texture: Option<GLuint>, indices_count: GLsizei) -> Self {
		Mesh {
			vao,
			model_matrix,
			program,
			texture,
			indices_count,
			matrix_values: Vec::new(),
			vector_values: Vec::new()
		}
	}
}

pub struct GLProgram {
	pub name: GLuint,
	pub matrix_locations: Vec<GLint>,
	pub vector_locations: Vec<GLint>
}