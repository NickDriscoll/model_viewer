use gl::types::*;

//A renderable 3D thing
#[derive(Clone)]
pub struct Mesh {
	pub vao: GLuint, //Vertex array object
	pub model_matrix: glm::TMat4<f32>, //Matrix that transforms points in model space to world space
	pub program: GLuint, //GLSL program to be rendered with
	pub texture: Option<GLuint>, //Texture
	pub indices_count: GLsizei //Number of indices in index array
}

impl Mesh {
	pub fn new(vao: GLuint, model_matrix: glm::TMat4<f32>, program: GLuint, texture: Option<GLuint>, indices_count: GLsizei) -> Self {
		Mesh {
			vao,
			model_matrix,
			program,
			texture,
			indices_count
		}
	}
}