use gl::types::*;
use std::slice::{Iter, IterMut};
use std::ops::{Index, IndexMut};
use crate::glutil::*;

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

pub struct OptionVec<T> {
	optionvec: Vec<Option<T>>
}

impl<T> OptionVec<T> {
	pub fn with_capacity(size: usize) -> Self {
		let optionvec = Vec::with_capacity(size);
		OptionVec {
			optionvec
		}
	}

	pub fn insert(&mut self, element: T) -> usize {
		let mut index = None;

		//Search for an empty space
		for i in 0..self.optionvec.len() {
			if let None = self.optionvec[i] {
				index = Some(i);
				break;
			}
		}

		//Fill the empty space if an index was found, push otherwise
		match index {
			Some(i) => {
				self.optionvec[i] = Some(element);
				i
			}
			None => {
				self.optionvec.push(Some(element));
				self.optionvec.len() - 1
			}
		}
	}

	pub fn split_at_mut(&mut self, mid: usize) -> (&mut [Option<T>], &mut [Option<T>]) {
		self.optionvec.split_at_mut(mid)
	}

	pub fn iter(&self) -> Iter<Option<T>> {
		self.optionvec.iter()
	}

	pub fn iter_mut(&mut self) -> IterMut<Option<T>> {
		self.optionvec.iter_mut()
	}
}

impl<T> Index<usize> for OptionVec<T> {
	type Output = Option<T>;

	fn index(&self, index: usize) -> &Self::Output {
		&self.optionvec[index]
	}
}

impl<T> IndexMut<usize> for OptionVec<T> {
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		&mut self.optionvec[index]
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
