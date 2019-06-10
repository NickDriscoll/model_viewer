use gl::types::*;
use openvr::ControllerState;
use std::slice::{Iter, IterMut};
use std::ops::{Index, IndexMut};

//A renderable 3D thing
pub struct Mesh {
	pub vao: GLuint, //Vertex array object
	pub model_matrix: glm::TMat4<f32>, //Matrix that transforms points in model space to world space
	pub program: GLuint, //GLSL program to be rendered with
	pub texture: Option<GLuint>, //Texture
	pub indices_count: GLsizei //Number of indices in index array
}

impl Mesh {
	pub fn new(vao: GLuint, model_matrix: glm::TMat4<f32>, glprogram: GLuint, texture: Option<GLuint>, indices_count: GLsizei) -> Self {
		Mesh {
			vao,
			model_matrix,
			program: glprogram,
			texture,
			indices_count
		}
	}
}

//Struct of arrays that stores VR controller data.
pub struct Controllers {
	pub controller_indices: [Option<u32>; Self::NUMBER_OF_CONTROLLERS],
	pub controller_mesh_indices: [Option<usize>; Self::NUMBER_OF_CONTROLLERS],
	pub controller_states: [Option<ControllerState>; Self::NUMBER_OF_CONTROLLERS],
	pub previous_controller_states: [Option<ControllerState>; Self::NUMBER_OF_CONTROLLERS]
}

impl Controllers {
	pub const NUMBER_OF_CONTROLLERS: usize = 2;

	pub fn new() -> Self {
		let controller_indices = [None; Self::NUMBER_OF_CONTROLLERS];
		let controller_mesh_indices = [None; Self::NUMBER_OF_CONTROLLERS];
		let controller_states = [None; Self::NUMBER_OF_CONTROLLERS];
		let previous_controller_states = [None; Self::NUMBER_OF_CONTROLLERS];

		Controllers {
			controller_indices,
			controller_mesh_indices,
			controller_states,
			previous_controller_states
		}
	}
}

//A wrapper for the useful Vec<Option<T>> pattern
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

	pub fn two_mut_refs(&mut self, index1: usize, index2: usize) -> (&mut Option<T>, &mut Option<T>) {
		let (first, second) = self.optionvec.split_at_mut(index1 + 1);
		let first_len = first.len();

		(&mut first[first_len - 1], &mut second[index2 - index1 - 1])
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