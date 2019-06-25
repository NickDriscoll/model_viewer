use gl::types::*;
use openvr::ControllerState;
use std::slice::Iter;
use std::ops::{Index, IndexMut};

//A renderable 3D thing
pub struct Mesh {
	pub vao: GLuint, //Vertex array object
	pub model_matrix: glm::TMat4<f32>, //Matrix that transforms points in model space to world space
	pub texture: GLuint, //Texture
	pub indices_count: GLsizei //Number of indices in index array
}

impl Mesh {
	pub fn new(vao: GLuint, model_matrix: glm::TMat4<f32>, texture: GLuint, indices_count: GLsizei) -> Self {
		Mesh {
			vao,
			model_matrix,
			texture,
			indices_count
		}
	}
}

pub struct Camera {	
	pub position: glm::TVec3<f32>,
	pub velocity: glm::TVec3<f32>,
	pub yaw: f32,
	pub pitch: f32,
	pub fov: f32,
	pub fov_delta: f32
}

impl Camera {
	pub const SPEED: f32 = 2.0;
	pub const FOV_SPEED: f32 = 5.0;

	pub fn new(position: glm::TVec3<f32>) -> Self {
		let velocity = glm::vec3(0.0, 0.0, 0.0);

		Camera {
			position,
			velocity,
			yaw: 0.0,
			pitch: 0.0,
			fov: 90.0,
			fov_delta: 0.0
		}
	}
}

/*
pub struct RenderContext {
	program: GLuint,
	mat4_locations: Vec<GLint>,
	vec4_locations: Vec<GLint>
}

impl RenderContext {
	pub fn new(program: GLuint, mat4_locations: Vec<GLint>, vec4_locations: Vec<GLint>) -> Self {
		RenderContext {
			program,
			mat4_locations,
			vec4_locations
		}
	}
}
*/

//Struct of arrays that stores VR controller data.
pub struct Controllers {
	pub device_indices: [Option<u32>; Self::NUMBER_OF_CONTROLLERS],
	pub mesh_indices: [Option<usize>; Self::NUMBER_OF_CONTROLLERS],
	pub states: [Option<ControllerState>; Self::NUMBER_OF_CONTROLLERS],
	pub previous_states: [Option<ControllerState>; Self::NUMBER_OF_CONTROLLERS],
	pub was_colliding: [bool; Self::NUMBER_OF_CONTROLLERS]
}

impl Controllers {
	pub const NUMBER_OF_CONTROLLERS: usize = 2;

	pub fn new() -> Self {
		let device_indices = [None; Self::NUMBER_OF_CONTROLLERS];
		let mesh_indices = [None; Self::NUMBER_OF_CONTROLLERS];
		let states = [None; Self::NUMBER_OF_CONTROLLERS];
		let previous_states = [None; Self::NUMBER_OF_CONTROLLERS];
		let was_colliding = [false; Self::NUMBER_OF_CONTROLLERS];

		Controllers {
			device_indices,
			mesh_indices,
			states,
			previous_states,
			was_colliding
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
		//Deternime which index is larger
		if index1 < index2 {
			let (first, second) = self.optionvec.split_at_mut(index1 + 1);
			let first_len = first.len();

			(&mut first[first_len - 1], &mut second[index2 - index1 - 1])
		} else {			
			let (first, second) = self.optionvec.split_at_mut(index2 + 1);
			let first_len = first.len();
			
			(&mut second[index1 - index2 - 1], &mut first[first_len - 1])
		}
	}

	pub fn iter(&self) -> Iter<Option<T>> {
		self.optionvec.iter()
	}

	/*
	pub fn iter_mut(&mut self) -> IterMut<Option<T>> {
		self.optionvec.iter_mut()
	}
	*/
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