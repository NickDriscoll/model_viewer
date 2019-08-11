use gl::types::*;
use openvr::ControllerState;
use std::slice::{Iter, IterMut};
use std::ops::{Index, IndexMut};

//A renderable 3D thing
#[derive(Clone)]
pub struct Mesh {
	pub vao: GLuint, //Vertex array object
	pub model_matrix: glm::TMat4<f32>, //Matrix that transforms points in model space to world space
	pub texture: GLuint, //Texture
	pub texture_path: String,
	pub shininess: f32,
	pub indices_count: GLsizei, //Number of indices in index array
	pub render_pass_visibilities: [bool; 3]
}

impl Mesh {
	pub fn new(vao: GLuint, model_matrix: glm::TMat4<f32>, path: &str, indices_count: GLsizei) -> Self {
		Mesh {
			vao,
			model_matrix,
			texture: 0,
			texture_path: path.to_string(),
			shininess: 8.0,
			indices_count,
			render_pass_visibilities: [true, true, true]
		}
	}
}

pub struct Camera {	
	pub position: glm::TVec3<f32>,	//In view space
	pub velocity: glm::TVec3<f32>,	//In view space
	pub yaw: f32, 					//In radians
	pub pitch: f32, 				//In radians
	pub fov: f32,					//In degrees
	pub fov_delta: f32,
	pub attached_to_hmd: bool
}

impl Camera {
	pub const SPEED: f32 = 10.0;
	pub const FOV_SPEED: f32 = 5.0;

	pub fn new(position: glm::TVec3<f32>) -> Self {
		Camera {
			position,
			velocity: glm::vec3(0.0, 0.0, 0.0),
			yaw: 0.0,
			pitch: 0.0,
			fov: 90.0,
			fov_delta: 0.0,
			attached_to_hmd: true
		}
	}


	pub fn get_freecam_matrix(&self) -> glm::TMat4<f32> {
		glm::rotation(self.pitch, &glm::vec3(1.0, 0.0, 0.0)) *
		glm::rotation(self.yaw, &glm::vec3(0.0, 1.0, 0.0)) *
		glm::translation(&self.position)
	}
}

//Struct of arrays that stores VR controller data.
pub struct Controllers {
	pub device_indices: [Option<u32>; Self::NUMBER_OF_CONTROLLERS],
	pub mesh_indices: [Option<usize>; Self::NUMBER_OF_CONTROLLERS],
	pub states: [Option<ControllerState>; Self::NUMBER_OF_CONTROLLERS],
	pub previous_states: [Option<ControllerState>; Self::NUMBER_OF_CONTROLLERS],
	pub colliding_with: [Vec<usize>; Self::NUMBER_OF_CONTROLLERS],
	pub collided_with: [Vec<usize>; Self::NUMBER_OF_CONTROLLERS]
}

impl Controllers {
	pub const NUMBER_OF_CONTROLLERS: usize = 2;

	pub fn new() -> Self {


		Controllers {
			device_indices: [None; Self::NUMBER_OF_CONTROLLERS],
			mesh_indices: [None; Self::NUMBER_OF_CONTROLLERS],
			states: [None; Self::NUMBER_OF_CONTROLLERS],
			previous_states: [None; Self::NUMBER_OF_CONTROLLERS],
			colliding_with: [Vec::new(), Vec::new()],
			collided_with: [Vec::new(), Vec::new()]
		}
	}
}

//A wrapper for the useful Vec<Option<T>> pattern
pub struct OptionVec<T> {
	optionvec: Vec<Option<T>>
}

impl<T> OptionVec<T> {
	pub fn with_capacity(size: usize) -> Self {
		OptionVec {
			optionvec: Vec::with_capacity(size)
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

		//Fill an empty space if one was found, push onto the end otherwise
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