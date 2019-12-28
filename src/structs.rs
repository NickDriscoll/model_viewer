use gl::types::*;
use openvr::ControllerState;
use std::slice::{Iter, IterMut};
use std::ops::{Index, IndexMut};
use crate::*;

pub struct MeshData {
	pub vertices: Vec<f32>,
	pub indices: Vec<u16>,
	pub geo_boundaries: Vec<GLsizei>,
	pub materials: Vec<Option<mtl::Material>>
}

//A renderable 3D thing
#[derive(Clone)]
pub struct Mesh {
	pub vao: GLuint, //Vertex array object
	pub geo_boundaries: Vec<GLsizei>, //The start of each geometry in the vao
	pub materials: Option<Vec<Option<mtl::Material>>>, //The materials associated with this mesh, in the same order as geo_boundaries
	pub model_matrix: glm::TMat4<f32>, //Matrix that transforms points in model space to world space
	pub texture: GLuint, //Texture
	pub specular_coefficient: f32,
	pub render_pass_visibilities: [bool; RENDER_PASSES]
}

impl Mesh {
	pub fn new(vao: GLuint, model_matrix: glm::TMat4<f32>, texture: GLuint, geo_boundaries: Vec<GLsizei>, materials: Option<Vec<Option<mtl::Material>>>) -> Self {
		Mesh {
			vao,
			geo_boundaries,
			materials,
			model_matrix,
			texture,
			specular_coefficient: 8.0,
			render_pass_visibilities: [true, true, true]
		}
	}
}

pub struct Camera {	
	pub position: glm::TVec4<f32>,	//In world space
	pub velocity: glm::TVec4<f32>,	//In view space
	pub yaw: f32, 					//In radians
	pub pitch: f32, 				//In radians
	pub fov: f32,					//In degrees
	pub fov_delta: f32,
	pub speed: f32,
	pub attached_to_hmd: bool
}

impl Camera {
	pub fn new(pos: glm::TVec3<f32>) -> Self {
		Camera {
			position: glm::vec4(pos.x, pos.y, pos.z, 1.0),
			velocity: glm::vec4(0.0, 0.0, 0.0, 0.0),
			yaw: 0.0,
			pitch: 0.0,
			fov: 90.0,
			fov_delta: 0.0,
			speed: 2.0,
			attached_to_hmd: true
		}
	}

	pub fn view_matrix(&self) -> glm::TMat4<f32> {
		glm::rotation(self.pitch, &glm::vec3(1.0, 0.0, 0.0)) *
		glm::rotation(self.yaw, &glm::vec3(0.0, 1.0, 0.0)) *
		glm::translation(&(-glm::vec4_to_vec3(&self.position)))		//We negate the position here so that the idea of the camera's position is intuitive
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

	//Flag should be one of the constants defined in openvr::button_id
	pub fn pressed_this_frame(&self, controller_index: usize, flag: u32) -> bool {
		if let (Some(state), Some(p_state)) = (self.states[controller_index], self.previous_states[controller_index]) {
			state.button_pressed & (1 as u64) << flag != 0 && p_state.button_pressed & (1 as u64) << flag == 0
		} else {
			false
		}
	}

	//Flag should be one of the constants defined in openvr::button_id
	pub fn holding_button(&self, controller_index: usize, flag: u32) -> bool {
		if let Some(state) = self.states[controller_index] {
			state.button_pressed & (1 as u64) << flag != 0
		} else {
			false
		}
	}

	//Flag should be one of the constants defined in openvr::button_id
	pub fn released_this_frame(&self, controller_index: usize, flag: u32) -> bool {
		if let (Some(state), Some(p_state)) = (self.states[controller_index], self.previous_states[controller_index]) {
			state.button_pressed & (1 as u64) << flag == 0 && p_state.button_pressed & (1 as u64) << flag != 0
		} else {
			false
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

		//Fill the empty space if one was found, push onto the end otherwise
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

	pub fn get_element(&mut self, index: Option<usize>) -> Option<&mut T> {	
		match index {
			Some(i) => {
				self[i].as_mut()
			}
			None => { None }
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

	pub fn _iter_mut(&mut self) -> IterMut<Option<T>> {
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

//Data related to rendering a particular frame
pub struct RenderContext<'a> {
	pub p_matrices: &'a [glm::TMat4<f32>],						//The projection matrices for a given frame
	pub v_matrices: &'a [glm::TMat4<f32>],						//The view matrices for a given frame
	pub view_positions: [glm::TVec4<f32>; RENDER_PASSES],		//The origins of the v_matrices with respect to world space
	pub light_direction: &'a glm::TVec4<f32>,					//The direction of the parallel light source
	pub shadow_map: GLuint,										//The shadow map for the parallel light source
	pub shadow_vp: &'a glm::TMat4<f32>,							//The matrix that transforms coordinate in world space to light space
	pub is_lighting: bool										//Flag that determines whether or not to apply the lighting model
}

impl<'a> RenderContext<'a> {
	pub fn new(p_matrices: &'a [glm::TMat4<f32>], v_matrices: &'a [glm::TMat4<f32>], light_direction: &'a glm::TVec4<f32>, shadow_map: GLuint, shadow_vp: &'a glm::TMat4<f32>, is_lighting: bool) -> Self {
		let mut view_positions = [glm::zero(); RENDER_PASSES];
		for i in 0..v_matrices.len() {
			view_positions[i] = get_frame_origin(&glm::affine_inverse(v_matrices[i]));
		}

		RenderContext {
			p_matrices,
			v_matrices,
			view_positions,
			light_direction,
			shadow_map,
			shadow_vp,
			is_lighting
		}
	}
}

pub struct Terrain {
	pub surface_normals: Vec<glm::TVec3<f32>>,
	pub simplex: OpenSimplex,
	pub simplex_scale: f64,
	pub scale: f32,
	pub amplitude: f32,
	pub width: usize,
	pub subsquare_count: usize
}

pub struct ImageData {
	pub data: Vec<u8>,
	pub width: i32,
	pub height: i32,
	pub format: GLenum,
	pub internal_format: GLenum
}