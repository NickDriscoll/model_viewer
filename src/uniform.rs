#[derive(Debug)]
pub enum Uniform {
	Vector3(glm::TVec3<f32>),
	Matrix4(glm::TMat4<f32>)
}