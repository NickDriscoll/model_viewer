pub fn create_unit_sphere(depth: usize) -> Vec<f32> {
	let va = glm::vec4(0.0, 0.0, -1.0, 1.0);
	let vb = glm::vec4(0.0, 0.942809, 0.333333, 1.0);
	let vc = glm::vec4(-0.816497, -0.471405, 0.333333, 1.0);
	let vd = glm::vec4(0.816497, -0.471405, 0.333333, 1.0);

	let mut verts = Vec::new();

	divide_triangle(&va, &vb, &vc, depth, &mut verts);
	divide_triangle(&vd, &vc, &vb, depth, &mut verts);
	divide_triangle(&va, &vd, &vb, depth, &mut verts);
	divide_triangle(&va, &vc, &vd, depth, &mut verts);

	verts
}

fn divide_triangle(a: &glm::TVec4<f32>, b: &glm::TVec4<f32>, c: &glm::TVec4<f32>, count: usize, verts: &mut Vec<f32>) {
	if (count > 0) {
		let ab = glm::normalize(&glm::mix(a, b, 0.5));
		let ac = glm::normalize(&glm::mix(a, c, 0.5));
		let bc = glm::normalize(&glm::mix(b, c, 0.5));
		divide_triangle(&a, &ab, &ac, count - 1, verts);
		divide_triangle(&ab, &b, &bc, count - 1, verts);
		divide_triangle(&bc, &c, &ac, count - 1, verts);
		divide_triangle(&ab, &bc, &ac, count - 1, verts);
	} else {
		for i in 0..4 {
			verts.push(a[i]);
		}

		for i in 0..4 {
			verts.push(b[i]);
		}

		for i in 0..4 {
			verts.push(c[i]);
		}
	}
}