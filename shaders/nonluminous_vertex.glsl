#version 330 core

in vec3 position;
in vec3 normal;
in vec2 tex_coords;
out vec4 f_pos;
out vec4 f_normal;
out vec2 v_tex_coords;

uniform mat4 mvp;
uniform mat4 model_matrix;

void main() {
	//Send world space representation of position
	f_pos = model_matrix * vec4(position, 1.0);

	//Send world space representation of normal vector
	//Create normal matrix that protects against the effects of non-uniform scaling
	mat3 normal_matrix = mat3(transpose(inverse(model_matrix)));
	f_normal = vec4(normal_matrix * normal, 0.0);

	//Send tex coords
	v_tex_coords = tex_coords;

	//Transform vertex position with model-view-projection matrix
	gl_Position = mvp * vec4(position, 1.0);
}
