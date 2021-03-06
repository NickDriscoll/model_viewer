#version 330 core

in vec3 position;
in vec3 normal;
in vec2 tex_coords;
out vec4 f_pos;
out vec4 f_normal;
out vec2 v_tex_coords;
out vec4 shadow_coord;

uniform mat4 mvp;
uniform mat4 model_matrix;
uniform mat4 shadow_mvp;

void main() {
	//Send world space representation of position
	f_pos = model_matrix * vec4(position, 1.0);

	//Send world space representation of normal vector
	//We use a normal matrix instead of just the model matrix so that non-uniform scaling doesn't mess up the normal vector
	mat4 normal_matrix = transpose(mat4(inverse(mat3(model_matrix))));
	f_normal = normal_matrix * vec4(normal, 0.0);

	//Send tex coords
	v_tex_coords = tex_coords;

	//Calculate vertex's position in light space
	shadow_coord = shadow_mvp * vec4(position, 1.0);

	//Transform vertex position with model-view-projection matrix
	gl_Position = mvp * vec4(position, 1.0);
}
