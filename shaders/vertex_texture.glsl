#version 330 core

in vec3 position;
in vec3 normal;
in vec2 tex_coords;
out vec4 f_normal;
out vec2 v_tex_coords;

uniform mat4 mvp;
uniform mat4 model_matrix;

void main() {
	f_normal = model_matrix * vec4(normal, 0.0);
	v_tex_coords = tex_coords;
	gl_Position = mvp * vec4(position, 1.0);
}
