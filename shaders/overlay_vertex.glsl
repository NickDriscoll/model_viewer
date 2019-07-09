#version 330 core

in vec2 position;
in vec2 tex_coords;
out vec2 v_tex_coords;

uniform mat4 projection_matrix;

void main() {
	v_tex_coords = tex_coords;

	gl_Position = vec4(position, 0.0, 1.0);
}