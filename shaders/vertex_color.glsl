#version 330 core

in vec3 position;
in vec3 color;
out vec3 f_color;
out vec3 frag_pos;

uniform mat4 mvp;
uniform mat4 model_matrix;

void main() {
	f_color = color;
	frag_pos = (model_matrix * vec4(position, 1.0)).xyz;
	gl_Position = mvp * vec4(position, 1.0);
}
