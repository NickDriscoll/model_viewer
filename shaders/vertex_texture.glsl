#version 330 core

in vec3 position;
in vec3 normal;
in vec2 tex_coords;
out vec3 f_normal;
out vec2 v_tex_coords;

uniform mat4 mvp;

void main() {
	f_normal = normal;
	v_tex_coords = tex_coords;
	gl_Position = mvp * vec4(position, 1.0);
}
