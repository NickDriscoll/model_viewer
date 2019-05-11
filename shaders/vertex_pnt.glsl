#version 330 core

in vec3 position;
in vec3 normal;
in vec2 tex_coords;
out vec2 v_tex_coords;

uniform mat4 mvp;
uniform float tex_scale = 1.0;

void main() {
	v_tex_coords = tex_scale * tex_coords;
	gl_Position = mvp * vec4(position, 1.0);
}
