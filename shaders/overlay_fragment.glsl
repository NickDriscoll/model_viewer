#version 330 core

in vec2 v_tex_coords;
out vec4 frag_color;

uniform sampler2D tex;

void main() {
	vec3 tex_color = texture(tex, v_tex_coords).rgb;

	frag_color = vec4(tex_color, 1.0);
}