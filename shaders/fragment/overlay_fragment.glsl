#version 330 core

in vec2 v_tex_coords;
out vec4 frag_color;

uniform sampler2D tex;

void main() {
	vec4 tex_color = texture(tex, v_tex_coords);

	if (tex_color.r < 0.1)
		frag_color = vec4(0.0, 0.0, 0.0, 1.0);
	else
		frag_color = vec4(1.0, 1.0, 1.0, 1.0);
}