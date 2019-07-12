#version 330 core

in vec2 v_tex_coords;
out vec4 frag_color;

uniform sampler2D tex;

void main() {
	vec4 tex_color = texture(tex, v_tex_coords);

	//frag_color = vec4(tex_color, 1.0);
	//frag_color = tex_color;
	//frag_color = vec4(1.0, 1.0, 1.0, tex_color.r);
	//frag_color = vec4(0.0, 1.0, 0.0, 1.0);
	//frag_color = vec4(1.0, 1.0, 1.0, step(0.1, tex_color.r));
	//frag_color = vec4(1.0, 1.0, 1.0, 0.0);

	if (tex_color.r < 0.1)
		discard;
	frag_color = vec4(1.0, 1.0, 1.0, 1.0);
}