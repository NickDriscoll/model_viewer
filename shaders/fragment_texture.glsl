#version 330 core

in vec2 v_tex_coords;
out vec4 color;

uniform sampler2D tex;
uniform vec3 light_pos;

void main() {
	float ambient_strength = 0.1;
	vec3 result = ambient_strength * texture(tex, v_tex_coords).rgb;
	color = vec4(result, 1.0);
}