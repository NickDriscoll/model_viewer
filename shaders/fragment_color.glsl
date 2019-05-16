#version 330 core

in vec3 f_color;
in vec3 frag_pos;
out vec4 color;

uniform vec3 light_pos;

void main() {
	float ambient_strength = 0.15;

	vec3 light_ray = light_pos - frag_pos;

	vec3 result = ambient_strength * f_color;
	color = vec4(result, 1.0);
}
