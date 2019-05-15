#version 330 core

in vec3 f_color;
out vec4 color;

uniform vec3 light_pos;

void main() {
	float ambient_strength = 1.0;
	vec3 result = ambient_strength * f_color;
	color = vec4(result, 1.0);
}
