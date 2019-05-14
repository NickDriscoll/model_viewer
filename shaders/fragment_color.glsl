#version 330 core

in vec3 f_color;
out vec4 color;

void main() {
	float ambient_strength = 0.1;
	vec3 result = ambient_strength * f_color;
	color = vec4(result, 1.0);
}
