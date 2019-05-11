#version 330 core

in vec3 position;
in vec3 color;
out vec3 f_color;

uniform mat4 mvp;

void main() {
	f_color = color;
	gl_Position = mvp * vec4(position, 1.0);
}
