#version 330 core

in vec3 position;

uniform mat4 shadowMVP;

void main() {
    gl_Position = shadowMVP * vec4(position, 1.0);
}