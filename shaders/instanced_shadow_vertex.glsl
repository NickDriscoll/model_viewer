#version 330 core

in vec3 position;
layout(location = 3) in mat4 model_matrix;

uniform mat4 shadowVP;

void main() {
    gl_Position = shadowVP * model_matrix * vec4(position, 1.0);
}