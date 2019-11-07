#version 330 core

in vec3 position;
layout(location = 3) in vec3 origin;

uniform mat4 shadowVP;

void main() {
    mat4 model_matrix = mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        origin.x, origin.y, origin.z, 1.0
    );

    gl_Position = shadowVP * model_matrix * vec4(position, 1.0);
}