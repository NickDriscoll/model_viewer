#version 330 core
in vec2 position;
in vec3 v_color;
out vec3 f_color;

uniform mat4 projection;

void main() {
    f_color = v_color;
    gl_Position = projection * vec4(position, 0.0, 1.0);
}