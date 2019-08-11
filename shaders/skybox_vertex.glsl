#version 330 core
in vec3 position;

out vec3 tex_coord;

uniform mat4 view_projection;

void main() {
    tex_coord = position;
    vec4 screen_space_pos = view_projection * vec4(position, 1.0);
    gl_Position = screen_space_pos;
}