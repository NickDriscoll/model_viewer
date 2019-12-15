#version 330 core

in vec3 position;
layout(location = 2) in vec2 tex_coords;
out vec2 v_tex_coords;

uniform mat4 shadowMVP;

void main() {
    //Send tex coords
    v_tex_coords = tex_coords;
    gl_Position = shadowMVP * vec4(position, 1.0);
}