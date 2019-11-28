#version 330 core

in vec3 position;
layout(location = 2) in vec2 tex_coords;
layout(location = 3) in mat4 model_matrix;
out vec2 v_tex_coords;

uniform mat4 shadowVP;

void main() {
    //Send tex coords
    v_tex_coords = tex_coords;
    gl_Position = shadowVP * model_matrix * vec4(position, 1.0);
}