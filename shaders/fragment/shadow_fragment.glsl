#version 330 core

in vec2 v_tex_coords;

uniform sampler2D tex;
uniform bool using_texture;

void main() {
    //Don't write a depth value if this fragment is transparent
    if (using_texture && texture(tex, v_tex_coords).a == 0.0)
        discard;
}