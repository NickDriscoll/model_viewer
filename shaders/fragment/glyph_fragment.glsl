#version 330 core
in vec2 glyph_uvs;
out vec4 frag_color;

uniform sampler2D glyph_texture;

void main() {
    if (texture(glyph_texture, glyph_uvs).r == 0.0)
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
    else
        frag_color = vec4(1.0, 0.0, 0.0, 1.0);
}