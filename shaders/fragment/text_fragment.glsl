#version 330 core
in vec2 glyph_uvs;
out vec4 frag_color;

uniform sampler2D glyph_texture;

void main() {
    if (texture(glyph_texture, glyph_uvs).r == 0.0)
        discard;
    frag_color = vec4(1.0, 1.0, 1.0, 1.0);
}