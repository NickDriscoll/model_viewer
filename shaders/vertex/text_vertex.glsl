#version 330 core
in vec3 position;
in vec2 uv;
out vec2 glyph_uvs;

uniform mat4 projection;

void main() {
    glyph_uvs = uv;
    gl_Position = projection * vec4(position, 1.0);
}