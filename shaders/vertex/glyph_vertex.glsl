#version 330 core
in vec3 position;
in vec3 v_color;
in vec2 uv;
out vec3 f_color;
out vec2 glyph_uvs;

uniform mat4 projection;

void main() {
    f_color = v_color;
    glyph_uvs = uv;
    gl_Position = projection * vec4(position, 1.0);
}