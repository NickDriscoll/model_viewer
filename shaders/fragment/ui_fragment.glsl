#version 330 core
in vec3 f_color;
out vec4 frag_color;

void main() {
    frag_color = vec4(f_color, 0.8);
}