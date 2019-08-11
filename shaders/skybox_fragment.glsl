#version 330 core
in vec3 tex_coord;

uniform samplerCube skybox;

void main() {
    gl_FragColor = texture(skybox, tex_coord);
}