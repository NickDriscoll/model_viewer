#version 330 core

in vec2 v_tex_coords;
in vec4 f_normal;
out vec4 color;

uniform sampler2D tex;

const vec4 LIGHT_DIRECTION = vec4(0.0, -1.0, 0.0, 0.0);

void main() {
	vec3 tex_color = texture(tex, v_tex_coords).rgb;

	//If f_normal and LIGHT_DIRECTION are normalized, the dot product will return cos(theta)
	float intensity = max(0.0, dot(f_normal, LIGHT_DIRECTION));

	vec3 result = tex_color * intensity;
	color = vec4(tex_color, 1.0);
}
