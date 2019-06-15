#version 330 core

in vec4 f_pos;
in vec4 f_normal;
in vec2 v_tex_coords;
out vec4 color;

uniform sampler2D tex;

const float angle = 90; //Angle is in degrees
const vec3 LIGHT_COLOR = vec3(1.0, 1.0, 1.0);
const vec4 LIGHT_POSITION = vec4(0.0, 1.5, 0.0, 1.0);
const float AMBIENT_STRENGTH = 0.1;

void main() {
	//Normalize any vectors
	vec4 norm = normalize(f_normal);

	//Get raw texel
	vec3 tex_color = texture(tex, v_tex_coords).rgb;

	//Get light direction vector from light position
	vec4 light_direction = normalize(LIGHT_POSITION - f_pos);

	//Get ambient contribution
	vec3 ambient = AMBIENT_STRENGTH * LIGHT_COLOR;

	//Get diffuse contribution
	float diff = max(0.0, dot(norm, light_direction));
	vec3 diffuse = diff * LIGHT_COLOR;

	//Get specular contribution

	vec3 result = tex_color * (ambient + diffuse);
	color = vec4(result, 1.0);
}
