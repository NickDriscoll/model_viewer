#version 330 core

in vec4 f_pos;
in vec4 f_normal;
in vec2 v_tex_coords;
out vec4 frag_color;

uniform sampler2D tex;
uniform vec4 light_position;
uniform vec4 view_position;

const vec3 LIGHT_COLOR = vec3(1.0, 1.0, 1.0);
const float AMBIENT_STRENGTH = 0.1;
const float SHININESS = 8.0;

void main() {
	//Normalize any vectors
	vec4 norm = normalize(f_normal);

	//Get raw texel
	vec3 tex_color = texture(tex, v_tex_coords).rgb;

	//Get light direction vector from light position
	//From frag location to light source
	vec4 light_direction = normalize(light_position - f_pos);

	//Get ambient contribution
	vec3 ambient = AMBIENT_STRENGTH * LIGHT_COLOR;

	//Get diffuse contribution
	float diff = max(0.0, dot(norm, light_direction));
	vec3 diffuse = diff * LIGHT_COLOR;

	//Get specular contribution (blinn-phong)
	vec4 view_direction = normalize(view_position - f_pos);
	vec4 half_dir = normalize(light_direction + view_direction);
	float specular_angle = max(0.0, dot(norm, half_dir));
	vec3 specular = pow(specular_angle, SHININESS) * LIGHT_COLOR;

	//Calculate distance attenuation
	float attenuation = clamp(2.0 / length(light_position - f_pos), 0.0, 1.0);

	vec3 result = attenuation * tex_color * (ambient + diffuse + specular);
	frag_color = vec4(result, 1.0);
}
