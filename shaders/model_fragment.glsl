#version 330 core

in vec4 f_pos;
in vec4 f_normal;
in vec2 v_tex_coords;
out vec4 frag_color;

uniform sampler2D tex;
//uniform vec4 light_position;
uniform vec4 view_position;
uniform float specular_coefficient;
uniform bool lighting;

const vec3 LIGHT_COLOR = vec3(1.0, 1.0, 1.0);
const float AMBIENT_STRENGTH = 0.1;
const float ATTENUATION_CONSTANT = 1.0;
const float BRIGHTNESS = 0.75;

void main() {
	//Normalize any vectors
	vec4 norm = normalize(f_normal);

	//Get raw texel
	vec4 tex_color = texture(tex, v_tex_coords);

	//Discard transparent fragments
	if (tex_color.a == 0.0)
		discard;

	//Exit early if we're not doing lighting calculations
	if (!lighting) {
		frag_color = vec4(tex_color.rgb, 1.0);
		return;
	}

	//Get light direction vector from light position
	//From frag location to light source
	//vec4 light_direction = normalize(light_position - f_pos);
	vec4 light_direction = normalize(vec4(0.0, 1.0, 0.0, 0.0));

	//Get ambient contribution
	vec3 ambient = AMBIENT_STRENGTH * LIGHT_COLOR;

	//Get diffuse contribution
	float diff = max(0.0, dot(norm, light_direction));
	vec3 diffuse = diff * LIGHT_COLOR;

	//Get specular contribution (blinn-phong)
	vec4 view_direction = normalize(view_position - f_pos);
	vec4 half_dir = normalize(light_direction + view_direction);
	float specular_angle = max(0.0, dot(norm, half_dir));
	vec3 specular = pow(specular_angle, specular_coefficient) * LIGHT_COLOR;

	//Calculate distance attenuation
	//float attenuation = clamp(ATTENUATION_CONSTANT / length(light_position - f_pos), 0.0, 1.0);
	float attenuation = 1.0;

	vec3 result = BRIGHTNESS * attenuation * tex_color.rgb * (ambient + diffuse + specular);
	
	frag_color = vec4(result, 1.0);
}
