#version 330 core

in vec4 f_pos;
in vec4 f_normal;
in vec2 v_tex_coords;
in vec4 shadow_coord;
out vec4 frag_color;

uniform sampler2D tex;
uniform sampler2D shadow_map;
uniform vec4 view_position;		//Origin of view space
uniform vec4 light_direction;	//Vector pointing at the parallel light source

//Material properties
uniform vec3 ambient_material;
uniform vec3 diffuse_material;
uniform vec3 specular_material;
uniform float specular_coefficient;

//Flags
uniform bool using_material;
uniform bool lighting;

const vec3 LIGHT_COLOR = vec3(1.0, 1.0, 1.0);
const float AMBIENT_STRENGTH = 0.05;
const float ATTENUATION_CONSTANT = 1.0;
const float BRIGHTNESS = 0.75;

void main() {
	//Normalize any vectors
	vec4 norm = normalize(f_normal);

	//Get raw texel
	vec4 tex_color = texture(tex, v_tex_coords);

	//Exit early if we're in a shadow
	if (texture(shadow_map, shadow_coord.xy).z < shadow_coord.z) {
		frag_color = vec4(0.0, 0.0, 0.0, 1.0);
		return;
	}

	//Exit early if we're not doing lighting calculations
	if (!lighting) {
		if (using_material)
			frag_color = vec4(diffuse_material + specular_material, 1.0);
		else
			frag_color = vec4(tex_color.rgb, 1.0);
		return;
	}

	//Get ambient contribution
	vec3 ambient_light = AMBIENT_STRENGTH * LIGHT_COLOR;

	//Get diffuse contribution
	float diff = max(0.0, dot(norm, light_direction));
	vec3 diffuse_light = diff * LIGHT_COLOR;

	//Get specular contribution (blinn-phong)
	vec4 view_direction = normalize(view_position - f_pos);
	vec4 half_dir = normalize(light_direction + view_direction);
	float specular_angle = max(0.0, dot(norm, half_dir));
	vec3 specular_light = pow(0.0, specular_coefficient) * LIGHT_COLOR;

	//Calculate distance attenuation
	//float attenuation = clamp(ATTENUATION_CONSTANT / length(light_position - f_pos), 0.0, 1.0);
	float attenuation = 1.0;
	
	vec3 result = vec3(0.0);
	if (using_material) {
		result = BRIGHTNESS * attenuation * (ambient_light * ambient_material + diffuse_light * diffuse_material + specular_light * specular_material);
	} else {
		result = BRIGHTNESS * attenuation * tex_color.rgb * (ambient_light + diffuse_light + specular_light);
	}
	
	frag_color = vec4(result, 1.0);
	//frag_color = texture(shadow_map, shadow_coord.xy).zzzz;
	//frag_color = vec4(norm.rgb, 1.0);
}
