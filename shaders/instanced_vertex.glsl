#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex_coords;
layout(location = 3) in vec3 origin;
out vec4 f_pos;
out vec4 f_normal;
out vec2 v_tex_coords;
out vec4 shadow_coord;

uniform mat4 view_projection;
uniform mat4 shadow_vp;

void main() {
    //Calculate model_matrix with this instance's position
    mat4 model_matrix = mat4(
        1.0, 0.0, 0.0, origin.x,
        0.0, 1.0, 0.0, origin.y,
        0.0, 0.0, 1.0, origin.z,
        0.0, 0.0, 0.0, 1.0
    );
    model_matrix = transpose(model_matrix);

	//Send world space representation of position
	f_pos = model_matrix * vec4(position, 1.0);

	//Send world space representation of normal vector
	//We use a normal matrix instead of just the model matrix so that non-uniform scaling doesn't mess up the normal vector
	mat4 normal_matrix = transpose(mat4(inverse(mat3(model_matrix))));
	f_normal = normal_matrix * vec4(normal, 0.0);

	//Send tex coords
	v_tex_coords = tex_coords;

	//Calculate vertex's position in light space
	shadow_coord = shadow_vp * model_matrix * vec4(position, 1.0);

	//Transform vertex position with model-view-projection matrix
	gl_Position = view_projection * model_matrix * vec4(position, 1.0);
}
