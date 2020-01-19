pub const fn cube() -> ([f32; 24], [u16; 36]) {
    const VERTICES: [f32; 24] = [
		-1.0, -1.0, -1.0,
		1.0, -1.0, -1.0,
		-1.0, 1.0, -1.0,
		1.0, 1.0, -1.0,
		-1.0, -1.0, 1.0,
		-1.0, 1.0, 1.0,
		1.0, -1.0, 1.0,
		1.0, 1.0, 1.0
	];
	const INDICES: [u16; 36] = [
		//Front
		0u16, 1, 2,
		3, 2, 1,
        
        //Left
		0, 2, 4,
		2, 5, 4,

		//Right
		3, 1, 6,
		7, 3, 6,

		//Back
		5, 7, 4,
		7, 6, 4,

		//Bottom
	    4, 1, 0,
    	4, 6, 1,
        
        //Top
		7, 5, 2,
		7, 2, 3
    ];
    (VERTICES, INDICES)    
}

pub const fn square() -> ([f32; 8], [u16; 6]) {
	const VERTICES: [f32; 8] = [
		0.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		1.0, 1.0
	];

	const INDICES: [u16; 6] = [
		0, 1, 2,
		3, 2, 1
	];

	(VERTICES, INDICES)
}