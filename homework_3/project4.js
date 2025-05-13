// This function takes the projection matrix, the translation, and two rotation angles (in radians) as input arguments.
// The two rotations are applied around x and y axes.
// It returns the combined 4x4 transformation matrix as an array in column-major order.
// The given projection matrix is also a 4x4 matrix stored as an array in column-major order.
// You can use the MatrixMult function defined in project4.html to multiply two 4x4 matrices in the same format.
function GetModelViewProjection( projectionMatrix, translationX, translationY, translationZ, rotationX, rotationY )
{
	// Rotation around X, column-major order
	let cosX = Math.cos(rotationX);
	let sinX = Math.sin(rotationX);
	let RotationMatrix_X = [
		1,    0,     0, 0,
		0, cosX, -sinX, 0,
		0, sinX,  cosX, 0,
		0,    0,     0, 1
	];

	// Rotation around Y, column-major order
	let cosY = Math.cos(rotationY);
	let sinY = Math.sin(rotationY);
	let RotationMatrix_Y = [
	 cosY, 0, sinY, 0,
	    0, 1,    0, 0,
	-sinY, 0, cosY, 0,
	    0, 0,    0, 1
	];

	// Translation, column-major order
	let Translation = [
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		translationX, translationY, translationZ, 1
	];

	// Let's apply the transformations in the order: rotation -> translation -> projection
	// so MVP = P * T * Rx * Ry
	var mvp = MatrixMult( projectionMatrix, Translation );
	mvp = MatrixMult( mvp, RotationMatrix_X );
	mvp = MatrixMult( mvp, RotationMatrix_Y );
	return mvp;
}


// VERTEX SHADER
// Let's process each vertex in the 3D model:
// for each input vertex we have the position (a_VertexPosition) and the texture coordinates (a_TextureCoordinates)
// if the boolean swapYZ is true, we swap Y and Z axes in the position vector (AdjustedPosition = vec3(AdjustedPosition.x, AdjustedPosition.z, AdjustedPosition.y))
// if the boolean swapYZ is false, the position vector remains unchanged (vec3 AdjustedPosition = a_VertexPosition)
// then we apply the model-view-projection transformation matrix (mvp) to convert the 3D model's coordinates to 2D screen space (gl_Position = mvp * vec4(AdjustedPosition, 1.0))
// and finally we pass the texture coordinates to the fragment shader (v_TextureCoordinates = a_TextureCoordinates)

var MeshVertexShader = `
	attribute vec3 a_VertexPosition;
	attribute vec2 a_TextureCoordinates;
	uniform mat4 mvp;
	uniform bool swapYZ;
	varying vec2 v_TextureCoordinates;
	void main()
	{
		vec3 AdjustedPosition = a_VertexPosition;
		if(swapYZ)
			AdjustedPosition = vec3(AdjustedPosition.x, AdjustedPosition.z, AdjustedPosition.y);
		gl_Position = mvp * vec4(AdjustedPosition, 1.0);
		v_TextureCoordinates = a_TextureCoordinates;
	}
`;
// FRAGMENT SHADER
// Let's now process the pixels:
// first we set the precision of floating point numbers to be used in the fragment shader to medium (mediump) which is generally a good balance between performance and visual quality
// if the boolean ShowTexture is true, we use the texture (TextureSampler) to get the color from the texture at the specified coordinates (v_TextureCoordinates) (gl_FragmentColor = texture2D(TextureSampler, v_TextureCoordinates))
// if the boolean ShowTexture is false, we use a debug color (the red channel is set to 1.0, and the green channel depends on the depth) instead of the texture (gl_FragmentColor = vec4(1.0, gl_FragCoord.z * gl_FragCoord.z, 0.0, 1.0))

var MeshFragmentShader = `
	precision mediump float;
	varying vec2 v_TextureCoordinates;
	uniform sampler2D TextureSampler;
	uniform bool ShowTexture;
	void main()
	{
		if (ShowTexture)
			gl_FragColor = texture2D(TextureSampler, v_TextureCoordinates);
		else
			gl_FragColor = vec4(1.0, gl_FragCoord.z * gl_FragCoord.z, 0.0, 1.0);
	}
`;

class MeshDrawer
{
	// The constructor is a good place for taking care of the necessary initializations.
	constructor()
	{
		// Compile and link the vertex and fragment shaders in the program (which is stored in this.prog and will be used to draw the mesh in WebGL)
		this.prog = InitShaderProgram(MeshVertexShader, MeshFragmentShader);
				
		// Get the location of the uniform variables
		this.mvp = gl.getUniformLocation(this.prog, 'mvp');
		this.swap = gl.getUniformLocation(this.prog, 'swapYZ');
		this.showTex = gl.getUniformLocation(this.prog, 'ShowTexture');
		this.sampTex = gl.getUniformLocation(this.prog, 'TextureSampler');
		
		// Get the location of the attribute variables
		this.vertPos = gl.getAttribLocation(this.prog, 'a_VertexPosition');
		this.texCoord = gl.getAttribLocation(this.prog, 'a_TextureCoordinates');
		
		// Create buffers for storing the vertex positions and texture coordinates
		this.vertbuffer = gl.createBuffer();
		this.texbuffer = gl.createBuffer();

		// Create and configure the texture
		this.texture = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, this.texture); // bind the texture object (this.texture) to the TEXTURE_2D target.
		
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); // set how the texture wraps in the horizontal direction: with CLAMP_TO_EDGE the texture coordinates are clamped between 0 and 1, and anything outside this range uses the color of the edge pixels
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); // set how the texture wraps in the vertical direction: same as the horizontal direction
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); // set the filtering method for minification (when the texture appears smaller on screen than its original size): with LINEAR the texture will be linearly interpolated between the closest texture pixels
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR); // set the filtering method for magnification (when the texture appears bigger on screen than its original size): same as minification

		// Default state variables
		this.swapYZBool = false;
		this.showTextureBool = true;
}
	
	// This method is called every time the user opens an OBJ file.
	// The arguments of this function is an array of 3D vertex positions and an array of 2D texture coordinates.
	// Every item in these arrays is a floating point value, representing one coordinate of the vertex position or texture coordinate.
	// Every three consecutive elements in the vertPos array forms one vertex position and every three consecutive vertex positions form a triangle.
	// Similarly, every two consecutive elements in the texCoords array form the texture coordinate of a vertex.
	// Note that this method can be called multiple times.
	setMesh( vertPos, texCoords )
	{
		this.numTriangles = vertPos.length / 3; // each triangle is made up of 3 vertices -> the number of triangles is calculated by dividing the total number of vertices by 3
		
		// Update the content of the vertex buffer objects with vertex positions
		gl.bindBuffer(gl.ARRAY_BUFFER, this.vertbuffer); // bind the buffers for vertex positions
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertPos), gl.STATIC_DRAW);

		// Update the content of the texture buffer with texture coordinates
		gl.bindBuffer(gl.ARRAY_BUFFER, this.texbuffer); // bind the buffers for texture coordinates
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(texCoords), gl.STATIC_DRAW);
	}
	
	// This method is called when the user changes the state of the "Swap Y-Z Axes" checkbox. 
	// The argument is a boolean that indicates if the checkbox is checked.
	swapYZ( swap )
	{
		this.swapYZBool = swap;
	}
	
	// This method is called to draw the triangular mesh.
	// The argument is the transformation matrix, the same matrix returned
	// by the GetModelViewProjection function above.
	draw( trans )
	{
		// Activate the shader program
		gl.useProgram(this.prog);

		// Set the Uniforms:
		// pass the model-view-projection matrix to the shader
		gl.uniformMatrix4fv(this.mvp, false, trans);

		// pass the swap ans show texture flags to the shader
		gl.uniform1i(this.swap, this.swapYZBool);
		gl.uniform1i(this.showTex, this.showTextureBool);
		
		// Bind the vertex buffer and set the attribute pointer
		gl.bindBuffer(gl.ARRAY_BUFFER, this.vertbuffer);
		gl.vertexAttribPointer(this.vertPos, 3, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(this.vertPos);
		
		// Bind the texture coordinate buffer and set the attribute pointer
		gl.bindBuffer(gl.ARRAY_BUFFER, this.texbuffer);
		gl.vertexAttribPointer(this.texCoord, 2, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(this.texCoord);

		// Bind texture
		gl.activeTexture(gl.TEXTURE0); // this tells WebGL that TEXTURE0 is the currently active texture unit, meaning any texture operation that follows will be applied to this texture unit
		gl.bindTexture(gl.TEXTURE_2D, this.texture);

	    // Draw the mesh (as triangles)
		gl.drawArrays(gl.TRIANGLES, 0, this.numTriangles);
	}
	
	// This method is called to set the texture of the mesh.
	// The argument is an HTML IMG element containing the texture data.
	setTexture( img )
	{
		// Bind the texture
		gl.bindTexture(gl.TEXTURE_2D, this.texture);
		// You can set the texture image data using the following command (given by the Professor):
		gl.texImage2D( gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img ); // this uploads the image to the texture

		// Now that we have a texture, it might be a good idea to set some uniform parameter(s) of the fragment shader, so that it uses the texture.
		// Mipmaps are precomputed textures at different levels of detail for performance optimization when textures are displayed at different sizes, let's compute them
		gl.generateMipmap(gl.TEXTURE_2D);
		// Let's now set the texture parameters
		gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT ); // Set the texture wrapping mode (S axis)
		gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT ); // Set the texture wrapping mode (T axis)
		gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR ); // Set filtering method for minification
		gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR ); // Set filtering method for magnification

		gl.activeTexture( gl.TEXTURE0 ); // this tells WebGL that TEXTURE0 is the currently active texture unit, meaning any texture operation that follows will be applied to this texture unit
		gl.bindTexture( gl.TEXTURE_2D, this.texture ); // bind the texture
		gl.useProgram(this.prog);
		gl.uniform1i(this.sampTex, 0);
	}
	
	// This method is called when the user changes the state of the "Show Texture" checkbox. 
	// The argument is a boolean that indicates if the checkbox is checked.
	showTexture( show )
	{
		this.showTextureBool = show;
	}
	
}
