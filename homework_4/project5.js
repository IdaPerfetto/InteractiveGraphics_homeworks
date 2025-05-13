// This function takes the translation and two rotation angles (in radians) as input arguments.
// The two rotations are applied around x and y axes.
// It returns the combined 4x4 transformation matrix as an array in column-major order.
// You can use the MatrixMult function defined in project5.html to multiply two 4x4 matrices in the same format.
function GetModelViewMatrix( translationX, translationY, translationZ, rotationX, rotationY )
{
	// Rotation around X, column-major order
	let cosX = Math.cos(rotationX);
	let sinX = Math.sin(rotationX);
	let RotationMatrix_X = [
		1, 0, 0, 0,
		0, cosX, -sinX, 0,
		0, sinX, cosX, 0,
		0, 0, 0, 1
	];

	// Rotation around Y, column-major order
	let cosY = Math.cos(rotationY);
	let sinY = Math.sin(rotationY);
	let RotationMatrix_Y = [
		cosY, 0, sinY, 0,
		0, 1, 0, 0,
		-sinY, 0, cosY, 0,
		0, 0, 0, 1
	];

	// Translation, column-major order
	let Translation = [
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		translationX, translationY, translationZ, 1
	];

	// Let's apply the transformations in the order: rotation -> translation
	// so MV = T * Rx * Ry
	var mv = MatrixMult( Translation, RotationMatrix_X );
	mv = MatrixMult( mv, RotationMatrix_Y );
	return mv;
}

// VERTEX SHADER
// Let's process each vertex in the 3D model:
// for each input vertex we have the position (a_VertexPosition), the normal (a_VertexNormal;) and the texture coordinates (a_TextureCoordinates)
// if the boolean swapYZ is true, we swap Y and Z axes in the position vector (AdjustedPosition = vec3(AdjustedPosition.x, AdjustedPosition.z, AdjustedPosition.y)) and the normal vector (v_Normal = normal * vec3(a_VertexNormal.x, a_VertexNormal.z, a_VertexNormal.y))
// if the boolean swapYZ is false, the position vector remains unchanged (vec3 AdjustedPosition = a_VertexPosition) and so does the normal vector (v_Normal = normal * a_VertexNormal)
// then we apply the model-view-projection transformation matrix (mvp) to convert the 3D model's coordinates to 2D screen space (gl_Position = mvp * vec4(AdjustedPosition, 1.0))
// while the position vector is multiplied by the model-view transformation matrix (mv) and passed to the fragment shader (v_Position = (mv * vec4(AdjustedPosition, 1.0)).xyz;)
// and finally we pass the texture coordinates to the fragment shader (v_TextureCoordinates = a_TextureCoordinates)

var MeshVertexShader = `
	attribute vec3 a_VertexPosition;
	attribute vec2 a_TextureCoordinates;
	attribute vec3 a_VertexNormal;

	uniform mat4 mvp;
	uniform mat3 normal;
	uniform mat4 mv;
	uniform vec3 LightDirection;
	uniform bool swapYZ;
	
	varying vec2 v_TextureCoordinates;
	varying vec3 v_Normal;
	varying vec3 v_Position;

	void main()
	{
		vec3 AdjustedPosition = a_VertexPosition;
		v_Normal = normal * a_VertexNormal;
		if (swapYZ)
		{
			AdjustedPosition = vec3(AdjustedPosition.x, AdjustedPosition.z, AdjustedPosition.y);
			v_Normal = normal * vec3(a_VertexNormal.x, a_VertexNormal.z, a_VertexNormal.y);
		}
		gl_Position = mvp * vec4(AdjustedPosition, 1.0);
		v_Position = (mv * vec4(AdjustedPosition, 1.0)).xyz;
		v_TextureCoordinates = a_TextureCoordinates;
	}
`;

// FRAGMENT SHADER
// Let's now process the pixels:
// first we set the precision of floating point numbers to be used in the fragment shader to medium (mediump) which is generally a good balance between performance and visual quality
// for each pixel its color is defined as the sum of the ambient light contribution, the diffuse light contribution (interaction between light and surface) and the specular light contribution (gl_FragColor = vec4(Diffuse + Specular + Ambient, baseColor.a) where:
// baseColor is determined based on whether the ShowTexture boolean is true or false:
// if the boolean ShowTexture is true, it is sampled from the texture (baseColor = texture2D(TextureSampler, v_TextureCoordinates))
// if the boolean ShowTexture is false, it is set to white (baseColor = vec4(1.0))
// and the alpha channel (baseColor.a) is used to preserve any transparency from the texture or base color

// DIFFUSE COMPONENT
// the diffuse reflection is defined as Kd multiplied by the maximum between 0.0 and the dot product between N and L (float costheta = max(0.0, dot(norm, LightDirection)); vec4 Diffuse = Kd * costheta) where:
// N is the surface normal (vec3 norm = normalize(v_Normal))
// L is the lighting direction (LightDirection)
// Kd is the diffuse reflection coefficient (vec3 Kd = baseColor.rgb)

// SPECULAR COMPONENT
// the specular reflection is defined as Ks multiplied by the maximum between 0.0 and the dot product between N and H to the power of alpha (float cosphi = max(0.0, dot(norm, h)); vec4 Specular = Ks * pow(cosphi, Alpha)) where:
// alpha is the shininess exponent
// H is the halfway vector between the light direction (LightDirection) and the camera direction (vec3 cameraDirection = normalize(-v_Position)) (vec3 h = normalize(LightDirection + cameraDirection))
// Ks is the specular reflection coefficient which is fixed to a value of 0.5 for simplicity (vec3 Ks = vec3(0.5))

// AMBIENT COMPONENT
// the ambient reflection is the product between Ka and the intensity of the ambient light where:
// the ambient light is usually a constant (set to 0.1 as done here https://learnopengl.com/Lighting/Basic-Lighting)
// Ka is the ambient reflection coefficient
// however, for semplicity purposes, we assume the material reflects ambient light in the same way it reflects diffuse light (vec3 Ka = Kd)

var MeshFragmentShader = `
	
	precision mediump float;
	varying vec3 v_Normal;
    varying vec3 v_Position;
	varying vec2 v_TextureCoordinates;
	uniform vec3 LightDirection;
	uniform sampler2D TextureSampler;
	uniform bool ShowTexture;
	uniform float Alpha;
	

	void main()
	{
		vec3 norm = normalize(v_Normal);
		vec3 cameraDirection = normalize(-v_Position);
		vec3 h = normalize(LightDirection + cameraDirection);

	    // Unified baseColor logic
	    vec4 baseColor;
	    if (ShowTexture)
	        baseColor = texture2D(TextureSampler, v_TextureCoordinates);
	    else
	        baseColor = vec4(1.0);
		
		// Material properties
		vec3 Kd = baseColor.rgb;
		vec3 Ka = Kd;
		vec3 Ks = vec3(0.5);
		
		// Lighting components
		float costheta = max(dot(norm, LightDirection), 0.0);
	    float cosphi = max(dot(norm, h), 0.0);
	    float AmbientLight = 0.10;

	    vec3 Diffuse = Kd * costheta;
	    vec3 Specular = Ks * pow(cosphi, Alpha);  // shininess controlled by Alpha
	    vec3 Ambient = AmbientLight * Ka;

	    // Final color
	    gl_FragColor = vec4(Diffuse + Specular + Ambient, baseColor.a);
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
		this.mv = gl.getUniformLocation(this.prog, 'mv');
		this.mvp = gl.getUniformLocation(this.prog, 'mvp');
		this.norm = gl.getUniformLocation(this.prog, 'normal');
		this.swap = gl.getUniformLocation(this.prog, 'swapYZ');
		this.showTex = gl.getUniformLocation(this.prog, 'ShowTexture');
		this.sampTex = gl.getUniformLocation(this.prog, 'TextureSampler');
		this.lightDir = gl.getUniformLocation(this.prog, 'LightDirection');
		this.shininess = gl.getUniformLocation(this.prog, 'Alpha');

		// Get the location of the attribute variables
		this.vertPos = gl.getAttribLocation(this.prog, 'a_VertexPosition');
		this.vertNormal = gl.getAttribLocation(this.prog, 'a_VertexNormal');
		this.texCoord = gl.getAttribLocation(this.prog, 'a_TextureCoordinates');

		// Create buffers for storing the vertex positions and texture coordinates
		this.vertbuffer = gl.createBuffer();
		this.normbuffer = gl.createBuffer();
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
		this.Alpha = 100;
		this.LightDirection = [0, 0, 0]
	}
	
	// This method is called every time the user opens an OBJ file.
	// The arguments of this function is an array of 3D vertex positions, an array of 2D texture coordinates, and an array of vertex normals.
	// Every item in these arrays is a floating point value, representing one coordinate of the vertex position or texture coordinate.
	// Every three consecutive elements in the vertPos array forms one vertex position and every three consecutive vertex positions form a triangle.
	// Similarly, every two consecutive elements in the texCoords array form the texture coordinate of a vertex and every three consecutive  elements in the normals array form a vertex normal.
	// Note that this method can be called multiple times.
	setMesh( vertPos, texCoords, normals )
	{
		this.numTriangles = vertPos.length / 3; // each triangle is made up of 3 vertices -> the number of triangles is calculated by dividing the total number of vertices by 3

		// Vertex positions
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertbuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertPos), gl.STATIC_DRAW);

        // Normals
        gl.bindBuffer(gl.ARRAY_BUFFER, this.normbuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);

        // Texture coordinates
        gl.bindBuffer(gl.ARRAY_BUFFER, this.texbuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(texCoords), gl.STATIC_DRAW);
	}
	
	// This method is called when the user changes the state of the
	// "Swap Y-Z Axes" checkbox. 
	// The argument is a boolean that indicates if the checkbox is checked.
	swapYZ( swap )
	{
		this.swapYZBool = swap;
	}
	
	// This method is called to draw the triangular mesh.
	// The arguments are the model-view-projection transformation matrixMVP,
	// the model-view transformation matrixMV, the same matrix returned
	// by the GetModelViewProjection function above, and the normal
	// transformation matrix, which is the inverse-transpose of matrixMV.
	draw( matrixMVP, matrixMV, matrixNormal )
	{
		// Activate the shader program
		gl.useProgram(this.prog);

		// Set the Uniforms:
		// pass the model-view-projection matrix, the model-view matrix and the normal matrix to the shader
		gl.uniformMatrix4fv(this.mvp, false, matrixMVP);
        gl.uniformMatrix4fv(this.mv, false, matrixMV);
        gl.uniformMatrix3fv(this.norm, false, matrixNormal);

		// pass the swap ans show texture flags to the shader
		gl.uniform1i(this.swap, this.swapYZBool);
		gl.uniform1i(this.showTex, this.showTextureBool);

		// pass the light direction to the shader
		gl.uniform3fv(this.lightDir, this.LightDirection);

		// pass the shininess to the shader
		gl.uniform1f(this.shininess, this.Alpha);

		// Bind the vertex buffer and set the attribute pointer
		gl.bindBuffer(gl.ARRAY_BUFFER, this.vertbuffer);
		gl.vertexAttribPointer(this.vertPos, 3, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(this.vertPos);

		// Bind the normal buffer and set the attribute pointer
		gl.bindBuffer(gl.ARRAY_BUFFER, this.normbuffer);
		gl.vertexAttribPointer(this.vertNormal, 3, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(this.vertNormal);

		// Bind the texture coordinate buffer and set the attribute pointer
		gl.bindBuffer(gl.ARRAY_BUFFER, this.texbuffer);
		gl.vertexAttribPointer(this.texCoord, 2, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(this.texCoord);

		// Bind texture
		gl.activeTexture(gl.TEXTURE0); // this tells WebGL that TEXTURE0 is the currently active texture unit, meaning any texture operation that follows will be applied to this texture unit
		gl.bindTexture(gl.TEXTURE_2D, this.texture);
		gl.uniform1i(this.sampTex, 0);

	    // Draw the mesh (as triangles)
		gl.drawArrays( gl.TRIANGLES, 0, this.numTriangles );
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

	// This method is called to set the incoming light direction
	setLightDir( x, y, z )
	{
		this.LightDirection = [x, y, z];
	}

	// This method is called to set the shininess of the material
	setShininess( shininess )
	{
		this.Alpha = shininess;
	}
}
