/*
This is the core file of the Bloomscape application that: 
- defines the application's global state
- manages the rendering pipeline
- implements the L-system generation logic
- handles all dynamic updates and animations
Overall Architecture:
1) Global Configuration (config): the file begins by defining a config object that centralizes all tunable parameters:
  from L-system rules and presets (inspired by "The Algorithmic Beauty of Plants") to animation speeds, environment settings (like the day/night cycle), and weather effects
  This object is made globally accessible (window.config) so it can be read and modified by the UI (ui.js)
2) Mathematical Utilities (Mat4): it includes a self-contained Mat4 object, a library of static functions for performing 4x4 matrix algebra essential for all 3D transformations
3) Camera Control (OrbitCamera): the OrbitCamera class implements a user-controlled camera that can orbit around a target point, pan across the scene, and zoom in and out
  It listens for mouse events (drag, scroll wheel) to update its position and orientation dynamically
4) Shader Compilation and Management: the file contains helper functions (compileShader, createProgram) to compile GLSL vertex and fragment shaders into executable WebGL programs
  It also defines the GLSL source code for all shaders used in the application, including:
  - a primary Blinn-Phong shader for objects with solid colors (VSHADER/FSHADER)
  - shaders for luminous objects like the sun and moon
  - specialized shaders for visual effects like sun rays, particles (rain, fireflies, stars), and procedural clouds
  - a shader for rendering textured objects (leaves and flowers)
  - a shader for the skybox gradient
  - a simple depth shader for rendering the shadow map
5) Rendering Primitives (PrimitiveDrawer, MeshDrawer): 2 classes are defined to handle the drawing of 3D objects:
  - MeshDrawer is used for standard indexed triangle meshes (like branches and the ground)
  - PrimitiveDrawer is a more general version capable of rendering non-indexed geometry and different primitive types (like gl.POINTS for particles)
6) L-System Engine: it contains the core logic for procedural plant generation with
  - expandLSystem: a function that takes an axiom and a set of rules and iteratively expands the L-system string for a given number of iterations
  - interpretLSystem: a "turtle graphics" interpreter that parses the final L-system string
  - LSysToSceneDesc and buildLSystem: helper functions that orchestrate the generation pipeline, converting the interpreted turtle commands into a scene description that can be rendered
7) Main Application Logic (window.addEventListener("load", ...)): this is the entry point of the application with
  - Initialization: it sets up the WebGL context, resizes the canvas, compiles all shaders, and initializes the camera and UI event listeners
  - Asset Loading: it asynchronously loads all necessary 3D models (.glb files for leaves and flowers) and prepares their geometry and textures for rendering
  - render Loop (render): this function is called repeatedly via requestAnimationFrame, in each frame, it does:
    - State Update: calculates deltaTime, advances the day/night cycle, processes the plant's animated growth and handles its life cycle (dying if not watered)
    - Shadow Pass: renders the scene from the light's perspective into a depth texture (the shadow map)
    - Main Pass: renders the full scene from the camera's perspective, drawing the skybox, environmental effects (sun, moon, particles), the ground, and finally the plant itself, applying lighting, materials, textures, shadows, and animations (including the complex, multi-part GLB animation for the flowers)
*/
const config = {
  // This object acts as a centralized "control panel" for the entire application, holding all the parameters that define the behavior of the plant, the animation, and the environment
  // L-System parameters
  lSystem: {
    // parameters controlled via UI
    angle: 25.7,                     // angle between branches
    iterations: 3,                   // number of iterations
    presets: [                       // from the text "The Algorithmic Beuty of Plants", figures 1.24 and 1.25
      // name: arbitrary name given to the preset
      // axiom: starting string  for the L-system
      // rules: production rules
      { // figure 1.24 (a)
        name: "High Slim Bush",
        axiom: "F",
        //rules: { "F": "F[+F]F[-F]F" }
        rules: { "F": "F[+FL]F[-FL]F" }  // added leaves (explicitly place a leaf at the end of each new sub-branch)
      },
      { // figure 1.24 (b)
        name: "High Bush",
        axiom: "F",
        //rules: { "F": "F[+F]F[-F][F]" }
        rules: { "F": "F[+FL]F[-FL][F]" }  // added leaves
      },
      { // figure 1.24 (c)
        name: "Shrub",
        axiom: "F",
        //rules: { "F": "FF-[-F+F+F]+[+F-F-F]" },
        rules: { "F": "FF-[-FL+FL+FL]+[+FL-FL-FL]" }  // added leaves
      },
      { // figure 1.24 (d)
        name: "Flowering Plant",
        axiom: "X",
        rules: {
          //"X": "F[+X]F[-X]+X",
          "X": "F[+XL]F[-XL]+X",  // added leaves
          "F": "FF"
        }
      },
      { // figure 1.24 (e)
        name: "Dense Flowering Plant",
        axiom: "X",
        rules: {
          //"X": "F[+X]F[-X]FX",
          "X": "F[+XL][-XL]FX",  // added leaves
          "F": "FF"
        }
      },
      { // figure 1.24 (f)
        name: "Arching Flowering Plant",
        axiom: "X",
        rules: {
          //"X": "F-[[X]+X]+F[+FX]-X",
          "X": "F-[[XL]+XL]+F[+FXL]-X",  // added leaves
          "F": "FF"
        }
      },
      { // figure 1.25
        name: "Bush",
        axiom: "A",
        rules: {
          "A": "[&FL!A]/////'[&FL!A]///////'[&FL!A]",
          "F": "S ///// F",
          "S": "FL",
          "L": "['''^^{-f+f+f-|-f+f+f}]"
          //"L": "['''^^{-F+F+F-|-F+F+F}]" 
        }
      },
    ],
    
    currentPreset: 0              // index of the preset currently in use (0 = High Slim Bush)
  },
  
  // Animation Parameters
  animation: {
    growthSpeed: 2,
    highlightDuration: 1500,
    growthDuration: 1000,
    leafStartTime: 2000,          // leaves appear after 2000ms (2s)
    flowerStartTime: 5000         // flowers appear 5000ms (5s)
  },
  // Enviroment Parameters
  environment: {
    autoCycle: true,             // automatic day/night cycle
    dayDuration: 120,            // lenght of a complete day in seconds (60s = 1 minut)
    plantBaseY: -8,              // generation height
    horizonY: 0.75,              // 0.75 = 75% from below
    dayTime: 0.0,                // state of the day, starting from dawn
    wasWateredToday: false,      // if the plant has been watered during the day
    isDead: false,               // if the plant is dead
  },
  // Weather Parameters
  weather: {
    isRaining: false,             // by default it doesn't rain
    showClouds: false,            // by default clouds are not visible
    showSunRays: false            // by default sunrays are not visible
  },
};
window.config = config;           // attaches the config object to the global window object, making it accessible from any script in the application  as window.config or simply config

// --- MAT4 UTILITY FOR COMPUTING GEOMETRIC TRANSFORMATIONS IN 3D SPACE ---
// includes all the operations needed to manipulate 4x4 transformation matrices
const Mat4 = {
  // Identity Matrix
  identity: () => {
    return new Float32Array([1,0,0,0,
                             0,1,0,0,
                             0,0,1,0,
                             0,0,0,1]);
  },
  
  multiply: (out, a, b) => {
    // Multiply two 4x4 matrices (a * b), result in out
    // essential for combining transformations like rotation + translation)
    // out = a * b 
    const aa = a, bb = b;                                               // assign input arrays to shorter variables for readability
    for (let i = 0; i < 4; ++i) {
      const ai0 = aa[i], ai1 = aa[i+4], ai2 = aa[i+8], ai3 = aa[i+12];  // extracts the values ​​of row i of the matrix a
      // calculate each element of column i of the out matrix
      out[i]      = ai0 * bb[0]  + ai1 * bb[1]  + ai2 * bb[2]  + ai3 * bb[3];   // row 0
      out[i+4]    = ai0 * bb[4]  + ai1 * bb[5]  + ai2 * bb[6]  + ai3 * bb[7];   // row 1
      out[i+8]    = ai0 * bb[8]  + ai1 * bb[9]  + ai2 * bb[10] + ai3 * bb[11];  // row 2
      out[i+12]   = ai0 * bb[12] + ai1 * bb[13] + ai2 * bb[14] + ai3 * bb[15];  // row 3
    }
    return out;
  },
  
  projection: (out, fovy, aspect, near, far) => {
    // Create the projection matrix responsible for the illusion of depth
    // objects that are further away appear smaller
    // fovy: vertical viewing angle (Field of View Y) in radians
    // aspect: ratio between the width and height of the window (aspect ratio)
    // near, far: clipping planes (anything closer than near or farther than far is clipped)
    const f = 1.0 / Math.tan(fovy / 2);      // scaling factor based on FOV
    const nf = 1 / (near - far);             // depth normalization factor
    out[0] = f / aspect;   out[1] = 0;          out[2] = 0;                          out[3] = 0;     // col 1
    out[4] = 0;            out[5] = f;          out[6] = 0;                          out[7] = 0;     // col 2
    out[8] = 0;            out[9] = 0;          out[10] = (far + near) * nf;         out[11] = -1;   // col 3
    out[12] = 0;           out[13] = 0;         out[14] = (2 * far * near) * nf;     out[15] = 0;    // col 4
    return out;
  },

  translate: (out, a, v) => {
    // Translate a 4x4 matrix (a) by a vector (v), result in out
    // calculate the new translation column (column 4 of the out matrix) by multiplying matrix a by vector v
    const x = v[0], y = v[1], z = v[2];     // extracts the components of the translation vector v
    out.set(a);                             // copies a in out
    out[12] = a[0] * x + a[4] * y + a[8] * z + a[12];
    out[13] = a[1] * x + a[5] * y + a[9] * z + a[13];
    out[14] = a[2] * x + a[6] * y + a[10] * z + a[14];
    out[15] = a[3] * x + a[7] * y + a[11] * z + a[15];
    return out;
  },

  // Rotation around the Y axis
  rotateY: (out, a, rad) => {
    // Multiply the matrix a by the rotation matrix r, result in out
    const s = Math.sin(rad), c = Math.cos(rad);
    const r = new Float32Array([
      c, 0, s, 0,
      0, 1, 0, 0,
     -s, 0, c, 0,
      0, 0, 0, 1
    ]);
    return Mat4.multiply(out, a, r);
  },

  // Rotation around the X axis
  rotateX: (out, a, rad) => {
    // Multiply the matrix a by the rotation matrix r, result in out
    const s = Math.sin(rad), c = Math.cos(rad);
    const r = new Float32Array([
      1, 0, 0, 0,
      0, c,-s, 0,
      0, s, c, 0,
      0, 0, 0, 1
    ]);
    return Mat4.multiply(out, a, r);
  },

  // Rotation around the Z axis
  rotateZ: (out, a, rad) => {
    // Multiply matrix a by the rotation matrix r, result in out
    const s = Math.sin(rad), c = Math.cos(rad);
    const r = new Float32Array([
       c,-s, 0, 0,
       s, c, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1
    ]);
    return Mat4.multiply(out, a, r);
  },

  // Scaling
  scale: (out, a, v) => {
    // Scale matrix a by the scale vector v, result in out
    // Multiply each row of the matrix by the corresponding scale factor
    const x = v[0], y = v[1], z = v[2];    // extracts the scale factors for each axis from the vector v
    out.set(a);                            // copies matrix a in out
    out[0] *= x;   out[1] *= x;   out[2] *= x;   out[3] *= x;
    out[4] *= y;   out[5] *= y;   out[6] *= y;   out[7] *= y;
    out[8] *= z;   out[9] *= z;   out[10] *= z;  out[11] *= z;
    return out;
  },

  // Clone a matrix
  clone: (a) => {
    return new Float32Array(a);
  },

  // Orthographic projection matrix
  ortho: (out, left, right, bottom, top, near, far) => {
    // Generate an orthographic projection matrix and stores it in the out array
    const lr = 1 / (left - right);   // reciprocal of the width of the viewing volume
    const bt = 1 / (bottom - top);   // reciprocal of the height of the viewing volume
    const nf = 1 / (near - far);     // reciprocal of the depth of the viewing volume
    out[0] = -2 * lr;                out[1] = 0;                      out[2] = 0;                    out[3] = 0;
    out[4] = 0;                      out[5] = -2 * bt;                out[6] = 0;                    out[7] = 0;
    out[8] = 0;                      out[9] = 0;                      out[10] = 2 * nf;              out[11] = 0;
    out[12] = (left + right) * lr;   out[13] = (top + bottom) * bt;   out[14] = (far + near) * nf;   out[15] = 1;
    return out;
  },

  // View Matrix
  lookAt: (out, eye, center, up) => {
    // simulates the inverse transformation of the observer
    // eye: position of the camera [x, y, z]
    // center: point towards which the observer is looking [x, y, z]
    // up: approximate "up" vector

    // Calculate forward vector f = normalize(center - eye) that goes from the camera to the target
    const [ex, ey, ez] = eye;
    const [cx, cy, cz] = center;
    const fx = cx - ex, fy = cy - ey, fz = cz - ez;
    let rlf = 1 / Math.hypot(fx, fy, fz);                // inverse lenght of f
    const nx = fx * rlf, ny = fy * rlf, nz = fz * rlf;   // normalized f vector (camera Z-axis)
    // Calculate side vector s = normalize(f × up) (camera X-axis)
    let sx = ny * up[2] - nz * up[1];
    let sy = nz * up[0] - nx * up[2];
    let sz = nx * up[1] - ny * up[0];
    let rls = 1 / Math.hypot(sx, sy, sz);                // inverse lenght of s
    sx *= rls; sy *= rls; sz *= rls;                     // normalized s vector
    // Calculate up vector u = s × f (camera Y-axis)
    const ux = sy * nz - sz * ny;
    const uy = sz * nx - sx * nz;
    const uz = sx * ny - sy * nx;
    // Construct the View Matrix (column-major order)
    out[0] = sx;                        out[1] = ux;                        out[2] = -nx;                      out[3] = 0;  // col 0 (camera X-axis)
    out[4] = sy;                        out[5] = uy;                        out[6] = -ny;                      out[7] = 0;  // col 1 (camera Y-axis)
    out[8] = sz;                        out[9] = uz;                        out[10]= -nz;                      out[11]= 0;  // col 2 (camera Z-axis)
    out[12] = -(sx*ex + sy*ey + sz*ez); out[13] = -(ux*ex + uy*ey + uz*ez); out[14] = (nx*ex + ny*ey + nz*ez); out[15] = 1; // col 3 (translation to bring the eye to the origin)

    return out
  }
};

// Normalize a 3D vector (makes it of length 1)
function normalizeVec3(v) {
  const len = Math.hypot(v[0], v[1], v[2]);  // calculates the length of the input vector v using Math.hypot(x, y, z), which computes sqrt(x*x + y*y + z*z)
  if (len === 0) return [0, 0, 0];
  return [v[0] / len, v[1] / len, v[2] / len];
}

// Orbital Camera (position and orientation)
class OrbitCamera {
  constructor() {
    this.theta = Math.PI/4;      // azimuth (angle of rotation around the Y-axis)
    this.phi = Math.PI/4;        // vertical elevation angle
    this.radius = 8;             // distance from target (zoom)
    this.target = [0,0,0];       // fixed point around which the rotation occurs (origin of the scene)
    this.lastX = 0;              // last recorded mouse X position
    this.lastY = 0;              // last recorded mouse Y position
    this.isOrbiting = false;     // flag that indicates whether the mouse is currently moving with the left button pressed
    this.isPanning = false;      // flag that indicates whether the mouse is currently moving with the right button pressed
    this.initListeners();        // calls the initListeners method (see line 254) to set up the mouse event listeners
  }

  // Handling of mouse input to change angles (drag) and mouse wheel to zoom
  initListeners() {
    const canvas = document.getElementById("glcanvas");
    
    canvas.addEventListener("contextmenu", e => e.preventDefault());  // prevent the context menu from appearing when right-clicking

    canvas.addEventListener("mousedown", e => {                       // mouse button press handling
      this.lastX = e.clientX;                                         // store the initial X position of the mouse
      this.lastY = e.clientY;                                         // store the initial Y position of the mouse
      if (e.button === 0) {                                           // left button
        this.isOrbiting = true;
      } 
      else if (e.button === 2) {                                     // right button
        this.isPanning = true;
      }
    });
    
    window.addEventListener("mouseup", () => {                      // left and/or right mouse buttons released
      this.isOrbiting = false;
      this.isPanning = false;
    });

    window.addEventListener("mousemove", e => {
      const dx = e.clientX - this.lastX;
      const dy = e.clientY - this.lastY;
      if (this.isOrbiting) {
        this.theta += dx * 0.005;                                                    // update the horizontal angle
        this.phi = Math.min(Math.max(0.1, this.phi + (dy * 0.005)), Math.PI - 0.1);  // update the vertical angle with limits to avoid tipping the camera
      }
      else if (this.isPanning) {
        // calculate the camera's local axes (right and up)
        const eye = this.getEyePosition();  // calls function getEyePosition (see line 382) to obtain the position of th camera
        const [ex, ey, ez] = eye;
        const [cx, cy, cz] = this.target;
        // Forward Vector (from camera to target)
        const fx = cx - ex, fy = cy - ey, fz = cz - ez;
        let rlf = 1 / Math.hypot(fx, fy, fz);
        const nx = fx * rlf, ny = fy * rlf, nz = fz * rlf;
        // Right Vector (vector product with the Y-axis of the world)
        const worldUp = [0,1,0];
        let sx = ny * worldUp[2] - nz * worldUp[1];
        let sy = nz * worldUp[0] - nx * worldUp[2];
        let sz = nx * worldUp[1] - ny * worldUp[0];
        let rls = 1 / Math.hypot(sx, sy, sz);
        sx *= rls; sy *= rls; sz *= rls;
        // Real Up Vector of the camera
        const ux = sy * nz - sz * ny;
        const uy = sz * nx - sx * nz;
        const uz = sx * ny - sy * nx;
        // Move the target along the axes
        const panSpeed = this.radius * 0.001;                // pan speed depends on zoom
        this.target[0] -= sx * dx * panSpeed;
        this.target[1] -= sy * dx * panSpeed;
        this.target[2] -= sz * dx * panSpeed;
        this.target[0] += ux * dy * panSpeed;
        this.target[1] += uy * dy * panSpeed;
        this.target[2] += uz * dy * panSpeed;
      }
      this.lastX = e.clientX;                                // update the last position X
      this.lastY = e.clientY;                                // update the last position Y
    });
    canvas.addEventListener("wheel", e => {
      e.preventDefault();                                    // prevents the page from scrolling
      this.radius *= Math.pow(0.95, e.deltaY * 0.01);        // update the radius (zoom) exponentially for a more natural effect
      this.radius = Math.min(Math.max(2, this.radius), 30);  // limits the zoom radius
    });
  }

  // Calculate the position of the camera
  getEyePosition() {
  // convert spherical coordinates (radius, phi, theta) to Cartesian coordinates (x, y, z) for the eye/camera position
    const x = this.radius * Math.sin(this.phi) * Math.sin(this.theta);
    const y = this.radius * Math.cos(this.phi);
    const z = this.radius * Math.sin(this.phi) * Math.cos(this.theta);
    return [x + this.target[0], y + this.target[1], z + this.target[2]];
  }

  // View Matrix
  getViewMatrix(out) {
    const eye = this.getEyePosition(); // calls the getEyePosition function (see line 382) to obtain the position of th camera
    // Cartesian position of the camera (eye) calculated with respect to the target
    Mat4.lookAt(out, eye, this.target, [0,1,0]);  // calls the lookAt function (see line 200) to obtain the View Matrix
    return out;
  }
}

// --- UTILITY TO MANAGE COMPILATION AND LINKING FOR THE SHADERS ---
function compileShader(gl, src, type) {
  // Takes three arguments:
  // - gl (the WebGL rendering context)
  // - src (a string containing the GLSL source code)
  // - type (a WebGL constant specifying the shader type, either gl.VERTEX_SHADER or gl.FRAGMENT_SHADER)
  const s = gl.createShader(type);                                         // calls the WebGL API to create an empty shader object of the specified type
  gl.shaderSource(s, src);                                                 // associates the GLSL source code string (src) with the newly created shader object (s)
  gl.compileShader(s);                                                     // instructs the GPU's driver to compile the source code that was just provided
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(s));
    gl.deleteShader(s);
    return null;
  }
  return s;
}

function createProgram(gl, vs, fs) {
  // Takes compiled vertex and fragment shaders and links them together into a single "shader program" that WebGL uses to render geometry
  const v = compileShader(gl, vs, gl.VERTEX_SHADER);    // calls the compileShader function (se line 400) to compile the vertex shader source code
  const f = compileShader(gl, fs, gl.FRAGMENT_SHADER);  // calls the compileShader function (se line 400) to compile the fragment shader source code
  const p = gl.createProgram();                         // calls the WebGL API to create an empty shader program object that will act as a container for the compiled shaders
  gl.attachShader(p, v);                                // attaches the compiled vertex shader (v) to the program object (p)
  gl.attachShader(p, f);                                // attaches the compiled fragment shader (f) to the program object (p)
  gl.linkProgram(p);                                    // instructs WebGL to link the attached shaders together
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(p));
    return null;
  }
  return p;
}

// --- SHADERS ---
// SIMPLE PHONG SHADER (no textures)
// Main shader pair for rendering most solid, untextured 3D objects in the scene, like the plant branches and the ground
// It implements a standard Blinn-Phong lighting model and includes shadow mapping calculations
const VSHADER = `#version 300 es                                      // specifies that the shader code is written in GLSL for OpenGL ES version 3.00, which is the standard for WebGL2
precision highp float;                                                // sets the default precision for floating-point numbers to highp (high precision)
layout(location=0) in vec3 a_position;                                // declares the input (attribute) for the vertex position bound to channel 0
layout(location=1) in vec3 a_normal;                                  // declares the input (attribute) for the vertex normal bound to channel 1

// definition of inputs (uniform) for transformation matrices (constant)
uniform mat4 u_model;                                                 // model matrix (transforms a vertex from its local model space to world space)
uniform mat4 u_view;                                                  // view matrix (transforms from world space to camera (view) space)
uniform mat4 u_proj;                                                  // projection matrix (transforms from view space to clip space (the final 2D projection))
uniform mat4 u_lightSpaceMatrix;                                      // light's combined view-projection matrix (from vertex positions to the light's coordinates for shadow mapping)

// definition of the variables (varying) that will be passed to the fragment shader after being interpolated
out vec3 v_normal;                                                    // normal vector
out vec3 v_worldPos;                                                  // vertex's position in world space
out vec4 v_lightSpacePos;                                             // vertex's position in the light's clip space

void main(){
  v_normal = mat3(transpose(inverse(u_model))) * a_normal;            // computes the normal matrix to correctly transform the normals
  v_worldPos = (u_model * vec4(a_position,1)).xyz;                    // computes the vertex position in world space (useful for lighting) by multiplying with the u_model matrix
  gl_Position = u_proj * u_view * vec4(v_worldPos,1);                 // performs the MVP transformation to compute the vertex position on the screen
  v_lightSpacePos = u_lightSpaceMatrix * vec4(v_worldPos, 1.0);       // transforms the position into light space
}
`;

const FSHADER = `#version 300 es                                            // specifies that the shader code is written in GLSL for OpenGL ES version 3.00, which is the standard for WebGL2
precision highp float;                                                      // sets the default precision for floating-point numbers to highp (high precision)

// receiving interpolated variables from the vertex shader
in vec3 v_normal;
in vec3 v_worldPos;
in vec4 v_lightSpacePos;

// uniforms for the light
uniform vec3 u_lightPosition;                                                // position
uniform vec3 u_lightColor;                                                   // color
uniform float u_lightIntensity;                                              // intensity

// uniform for the camera 
uniform vec3 u_viewPosition;                                                 // position

// uniforms for the material
uniform vec3 u_diffuseColor;
uniform vec3 u_specularColor;
uniform float u_shininess;
uniform float u_alpha;                                                       // trasparency

// uniform for shadow
uniform sampler2D u_shadowMap;                                               // used to access the depth texture generated during the shadow pass

out vec4 outColor;                                                           // defines the output color of the pixel

// Calculate shadow factor from 0 to 1
float calculateShadow(vec4 lightSpacePos) {
  vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;                     // converts the light-space position from homogeneous clip space to normalized device coordinates, [-1, 1]
  projCoords = projCoords * 0.5 + 0.5;                                       // from the [-1, 1] range to the [0, 1] range, required for sampling textures
  if (projCoords.z > 1.0) return 0.0;                                        // discard the fragments beyond the distant plane of light, those fragments are not in shadow
  float closestDepth = texture(u_shadowMap, projCoords.xy).r;                // samples the shadow map texture and retrieves the depth of the closest object from the light's point of view
  float currentDepth = projCoords.z;                                         
  float bias = 0.005;                                                        // bias to avoid shadow acne (where a surface incorrectly shadows itself due to precision errors)
  return currentDepth - bias > closestDepth ? 1.0 : 0.0;                     // if the current fragment is further away it is in shadow (return 1.0) otherwise it is lit (return 0.0)
}

void main(){
  vec3 n = normalize(v_normal);                                              // normalize the interpolated normal to ensure it is of length 1
  vec3 viewDir = normalize(u_viewPosition - v_worldPos);                     // calculates the direction vector from the fragment's position to the camera
  vec3 l = normalize(u_lightPosition - v_worldPos);                          // calculates the direction vector from the fragment's position to the light source
  vec3 h = normalize(l + viewDir);                                           // calculates the halfway vector between the light and view directions, used for Blinn-Phong specular lighting

  // Blinn-Phong lighting
  vec3 ambient = 0.1 * u_diffuseColor * u_lightIntensity;                    // ambient light component (constant)
  float diff = max(dot(n, l), 0.0);                                          // diffuse light factor, max ensures the light doesn't have a negative effect on surfaces facing away from it
  vec3 diffuse = diff * u_lightColor * u_diffuseColor * u_lightIntensity;    // diffuse light component
  float spec = pow(max(dot(n, h), 0.0), u_shininess);                        // specular light factor
  vec3 specular = spec * u_lightColor * u_specularColor * u_lightIntensity;  // specular light component

  // Calcolo ombra
  float shadow = calculateShadow(v_lightSpacePos);                           // calls calculateShadow (see line 487) to calculate the shadow factor
  
  vec3 finalColor = ambient + (1.0 - shadow) * (diffuse + specular);         // sums all component to obtain the final color
  outColor = vec4(finalColor, u_alpha);                                      // sets the final color of the pixel
}
`;

// SHADER FOR LUMINOUS OBJECTS
// A very simple shader pair for objects that are not affected by lighting (such as the sun and moon), they simply render with a solid, uniform color
const VSHADER_LUMINOUS = `#version 300 es             // specifies that the shader code is written in GLSL for OpenGL ES version 3.00, which is the standard for WebGL2
precision highp float;                                // sets the default precision for floating-point numbers to highp (high precision)
layout(location=0) in vec3 a_position;                // declares the input (attribute) for the vertex position bound to channel 0

uniform mat4 u_mvp;                                   // takes a single pre-combined Model-View-Projection matrix as input

void main(){
  gl_Position = u_mvp * vec4(a_position, 1.0);        // calculates the final position of the vertex on the screen
}
`;

const FSHADER_LUMINOUS = `#version 300 es             // specifies that the shader code is written in GLSL for OpenGL ES version 3.00, which is the standard for WebGL2
precision highp float;                                // sets the default precision for floating-point numbers to highp (high precision)
uniform vec3 u_color;                                 // uniform for the object's solid color
out vec4 outColor;

void main(){
  outColor = vec4(u_color, 1.0);                      // sets the final pixel color to the uniform color, with full opacity
}
`;

// SHADER FOR SUN RAYS
// This shader pair creates a post-processing effect for volumetric light scattering (sun rays)
// It renders a fullscreen triangle and calculates the color of each pixel based on its position relative to the sun's on-screen location
const VSHADER_SUNRAYS = `#version 300 es            // specifies that the shader code is written in GLSL for OpenGL ES version 3.00, which is the standard for WebGL2
precision highp float;                              // sets the default precision for floating-point numbers to highp (high precision)
layout(location=0) in vec3 a_position;              // declares the input (attribute) for the vertex position bound to channel 0
out vec2 v_uv;                                      // declares an output to pass UV coordinates (ranging from 0 to 1) to the fragment shader
void main() {
  v_uv = a_position.xy * 0.5 + 0.5;                 // converts the input vertex position (which is in clip space, [-1, 1]) to standard texture coordinates in the [0, 1] range
  gl_Position = vec4(a_position, 1.0);              // final position on screen
}
`;

const FSHADER_SUNRAYS = `#version 300 es                              // specifies that the shader code is written in GLSL for OpenGL ES version 3.00, which is the standard for WebGL2
precision highp float;                                                // sets the default precision for floating-point numbers to highp (high precision)
in vec2 v_uv;
out vec4 outColor;

uniform vec2  u_sunPos;                                               // uniform for the sun's position on the screen, in [0, 1] coordinates
uniform float u_time;                                                 // global time used for the animation
uniform vec3  u_color;
uniform float u_intensity;
uniform float u_aspect;                                               // width/height

void main(){
  vec2 p = v_uv - u_sunPos;                                           // calculates a vector from the sun's position to the current pixel's position
  p.x *= u_aspect;                                                    // corrects for the screen's aspect ratio to ensure the rays are circular rather than elliptical
  float r = length(p);                                                // calculates the radial distance of the pixel from the sun's center
  float a = atan(p.y, p.x);                                           // calculates the angle of the pixel relative to the sun's center

  // Creates a procedural pattern of radial bands using a cosine function based on the angle
  float band = 0.5 + 0.5 * cos(a * 22.0 + u_time * 0.6);              // multiplying a creates more bands, and adding u_time animates them to shimmer
  
  float blur = 0.22;
  float rays = smoothstep(0.5 - blur, 0.5 + blur, band);              // softens the edges of the bands, creating a blurred, ray-like appearance
  float falloff = exp(-r * 3.0);                                      // calculates a falloff factor that makes the rays fade out as their distance (r) from the sun increases
  float glow = exp(-r * 8.0);                                         // calculates a more concentrated falloff to create a central glow effect around the sun

  float alpha = (rays * falloff + glow * 0.6) * (u_intensity * 0.25); // combines the ray, falloff, and glow multiplied by the overall intensity to get the final transparency for the pixel

  outColor = vec4(u_color, alpha);                                    // sets the final color, using the calculated alpha for blending
}
`;

// SHADER FOR LUMINOUS PARTICLES (FIREFLIES AND STARS)
// This group of shaders is designed to render large numbers of luminous particles efficiently using gl.POINTS
// The vertex shaders handle animation and sizing, while a shared fragment shader draws a soft, circular shape for each particle
const VSHADER_FIREFLIES = `#version 300 es            // specifies that the shader code is written in GLSL for OpenGL ES version 3.00, which is the standard for WebGL2
precision highp float;                                // sets the default precision for floating-point numbers to highp (high precision)
layout(location=0) in vec3 a_position;                // declares the input (attribute) for the vertex position bound to channel 0

uniform mat4 u_mvp;                                   // takes a single pre-combined Model-View-Projection matrix as input
uniform float u_time;                                 // global time used for the animation

void main(){
  vec3 pos = a_position;

  // Animation: makes particles float up and down:
  // - for the Y position it uses a sine wave with he particle's original X position used to offset the phase, so they don't all move in sync
  // - for the X position it uses a cosine sine wave with he particle's original Y position used to offset the phase, so they don't all move in sync
  pos.y += sin(u_time * 0.5 + pos.x * 0.1) * 2.0;
  pos.x += cos(u_time * 0.3 + pos.y * 0.1) * 1.5;

  vec4 screenPos = u_mvp * vec4(pos, 1.0);

  gl_PointSize = 4.0;                                 // sets a fixed size for each firefly particle

  gl_Position = screenPos;
}
`;

const VSHADER_STARS = `#version 300 es                      // specifies that the shader code is written in GLSL for OpenGL ES version 3.00, which is the standard for WebGL2
precision highp float;                                      // sets the default precision for floating-point numbers to highp (high precision)
layout(location=0) in vec3 a_position;                      // declares the input (attribute) for the vertex position bound to channel 0

uniform mat4 u_mvp;                                         // takes a single pre-combined Model-View-Projection matrix as input

void main(){
  vec4 screenPos = u_mvp * vec4(a_position, 1.0);

  gl_PointSize = 1.0;                                       // ets a fixed size for each star particle
  
  gl_Position = screenPos;
}
`;

const FSHADER_LUMINOUS_PARTICLES = `#version 300 es      // specifies that the shader code is written in GLSL for OpenGL ES version 3.00, which is the standard for WebGL2
precision highp float;                                   // sets the default precision for floating-point numbers to highp (high precision)
uniform vec3 u_color;                                    // uniform for the object's solid color
out vec4 outColor;

void main(){
  float dist = distance(gl_PointCoord, vec2(0.5));       // calculates the distance from the center of the point
  float alpha = 1.0 - smoothstep(0.2, 0.5, dist);        // creates a soft circular shape by making the alpha value fall off smoothly near the edge of the point primitive
  if (alpha < 0.01) discard;                             // prevent the fragment from being written to the screen if its alpha is very low creating a clean circular edge

  vec3 glow = u_color * (1.0 / (dist * 6.0 + 1.0));      // creates a "glow" effect by making the color brighter near the center of the particle (where dist is small)
  outColor = vec4(glow, alpha);                          // sets the final color with the glow effect and the soft alpha falloff
}
`;

// SHADER FOR RAIN
// This shader pair is designed to render large numbers of particles efficiently using gl.POINTS
const VSHADER_RAIN = `#version 300 es                                              // specifies that the shader code is written in GLSL for OpenGL ES version 3.00, which is the standard for WebGL2
precision highp float;                                                             // sets the default precision for floating-point numbers to highp (high precision)
layout(location=0) in vec3 a_position;                                             // declares the input (attribute) for the vertex position bound to channel 0

uniform mat4 u_mvp;                                                                // takes a single pre-combined Model-View-Projection matrix as input
uniform mat4 u_proj;                                                               // projection matrix (transforms from view space to clip space (the final 2D projection))
uniform float u_time;                                                              // global time used for the animation

// Area in which the rain falls
uniform float u_rainTop;
uniform float u_rainBottom;
const float fallSpeed = 20.0;

void main(){
  vec3 pos = a_position;
  float range = (u_rainTop - u_rainBottom);
  
  // Animation of the rain drop's vertical position:
  // - moves downwards at a constant fallSpeed
  // - uses the modulo operator to wrap the particle back to the top (u_rainTop) once it falls below the bottom (u_rainBottom) creating a continuous loop
  pos.y = mod(pos.y - u_time * fallSpeed - u_rainBottom, range) + u_rainBottom;

  vec4 screenPos = u_mvp * vec4(pos, 1.0);

  float velocity = 15.0 / screenPos.w;                                            // calculates an apparent "speed" for the drop, proportional to how far it is from the camera

  // Create a motion-blur streak: b
  // by slightly shifting the vertex position downwards in screen space, it hints to the GPU that the primitive is a line, but it is still rendered as a point
  // the gl_PointSize is what gives it length
  screenPos.y -= velocity * 0.01; // 0.07 è un fattore per la lunghezza della striscia

  gl_PointSize = max(2.0, 7.0 / screenPos.w);                                     // sets the size of the rain particle. It becomes larger as it gets closer to the camera (screenPos.w is smaller)
  
  gl_Position = screenPos;
}
`;

const FSHADER_RAIN = `#version 300 es                   // specifies that the shader code is written in GLSL for OpenGL ES version 3.00, which is the standard for WebGL2
precision highp float;                                  // sets the default precision for floating-point numbers to highp (high precision)
uniform vec3 u_color;                                   // uniform for the object's solid color
out vec4 outColor;

void main(){
  float dist = distance(gl_PointCoord, vec2(0.5));      // calculates the distance from the center of the point
  float alpha = 1.0 - smoothstep(0.4, 0.5, dist);       // creates a soft circular shape by making the alpha value fall off smoothly near the edge of the point primitive

  if (alpha < 0.01) discard;                            // prevents the fragment from being written to the screen if its alpha is very low creating a clean circular edge

  outColor = vec4(u_color, alpha);
}
`;

// SHADER FOR CLOUDS
// This pair defines the shaders for rendering the procedural clouds
// The vertex shader is standard, simply transforming the position and passing the world position to the fragment shader
// The fragment shader is complex, it implements functions for:
// - random noise
// - smooth noise (Perlin-like)
// - fbm (Fractional Brownian Motion)
// to generate a multi-layered, animated fractal noise pattern, this noise value is then shaped with smoothstep to create the final transparent cloud shapes
const VSHADER_CLOUDS = `#version 300 es                               // specifies that the shader code is written in GLSL for OpenGL ES version 3.00, which is the standard for WebGL2
precision highp float;                                                // sets the default precision for floating-point numbers to highp (high precision)
layout(location=0) in vec3 a_position;                                // declares the input (attribute) for the vertex position bound to channel 0

uniform mat4 u_model;                                                 // model matrix (transforms a vertex from its local model space to world space)
uniform mat4 u_view;                                                  // view matrix (transforms from world space to camera (view) space)
uniform mat4 u_proj;                                                  // projection matrix (transforms from view space to clip space (the final 2D projection))

out vec3 v_worldPos;                                                  // vertex's position in world space

void main(){
  v_worldPos = (u_model * vec4(a_position, 1.0)).xyz;
  gl_Position = u_proj * u_view * u_model * vec4(a_position, 1.0);
}
`;

// Fragment Shader, genera noise frattale procedurale
const FSHADER_CLOUDS = `#version 300 es                                  // specifies that the shader code is written in GLSL for OpenGL ES version 3.00, which is the standard for WebGL2
precision highp float;                                                   // sets the default precision for floating-point numbers to highp (high precision)

uniform float u_time;                                                    // global time used for the animation

in vec3 v_worldPos;
out vec4 outColor;

// Generate a pseudo-random floating-point number between 0.0 and 1.0 based on a 2D input vector st. This is a common "hash function" used in procedural generation
float random(vec2 st) {
  return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);  // from http://lumina.sourceforge.net/Tutorials/Noise.html
}

// Classic Perlin/Value noise
// It takes a 2D coordinate st and generates a smooth continuous noise value
float noise(vec2 st) {
    vec2 i = floor(st);                                                  // gets the integer part of the input coordinate, which defines the corners of the grid cell this point is in
    vec2 f = fract(st);                                                  // gets the fractional part of the input coordinate, which represents the position within that grid cell

    //Get the four pseudo-random values for the four corners of the grid cell (a=bottom-left, b=bottom-right, c=top-left, d=top-right)
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    // Smooth bilinear interpolation
    vec2 u = f * f * (3.0 - 2.0 * f);                                    // calculates a smoot interpolation factor u from the fractional part f using the smoothstep formula 3t^2 - 2t^3
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.y * u.x;   // bilinear interpolation between the four corner values using the smoothed factor u
}

// Fractional Brownian Motion function
// This creates a more detailed and natural-looking fractal pattern by layering several "octaves" of the basic noise function at different frequencies and amplitudes
float fbm(vec2 st) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 0.0;

    for (int i = 0; i < 4; i++) {
        value += amplitude * noise(st);
        st *= 2.0;                                                      // double the frequency
        amplitude *= 0.5;                                               // half the amplitude
    }
    return value;
}

void main(){
    vec2 uv = v_worldPos.xz * 0.05;                                     // uses the X and Z components of the fragment's world position scaled down to make the clouds larger
    uv.x += u_time * 0.03;                                              // animates the clouds by scrolling the noise coordinates horizontally over time

    float noiseValue = fbm(uv);
    float cloudShape = smoothstep(0.4, 0.7, noiseValue);                // remaps the noise value: values < 0.4 -> 0 (transparent), values > 0.7 -> 1 (opaque), in between transition smoothly
    outColor = vec4(1.0, 1.0, 1.0, cloudShape);                         // white color with alpha based on the cloud shape
}
`;

// SHADER FOR OBJECTS WITH TEXTURE
const VSHADER_TEXTURED = `#version 300 es                          // specifies that the shader code is written in GLSL for OpenGL ES version 3.00, which is the standard for WebGL2
precision highp float;                                             // sets the default precision for floating-point numbers to highp (high precision)
layout(location=0) in vec3 a_position;                             // declares the input (attribute) for the vertex position bound to channel 0
layout(location=1) in vec3 a_normal;                               // declares the input (attribute) for the vertex normal bound to channel 1
layout(location=2) in vec2 a_uv;                                   // declares the input (attribute) for the uv cordinates bound to channel 2

uniform mat4 u_model;                                              // model matrix (transforms a vertex from its local model space to world space)
uniform mat4 u_view;                                               // view matrix (transforms from world space to camera (view) space)
uniform mat4 u_proj;                                               // projection matrix (transforms from view space to clip space (the final 2D projection))
uniform mat4 u_lightSpaceMatrix;

out vec3 v_normal;                                                 // normal vector
out vec3 v_worldPos;                                               // vertex's position in world space
out vec4 v_lightSpacePos;                                          // vertex's position in the light's clip space
out vec2 v_uv;                                                     // uv coordinates

void main(){
  v_worldPos = (u_model * vec4(a_position,1)).xyz;                 // computes the vertex position in world space (useful for lighting) by multiplying with the u_model matrix
  v_normal = mat3(transpose(inverse(u_model))) * a_normal;         // computes the normal matrix to correctly transform the normals
  gl_Position = u_proj * u_view * vec4(v_worldPos,1);              // performs the MVP transformation to compute the vertex position on the screen
  v_lightSpacePos = u_lightSpaceMatrix * vec4(v_worldPos, 1.0);    // transforms the position into light space
  v_uv = a_uv;
}
`;

const FSHADER_TEXTURED = `#version 300 es                                     // specifies that the shader code is written in GLSL for OpenGL ES version 3.00, which is the standard for WebGL2
precision highp float;                                                        // sets the default precision for floating-point numbers to highp (high precision)
in vec3 v_normal;
in vec3 v_worldPos;
in vec4 v_lightSpacePos;
in vec2 v_uv;

// uniforms for the light
uniform vec3 u_lightPosition;
uniform vec3 u_lightColor;
uniform float u_lightIntensity;

// uniforms for the camera
uniform vec3 u_viewPosition;

// uniforms for the material
uniform sampler2D u_diffuseTexture;
uniform vec3 u_specularColor;
uniform float u_shininess;
uniform float u_alpha;

// uniform for shadow
uniform sampler2D u_shadowMap;

out vec4 outColor;

// Calculate shadow factor from 0 to 1
float calculateShadow(vec4 lightSpacePos) {
  vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;                     // converts the light-space position from homogeneous clip space to normalized device coordinates, [-1, 1]
  projCoords = projCoords * 0.5 + 0.5;                                       // from the [-1, 1] range to the [0, 1] range, required for sampling textures
  if (projCoords.z > 1.0) return 0.0;                                        // discard the fragments beyond the distant plane of light, those fragments are not in shadow
  float closestDepth = texture(u_shadowMap, projCoords.xy).r;                // samples the shadow map texture and retrieves the depth of the closest object from the light's point of view
  float currentDepth = projCoords.z;                                         
  float bias = 0.005;                                                        // bias to avoid shadow acne (where a surface incorrectly shadows itself due to precision errors)
  return currentDepth - bias > closestDepth ? 1.0 : 0.0;                     // if the current fragment is further away it is in shadow (return 1.0) otherwise it is lit (return 0.0)
}

void main(){
  vec3 diffuseColor = texture(u_diffuseTexture, v_uv).rgb;                   // base color of the texture

  vec3 n = normalize(v_normal);                                              // normalize the interpolated normal to ensure it is of length 1
  vec3 viewDir = normalize(u_viewPosition - v_worldPos);                     // calculates the direction vector from the fragment's position to the camera
  vec3 l = normalize(u_lightPosition - v_worldPos);                          // calculates the direction vector from the fragment's position to the light source
  vec3 h = normalize(l + viewDir);                                           // calculates the halfway vector between the light and view directions, used for Blinn-Phong specular lighting

  // Blinn-Phong lighting
  vec3 ambient = 0.1 * diffuseColor * u_lightIntensity;                      // ambient light component (constant)
  float diff = max(dot(n, l), 0.0);                                          // diffuse light factor, max ensures the light doesn't have a negative effect on surfaces facing away from it
  vec3 diffuse = diff * u_lightColor * diffuseColor * u_lightIntensity;      // diffuse light component
  float spec = pow(max(dot(n, h), 0.0), u_shininess);                        // specular light factor
  vec3 specular = spec * u_lightColor * u_specularColor * u_lightIntensity;  // specular light component
  
  float shadow = calculateShadow(v_lightSpacePos);                           // calls calculateShadow (see line 829) to calculate the shadow factor
  
  vec3 finalColor = ambient + (1.0 - shadow) * (diffuse + specular);         // sums all component to obtain the final color
  outColor = vec4(finalColor, u_alpha);
}
`;

// SHADER FOR THE BACKGROUND
// This pair renders the sky
// The vertex shader draws a fullscreen triangle without needing any vertex buffers, by generating the vertices directly using gl_VertexID.
// The fragment shader takes the pixel's vertical position and uses mix and smoothstep to create a smooth 3-color gradient between the provided zenith, mid, and horizon colors
const VSHADER_BACKGROUND = `#version 300 es            // specifies that the shader code is written in GLSL for OpenGL ES version 3.00, which is the standard for WebGL2
precision highp float;                                 // sets the default precision for floating-point numbers to highp (high precision)

const vec2 positions[4] = vec2[](                                     // array of verteces of the fullscreen rectangle
  vec2(-1.0, -1.0),
  vec2( 1.0, -1.0),
  vec2(-1.0,  1.0),
  vec2( 1.0,  1.0)
);

out float v_y_pos;                                                    // passes the Y cordinate to the fragment shader

void main() {
  vec2 pos = positions[gl_VertexID % 4];                              // gl_VertexID gives the index of the current vertez
  gl_Position = vec4(pos, 0.0, 1.0);
  v_y_pos = pos.y;                                                    // pases the Y position (-1 at the bottom, +1 at the top) to the fragment shader
}
`;

// Fragment Shader (3-colors gradient)
const FSHADER_BACKGROUND = `#version 300 es            // specifies that the shader code is written in GLSL for OpenGL ES version 3.00, which is the standard for WebGL2
precision highp float;                                 // sets the default precision for floating-point numbers to highp (high precision)

uniform vec3 u_zenithColor;
uniform vec3 u_midColor;
uniform vec3 u_horizonColor;
uniform float u_horizonPoint;                          // position of the horizon (0.0 - 1.0)

in float v_y_pos;
out vec4 outColor;

void main() {
  float midPoint = u_horizonPoint;
  float transitionSize = 0.2;                                                                              // dimension of the gradient between colors
  float t = (v_y_pos + 1.0) / 2.0;                                                                         // maps vertical position from [-1, 1] to [0, 1]
  vec3 bottomGradient = mix(u_horizonColor, u_midColor, smoothstep(0.0, midPoint + transitionSize, t));    // interpolates between mid and horizon color
  vec3 topGradient = mix(u_midColor, u_zenithColor, smoothstep(midPoint - transitionSize, 1.0, t));        // interpolates between mid and zenith color
  // Blend the top and bottom gradients together in the middle region around the horizon point to create a seamless three-color gradient
  vec3 finalColor = mix(bottomGradient, topGradient, smoothstep(midPoint - (transitionSize/2.0), midPoint + (transitionSize/2.0), t));
  outColor = vec4(finalColor, 1.0);
}
`;

// SHADER FOR DEPTH PASS
const VSHADER_DEPTH = `#version 300 es                                     // specifies that the shader code is written in GLSL for OpenGL ES version 3.00, which is the standard for WebGL2
precision highp float;                                                     // sets the default precision for floating-point numbers to highp (high precision)
layout(location=0) in vec3 a_position;                                     // declares the input (attribute) for the vertex position bound to channel 0

uniform mat4 u_lightSpaceMatrix;                                           // light's combined view-projection matrix (from vertex positions to the light's coordinates for shadow mapping)
uniform mat4 u_model;                                                      // model matrix (transforms a vertex from its local model space to world space)

void main() {
  gl_Position = u_lightSpaceMatrix * u_model * vec4(a_position, 1.0);
}
`;

const FSHADER_DEPTH = `#version 300 es              // specifies that the shader code is written in GLSL for OpenGL ES version 3.00, which is the standard for WebGL2
precision highp float;                              // sets the default precision for floating-point numbers to highp (high precision)
void main() {
}
`;

// --- DRAWERS ---
// For particles rendering
class PrimitiveDrawer {
  constructor(gl, program) { 
    this.gl = gl;
    this.program = program;
    this.uniformLocations = {};
  }

  getUniformLocation(name) {
    if (this.uniformLocations[name] === undefined) {
      this.uniformLocations[name] = this.gl.getUniformLocation(this.program, name);
    }
    return this.uniformLocations[name];
  }

    // Sets up the geometry by uploading the mesh data to the GPU.
  createMesh(meshData) {
    const gl = this.gl;
    const vao = gl.createVertexArray();                                                      // Creates a Vertex Array Object (VAO) to store the mesh's state
    gl.bindVertexArray(vao);                                                                 // Binds the VAO, making it active

    // Positions Buffer (stores vertex coordinates)
    const posBuf = gl.createBuffer();                                                        // Creates the position buffer on the GPU
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);                                                  // Specifies that this buffer will contain vertex attribute data
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(meshData.positions), gl.STATIC_DRAW);    // Uploads the data from the CPU (JavaScript array) to the GPU
    gl.enableVertexAttribArray(0);                                                           // Enables the vertex attribute at location 0
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);                                     // Specifies how to read the data from the buffer (3 floats per vertex)

    // Normals Buffer
    if (meshData.normals) {
      const normBuf = gl.createBuffer();                                                     // Creates the normal buffer on the GPU
      gl.bindBuffer(gl.ARRAY_BUFFER, normBuf);                                               // Binds the buffer for normal data
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(meshData.normals), gl.STATIC_DRAW);    // Uploads the normal data from the CPU to the GPU
      gl.enableVertexAttribArray(1);                                                         // Enables the vertex attribute at location 1
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);                                   // Specifies how to read the data from the buffer (3 floats per normal)
    }

    // Indices Buffer
    if (meshData.indices) {
      const idxBuf = gl.createBuffer();                                                             // Creates the index buffer on the GPU
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);                                               // Specifies that this buffer will contain element indices
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(meshData.indices), gl.STATIC_DRAW);    // Uploads the index data from the CPU to the GPU
    }

    gl.bindVertexArray(null);                                                                       // Unbinds the VAO (it's now configured and ready to use)

    return {
      vao: vao,
      count: meshData.indices ? meshData.indices.length : meshData.positions.length / 3,
      hasIndices: !!meshData.indices
    };
  }

  // Disegno per frame
    // Draws the mesh for the current frame.
  draw(meshHandle, uniforms, mode) {
    const gl = this.gl;
    gl.useProgram(this.program);                  // Specifies which shader program to use for this draw call
    gl.bindVertexArray(meshHandle.vao);           // Re-binds the VAO (instantly restores all buffer-attribute links set up in createMesh)

    // Set all the uniforms we know about.
    for (const name in uniforms) {
      const location = this.getUniformLocation(name);
      if (location) {
        const value = uniforms[name];
        if (typeof value === 'number') gl.uniform1f(location, value);
        else if (value.length === 16) gl.uniformMatrix4fv(location, false, value);
        else if (value.length === 3) gl.uniform3fv(location, value);
      }
    }

    // Draws the geometry using the vertices in the active VAO, following the order specified in the Index Buffer
    if (meshHandle.hasIndices) {
      gl.drawElements(mode, meshHandle.count, gl.UNSIGNED_SHORT, 0);            // meshHandle.count = number of vertices to draw
    }
    else {
      gl.drawArrays(mode, 0, meshHandle.count);
    }
    gl.bindVertexArray(null);                                                    // Unbinds the VAO for cleanup
  }
}

// MeshDrawer (for mesh rendering)
class MeshDrawer {
  constructor(gl, program) {  // Initializes the MeshDrawer
    this.gl = gl;
    this.program = program;  // Stores the compiled shader program
    this.uniformLocations = {};
  }

  getUniformLocation(name) {
    if (this.uniformLocations[name] === undefined) {
      this.uniformLocations[name] = this.gl.getUniformLocation(this.program, name);
    }
    return this.uniformLocations[name];
  }

    // Geometry setup (uploads mesh data to the GPU)
  createMesh(meshData) {
    const gl = this.gl;
    const vao = gl.createVertexArray();                                                     // creates a Vertex Array Object (VAO) to store the mesh's state
    gl.bindVertexArray(vao);                                                                // binds (activates) the VAO

    // Positions Buffer (stores vertex coordinates)
    const posBuf = gl.createBuffer();                                                        // creates the position buffer on the GPU
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);                                                  // specifies that this buffer will contain vertex data
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(meshData.positions), gl.STATIC_DRAW);    // uploads the data from the CPU to the GPU
    gl.enableVertexAttribArray(0);                                                           // data in posBuf goes to attribute location 0
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);                                     // specifies how to read data from this buffer (3 floats per vertex)

    // Normals Buffer
    const normBuf = gl.createBuffer();                                                       // creates the normal buffer on the GPU
    gl.bindBuffer(gl.ARRAY_BUFFER, normBuf);                                                 // specifies that this buffer will contain normal data
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(meshData.normals), gl.STATIC_DRAW);      // uploads the data from the CPU to the GPU
    gl.enableVertexAttribArray(1);                                                           // data in normBuf goes to attribute location 1
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);                                     // specifies how to read data from this buffer (3 floats per normal)

    if (meshData.uvs) {
      // --- DEBUG LINE ---
      console.log("Creating UV buffer for the mesh");
      const uvBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(meshData.uvs), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 0, 0);
    }

    // Indices Buffer
    const idxBuf = gl.createBuffer();                                                             // creates the index buffer on the GPU
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);                                               // specifies that this buffer will contain indices
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(meshData.indices), gl.STATIC_DRAW);    // uploads the data from the CPU to the GPU

    gl.bindVertexArray(null);                                                                    //  unbinds the VAO (it is now configured and ready for use)

    return { vao, count: meshData.indices.length };                                              // returns a handle that represents the mesh, ready to be drawn
  }

  // Per-frame draw call
  draw(meshHandle, uniforms = {}) {
    const gl = this.gl;
    gl.useProgram(this.program);                                                                 // specifies which shader program to use for this draw call
    gl.bindVertexArray(meshHandle.vao);                                                          // re-binds the VAO (instantly restores all buffer-attribute links set up in createMesh)

    // Set all the known uniforms
    for (const name in uniforms) {
      const location = this.getUniformLocation(name);
      if (location) {
        const value = uniforms[name];
        if (typeof value === 'number') {
          // Checks if the uniform is a texture sampler
          if (name === 'u_shadowMap' || name === 'u_diffuseTexture') {
            gl.uniform1i(location, value); // Samplers are integers
          } else {
            gl.uniform1f(location, value); // Everything else (alpha, shininess, intensity) is a float
          }
        } 
        else if (value.length === 16) {
          gl.uniformMatrix4fv(location, false, value);
        } 
        else if (value.length === 3) {
          gl.uniform3fv(location, value);
        }
        else if (value.length === 2) {
          gl.uniform2fv(location, value);
        }

      }
    }

    // Draws the triangles using the vertices in the active VAO, following the order specified in the Index Buffer
    gl.drawElements(gl.TRIANGLES, meshHandle.count, gl.UNSIGNED_SHORT, 0);            // meshHandle.count = number of vertices to draw
    gl.bindVertexArray(null);                                                         // unbinds the VAO
  }
}

// Main (application entry point)
window.addEventListener("load", async () => {  // ensures the code runs only after the entire HTML page has loaded

  // INITIAL SETUP
  const canvas = document.getElementById("glcanvas");  // gets the reference to the canvas element
  const gl = canvas.getContext("webgl2");  // initializes WebGL with the webgl2 context
  if (!gl) {  // check for browser's WebGL2 support
    alert("WebGL2 is not available");
    return;
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

   runLSystemSymbolicTest();  // calls the runLSystemSymbolicTest function (see line 603) to perform a symbolic test

  // helper for window resizing
  function resize() {
    canvas.width = window.innerWidth;                                 // sets the canvas width
    canvas.height = window.innerHeight;                               // sets the canvas height
    gl.viewport(0,0,canvas.width, canvas.height);                     // tells WebGL how to map normalized coordinate space to the window
  }

  // Compilation
  const meshProgram = createProgram(gl, VSHADER, FSHADER);                                  // shader program
  const meshDrawer = new MeshDrawer(gl, meshProgram);                                       // creates a renderer instance
  const sunMoonProgram = createProgram(gl, VSHADER_LUMINOUS, FSHADER_LUMINOUS)              // shader program for sun and moon
  const sunMoonDrawer = new MeshDrawer(gl, sunMoonProgram);
  const sunRaysProgram = createProgram(gl, VSHADER_SUNRAYS, FSHADER_SUNRAYS);               // program for sun rays
  const sunRaysDrawer = new MeshDrawer(gl, sunRaysProgram);
  const depthProgram = createProgram(gl, VSHADER_DEPTH, FSHADER_DEPTH);                     // program for shadows
  const shadowDrawer = new MeshDrawer (gl, depthProgram)
  const texturedProgram = createProgram(gl, VSHADER_TEXTURED, FSHADER_TEXTURED);            // program for textured objects
  const texturedDrawer = new MeshDrawer(gl, texturedProgram);
  const fireflyProgram = createProgram(gl, VSHADER_FIREFLIES, FSHADER_LUMINOUS_PARTICLES);  // program for fireflies
  const fireflyDrawer = new PrimitiveDrawer(gl, fireflyProgram);
  const starProgram = createProgram(gl, VSHADER_STARS, FSHADER_LUMINOUS_PARTICLES);         // program for stars
  const starDrawer = new PrimitiveDrawer(gl, starProgram);
  const rainProgram = createProgram(gl, VSHADER_RAIN, FSHADER_RAIN);                        // program for rain
  const rainDrawer = new PrimitiveDrawer(gl, rainProgram);
  const cloudProgram = createProgram(gl, VSHADER_CLOUDS, FSHADER_CLOUDS);                   // program for clouds
  const cloudDrawer = new PrimitiveDrawer(gl, cloudProgram);
  const backgroundProgram = createProgram(gl, VSHADER_BACKGROUND, FSHADER_BACKGROUND);      // compilation of the background program
  const backgroundDrawer = {
    program: backgroundProgram,
    uniformLocations: {
      zenith: gl.getUniformLocation(backgroundProgram, "u_zenithColor"),
      mid: gl.getUniformLocation(backgroundProgram, "u_midColor"),
      horizon: gl.getUniformLocation(backgroundProgram, "u_horizonColor"),
      horizonPoint: gl.getUniformLocation(backgroundProgram, "u_horizonPoint")
    },
    draw: (zenithColor, midColor, horizonColor, horizonPoint) => {
      gl.useProgram(backgroundProgram);
      gl.uniform3fv(backgroundDrawer.uniformLocations.zenith, zenithColor);
      gl.uniform3fv(backgroundDrawer.uniformLocations.mid, midColor);
      gl.uniform3fv(backgroundDrawer.uniformLocations.horizon, horizonColor);
      gl.uniform1f(backgroundDrawer.uniformLocations.horizonPoint, horizonPoint);
      gl.disable(gl.DEPTH_TEST);                                  // disables depth testing so the background is always drawn behind everything else
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);                     // draws the rectangle
      gl.enable(gl.DEPTH_TEST);                                   // re-enables depth testing for the rest of the scene
    }
  };

  // Camera
  const camera = new OrbitCamera();  // creates an instance of the orbital camera

  window.addEventListener("resize", resize);  // adds a listener to call the resize function (see line 465) when the window size changes
  resize();

  // Setup of the Shadow Framebuffer
  const SHADOW_MAP_SIZE = 2048;                       // defines a constant for the resolution of the shadow map texture
  const shadowFramebuffer = gl.createFramebuffer();   // creates a new Framebuffer, a collection of attachments that can be used as a rendering target instead of the default screen
  gl.bindFramebuffer(gl.FRAMEBUFFER, shadowFramebuffer);  // binds the new FBO as the current rendering target, all subsequent rendering commands will now draw into it of the screen's canvas
  const shadowDepthTexture = gl.createTexture();          // this texture will be used as the depth attachment for the FBO and will store the depth information
  gl.bindTexture(gl.TEXTURE_2D, shadowDepthTexture);      // binds the new texture as the active 2D texture so that the following configuration commands will apply to it
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT32F, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 0, gl.DEPTH_COMPONENT, gl.FLOAT, null);  // allocates memory for the texture on the GPU
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);    // sets the texture's minification filter to NEAREST: the color is chosen by selecting the single nearest texel
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);    // sets the texture's magnification filter to NEAREST: the color is chosen by selecting the single nearest texel
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);  // sets the texture wrapping mode for the S (horizontal) coordinate to CLAMP_TO_EDGE
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);  // sets the texture wrapping mode for the T (vrtical) coordinate to CLAMP_TO_EDGE
  // Attach the configured shadowDepthTexture to the DEPTH_ATTACHMENT point of the currently bound framebuffer (shadowFramebuffer):
  // now, when rendering to this FBO, the GPU will write depth information into this texture
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, shadowDepthTexture, 0);
  gl.drawBuffers([gl.NONE]);  // explicitly tells WebGL that this framebuffer has no color buffer to draw to (we are only interested in depth)
  gl.readBuffer(gl.NONE);     // explicitly tells WebGL that there is no color buffer to read from with functions like gl.readPixels
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);  // unbinds the shadow framebuffer and re-binds the default framebuffer (the screen canvas) as the current rendering target

  // APPLICATION STATE
  // Scene management (handles the objects to be drawn)
  const sceneObjects = [];  // array that will contain all objects { meshHandle, modelMatrix, color } in the scene

  // Time variables
  let fpsTime = 0;    
  let frameCount = 0;
  let fps = 0;
  let prevDayTime = 0;

  // Animated growth
  let timeSinceLastGrowth = 0;
  let growthQueue = []
  let deadOverlay = null;
  
  // Day/night cycle
  const lerp = (a, b, t) => a.map((v, i) => v * (1 - t) + b[i] * t);     // helper function for linear interpolation
  let sunMoonMeshHandle = null;                                          // shared geometry for the sun and moon
  let sunRaysMeshHandle = null;                                          // sunrays geometry
  let defaultBranchMeshHandle = null;                                    // default branch geometry
  let groundMeshHandle = null;                                           // ground geometry
  let leafParts = [];
  let flowerGLTF = null;                                                 // GLTF scene (meshes + animator)
  let flowerParts = [];                                                  // parts of the flower [{ meshHandle, nodeIndex, material }]
  let fireflyMeshHandle = null;                                          // fireflies geometry
  let starMeshHandle = null;                                             // stars geometry
  let rainMeshHandle = null;                                             // rain geometry
  let cloudMeshHandle = null;                                            // clouds geometry

  // Flowers parameters: modify if the number of flowers slows the page down
  const PERF = {                 // performance settings object
    maxFlowers: 20,              // absolute cap on flowers
    flowerSpawnProbability: 1    // probability of accepting a generated flower
  };
  let flowerCount = 0;           // runtime counter for instantiated flowers

  // Delay + speed for flower animation
  // Note: flowerStartTime already exists and handles the "appearance": here we add an extra 2s before the animation
  const FLOWER_ANIM = {
    extraStartDelayMs: 2000,     // +2s after full appearance
    speed: 1.0,                  // speed multiplier (1 = real), can be modified as desired
    // "day" window: the animation plays forward; outside this window, it rewinds
    dayRange: [0.1, 0.5]         // 10% -> 50% of the day/night cycle
  };


  // Base material colors
    const groundMaterial = {
      diffuse: [0.3, 0.4, 0.2],
      specular: [0.1, 0.1, 0.1],
      shininess: 32.0
    };

    const branchMaterial = {
      specular: [0.2, 0.2, 0.2],
      shininess: 64.0
    };

    const leafMaterial = {
      specular: [0.1, 0.1, 0.1],
      shininess: 16.0
    };

  // USER INTERFACE
  uiInit();                                                              // initializes the UI

  // Callback to regenerate the plant
  ui.onRegenerate = () => {
    console.log("Regenerating plant and resetting growth...");

    timeSinceLastGrowth = 0;                                             // resets the timer for adding new parts
    growthQueue.length = 0;                                              // clears the queue of parts waiting to grow
  
    const now = performance.now();                                       // gets the current high-resolution timestamp
    for (const obj of sceneObjects) {                                    // loops through all objects already in the scene
      obj.birthTime = now;                                               // resets their birth time to the current moment
      obj._startReady = false;                                           // resets the flower animation flag
      obj.animPhase = undefined;                                         // resets the flower animation phase
      obj.animTime = 0;                                                  // resets the flower animation time
    }
  
    config.environment.isDead = false;                                   // marks the plant as alive again
    config.environment.wasWateredToday = false;                          // resets the daily watering status
    window._lastFrameTime = undefined;                                   // resets the global frame timer
  
    buildLSystem();                                                      // calls the buildLSystem function (see line 1881) to start building the new plant structure
  };

  // Callback for changing the preset
  ui.onPresetChange = () => {
      buildLSystem();                                                    // calls the buildLSystem function (see line 1881) to build a plant with the new preset
  };

  ui.onWater = () => {
    console.log("Watering the plant");
    config.environment.wasWateredToday = true;                           // sets the flag indicating the plant has been watered today
  };

  // FUNCTIONS - MAIN LOGIC
  function interpolateMultiStop(stops, t, useAngleLerp = false) {
    // stops: an array of keyframe objects like {time: ..., value: ...})
    // t: the current time, normalized between 0.0 and 1.0
    // useAngleLerp: boolean to toggle special angle interpolation logic
    if (t <= stops[0].time) { return stops[0].value; }                                    // returns the first value if t is at or before the start
    if (t >= stops[stops.length - 1].time) { return stops[stops.length - 1].value; }      // returns the last value if t is at or after the end

    for (let i = 0; i < stops.length; i++) {                                              // iterates through the stops to find the current segment
      const stopEnd = stops[i];                                                           // gets the potential end stop of the segment
      if (t < stopEnd.time) {                                                             // checks if the current time t is before this stop's time
        const stopStart = stops[i - 1];                                                   // gets the start stop of the current segment
        const segmentDuration = stopEnd.time - stopStart.time;                            // duration of the segment
        const timeInSegment = t - stopStart.time;                                         // time elapsed within the segment
        
        if (segmentDuration === 0) {                                                      // avoids division by zero if the segment has no duration
          return stopStart.value;                                                         // returns the start value to prevent errors
        }
        
        const localT = timeInSegment / segmentDuration;                                   // calculates the local interpolation factor (0 to 1)
        
        if (useAngleLerp) {                                                               // checks if angle-specific interpolation should be used
          const angle = lerpAngle(stopStart.value[0], stopEnd.value[0], localT);          // uses the lerpAngle function (see line 1319) ensuring values are numbers
          return [angle];                                                                 // returns the interpolated angle in an array
        }
        else {                                                                            // if not using angle interpolation
          return lerp(stopStart.value, stopEnd.value, localT);                            // uses the standard function for colors and positions
        }
      }
    }
    return stops[stops.length - 1].value;                                                 // returns the last value as a safe fallback
  }

  // Funzione di interpolazione degli angoli
  function lerpAngle(a0, a1, t) {
    const da = (a1 - a0) % (Math.PI * 2);                        // calculates the wrapped angle difference within a full circle
    const shortestAngle = 2 * da % (Math.PI * 2) - da;           // determines the shortest path between the two angles
    return a0 + shortestAngle * t;                               // returns the interpolated angle along the shortest path
  }

  // Geometry of a cylinder (for the branches)
  function createCylinder(radius, height, segments) {
    // segments: determines the roundness
    // initialize the empty arrays that will hold the geometry data
    const positions = [];                         // array for vertex coordinates
    const normals = [];                           // array for normals
    const indices = [];                           // array for indices
    // we consider the cylinder as a curved rectangle enclosed by two round discs, we will use n = segments to create it
    const halfHeight = height / 2;               // calculates the half height to center the cylinder on the origin (from -halfHeight to +halfHeight)
    const angleStep = (2 * Math.PI) / segments;  // calculates the angular increment for each segment by dividing a full circle by the required number of segments

    // vertices and normals
    for (let i = 0; i <= segments; i++) {        // the circle will be divided into n = segments angular slices
        const theta = i * angleStep;             // angle (theta) of the slice
        // current position along the circumference
        const x = Math.cos(theta);
        const z = Math.sin(theta);

        // two vertices for each segment (one above for the top disc and one below for the bottom disc)
        positions.push(radius * x, +halfHeight, radius * z);  // adds the top vertex (its normal points radially outward)
        normals.push(x, 0, z);
        positions.push(radius * x, -halfHeight, radius * z);  // adds the bottom vertex
        normals.push(x, 0, z);                                // the normal is the same as the top vertex for a smooth side surface
    }

    // indices
    for (let i = 0; i < segments; i++) {
        // calculate the indices of the four vertices that form the cylinder's rectangle
        const top1 = i * 2;                                  // current top vertex
        const bot1 = top1 + 1;                               // current bottom vertex
        const top2 = top1 + 2;                               // next top vertex
        const bot2 = top1 + 3;                               // next bottom vertex
        // approximate the rectangle with two triangles
        indices.push(top1, bot1, top2);                      // first triangle of the rectangle
        indices.push(top2, bot1, bot2);                      // second triangle of the rectangle
    }

    // center vertices for the discs (to which the edge vertices connect)
    const topCenterIndex = positions.length / 3;            // index of the new vertex to add for the top disc
    positions.push(0, +halfHeight, 0);                      // adds the new vertex to the center of the top disc
    normals.push(0, 1, 0);                                  // the normal of the top disc points straight up (positive Y-axis)
    const bottomCenterIndex = topCenterIndex + 1;           // index of the new vertex to add for the bottom disc
    positions.push(0, -halfHeight, 0);                      // adds the new vertex to the center of the bottom disc
    normals.push(0, -1, 0);                                 // the normal of the bottom disc points straight down (negative Y-axis)

    // indices for the discs
    // top disc
    for (let i = 0; i < segments; i++) {
        const top = i * 2;                                  // index of the current vertex on the edge 
        const nextTop = (i < segments - 1) ? top + 2 : 0;   // index of the next vertex on the edge
        // (i < segments - 1) ? top + 2 : 0 allows returning to the first vertex of the circumference when i is the last one (i == segments - 1), closing the circle
        indices.push(topCenterIndex, nextTop, top);         // creates the triangle between the center and the two vertices on the edge
    }
    // bottom disc
    for (let i = 0; i < segments; i++) {
        const bot = i * 2 + 1;                             // index of the current vertex on the edge 
        const nextBot = (i < segments - 1) ? bot + 2 : 1;  // index of the next vertex on the edge
        indices.push(bottomCenterIndex, bot, nextBot);     // the order (bot, nextBot) is reversed to maintain the correct face orientation (culling)
    }
    return { positions, normals, indices };
  }

  // Geometry of a plane (for the ground and the clouds)
  function createPlane(width, depth) {
    const w = width / 2;                                                  // calculates half the width for centering
    const d = depth / 2;                                                  // calculates half the depth for centering
    const positions = [ -w, 0, -d,   w, 0, -d,   w, 0, d,   -w, 0, d ];   // defines the four corner vertices of the plane on the XZ axis
    const normals   = [ 0, 1, 0,   0, 1, 0,   0, 1, 0,   0, 1, 0 ];       // defines the normal vector, pointing up (positive Y) for all vertices
    const indices   = [ 0, 1, 2,   0, 2, 3 ];                             // defines the two triangles that form the plane's rectangle
    return { positions, normals, indices };
  }

  // Geometry of a sphere (for sun and moon)
  function createSphere(radius, latitudeBands, longitudeBands) {
    // radius: size of the sphere
    // latitudeBands/longitudeBands: number of subdivisions for detail
    const positions = [];                                                 // array for vertex positions
    const normals = [];                                                   // array for vertex normals
    const indices = [];                                                   // array for vertex indices

    // iterates through latitude rings, from the south pole to the north pole
    for (let latNumber = 0; latNumber <= latitudeBands; latNumber++) {
        const theta = latNumber * Math.PI / latitudeBands;                // theta angle (vertical), from 0 to PI
        const sinTheta = Math.sin(theta);
        const cosTheta = Math.cos(theta);

        // iterates through longitude wedges, going around the sphere
        for (let longNumber = 0; longNumber <= longitudeBands; longNumber++) {
            const phi = longNumber * 2 * Math.PI / longitudeBands;       // phi angle (horizontal), from 0 to 2*PI
            const sinPhi = Math.sin(phi);
            const cosPhi = Math.cos(phi);

            // calculates the vertex coordinates using spherical coordinates
            const x = cosPhi * sinTheta;
            const y = cosTheta;
            const z = sinPhi * sinTheta;

            // the normal of a sphere centered at the origin is simply the normalized vector from its position
            // since we are building on a sphere of radius 1, the (x,y,z) coordinates are already the normal
            normals.push(x, y, z);                                       // adds the normal vector to the normals array
            // scales the positions by the final radius
            positions.push(radius * x, radius * y, radius * z);          // adds the scaled vertex position to the positions array
        }
    }

    // builds the indices to connect the vertices into triangles
    for (let latNumber = 0; latNumber < latitudeBands; latNumber++) {
        for (let longNumber = 0; longNumber < longitudeBands; longNumber++) {
            // indices of the four vertices that form a "quad" on the sphere's surface
            const first = (latNumber * (longitudeBands + 1)) + longNumber;     // index of the first vertex in the quad
            const second = first + longitudeBands + 1;                         // index of the vertex below the first one
            
            // creates two triangles to form the quad
            indices.push(first, second, first + 1);                            // defines the first triangle
            indices.push(second, second + 1, first + 1);                       // defines the second triangle
        }
    }

    return { positions, normals, indices };                                    // returns the complete geometry data
  }

  // Fullscreen triangle in clip-space (z=0), compatibile con MeshDrawer
  function createFullscreenTriangleGeo() {
    return {                                                                  // returns a geometry object
      positions: new Float32Array([                                           // defines the vertex positions as a Float32Array
        -1, -1,  0,                                                           // bottom-left vertex
         3, -1,  0,                                                           // far-right vertex
        -1,  3,  0                                                            // far-top vertex
      ]),
      normals: new Float32Array([                                             // defines the vertex normals as a Float32Array
         0, 0, 1,
         0, 0, 1,
         0, 0, 1 
      ]),
      indices: new Uint16Array([0,1,2])                                      // defines the single triangle connecting the vertices
    };
  }

  function createParticleCloud(count, boxSize, minY) {
    // count: number of particles to generate
    // boxSize: side length of the cube in which to distribute the particles
    // minY: minimum Y-level for the particles
    const positions = [];                                         // array for particle positions
    const halfBox = boxSize / 2;                                  // calculates half the box size for centering
    for (let i = 0; i < count; i++) {                             // loops to create the specified number of particles
      const x = Math.random() * boxSize - halfBox;                // generates a random x coordinate scaled by boxSize and shifted by halfBox to be within the desired range
      let y = Math.random() * boxSize - halfBox;                  // generates a random y coordinate within the box
      if (minY) {                                                 // checks if a minimum y-level was provided
        if (y < minY) y = minY + Math.random() * 4.0;             // recalculates y if it's below the minimum, adding a small random offset
      }
      const z = Math.random() * boxSize - halfBox;                // generates a random z coordinate within the box
      positions.push(x, y, z);                                    // adds the new particle's coordinates to the positions array
    }
    return { positions };                                         // returns the geometry data containing only the positions
  }

  // Function to create a WebGL texture from an image Blob
  async function glTFImageToTexture(gl, imageBlob) {
    if (!imageBlob) return null;                                 // returns null if the image blob is not provided
      const texture = gl.createTexture();                        // creates a new WebGL texture object
      gl.bindTexture(gl.TEXTURE_2D, texture);                    // binds the new texture to the 2D texture target

    try {
      const imageBitmap = await createImageBitmap(imageBlob);    // asynchronously creates an image bitmap from the blob (built-in browser function)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, imageBitmap);     // uploads the image bitmap data to the GPU texture
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);                 // sets the texture wrapping for the S (U) coordinate to clamp to the edge
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);                 // sets the texture wrapping for the T (V) coordinate to clamp to the edge
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);      // sets the minification filter to use linear interpolation on mipmaps
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);                    // sets the magnification filter to use linear interpolation
      gl.generateMipmap(gl.TEXTURE_2D);                                                     // generates a set of mipmaps for the texture
      console.log("Texture created successfully from GLB data");
    } 
    catch (e) {
      console.error("Error creating texture:", e);
      gl.deleteTexture(texture);
      return null;
    }
  
    gl.bindTexture(gl.TEXTURE_2D, null);                    // unbinds the texture from the target for cleanup
    return texture;                                         // returns the fully configured WebGL texture object
  }

  // initialization of all base geometries
  async function initBaseGeometries() {
    const loader = new GLTFLoader();                       // creates a new instance of the GLTF loader
    
    // Initialize the branch, if not already done
    if (!defaultBranchMeshHandle) {
      const defaultBranch = createCylinder(0.05, 1.0, 8);               // calls createCylinder (see line 1328) with radius 0.05, height 1.0, 8 segments
      defaultBranchMeshHandle = meshDrawer.createMesh(defaultBranch);   // calls createMesh (see line 1026) to upload the data to the GPU and saves the handle in the global variable
    }

    // Initialize sun/moon, if not already done
    if (!sunMoonMeshHandle) {
      const sphere = createSphere(0.5, 16, 16);                          // calls createSphere (see line 1400) creates the sphere geometry with radius 0.5, latitudeBands/longitudeBands 16
      sunMoonMeshHandle = sunMoonDrawer.createMesh(sphere);              // calls createMesh (see line 1026) to upload the data to the GPU and saves the handle in the global variable
    }

    // Initialize sunrays, if not already done
    if (!sunRaysMeshHandle) {
      const rayGeo = createFullscreenTriangleGeo();                      // calls createFullscreenTriangleGeo (see line 1449) to create the geometry for the sun rays
      sunRaysMeshHandle = sunRaysDrawer.createMesh(rayGeo);              // calls createMesh (see line 1026) to upload the mesh data to the GPU and saves the handle in the global variable
    }

    // Initialize fireflies, if not already done
    if (!fireflyMeshHandle) {
      const minY = config.environment.plantBaseY                         // sets the minimum y coordinate for the fireflies
      const fireflyGeo = createParticleCloud(50, 60, minY);              // calls createParticleCloud (see line 1465) to generate 50 fireflies across the entire ground geometry
      fireflyMeshHandle = fireflyDrawer.createMesh(fireflyGeo);          // calls createMesh (see line 944) to upload the mesh ata to the GPU and saves the handle in the global variable
    }

    // Initializes stars, if not already done
    if (!starMeshHandle) {
      const minY = null                                                  // stars have no minimum y coordinate
      const starGeo = createParticleCloud(10000, 400, minY);             // calls createParticleCloud (see line 1465) to generate 10000 stars in a huge box
      starMeshHandle = starDrawer.createMesh(starGeo);                   // calls createMesh (see line 944) to upload the mesh data to the GPU and saves the handle in the global variable
    }

    // Initializes rain, if not already done
    if (!rainMeshHandle) {
      const rainGeo = createParticleCloud(1500, 50);                    // calls createParticleCloud (see line 1465) to generate 500 raindrops in a 50x50x50 box
      rainMeshHandle = rainDrawer.createMesh(rainGeo);                  // calls createMesh (see line 944) to upload the mesh data to the GPU and saves the handle in the global variable
    }

    // Initializes clouds, if not already done
    if (!cloudMeshHandle) {
      const cloudGeo = createPlane(400, 400);                           // calls createPlane (see line 1390) to create a large plane for the clouds
      cloudMeshHandle = cloudDrawer.createMesh(cloudGeo);               // calls createMesh (see line 944) to upload the mesh data to the GPU and saves the handle in the global variable
    }

    // Initializes leaves, if not already done, by loading the .glb file
    if (leafParts.length === 0) {
      console.log("Loading leaf model...");
      try {
        const gltf = await loader.load('./Leaf.glb');                   // asynchronously loads the Leaf.glb file
        for (const meshPart of gltf.meshes) {                           // iterates through each mesh part in the loaded gltf file
          const textureHandle = await glTFImageToTexture(gl, meshPart.material.image); // calls glTFImageToTexture (see line 1484) to create a WebGL texture from the mesh's image data
          leafParts.push({                                              // adds a new leaf part object to the leafParts array
            handle: texturedDrawer.createMesh(meshPart),                // calls createMesh (see line 1026) to upload the mesh data to the GPU and stores its handle
            drawer: texturedDrawer,                                     // assigns the textured drawer for rendering
            texture: textureHandle,                                     // stores the texture handle
            material: leafMaterial,                                     // assigns the predefined leaf material
            localMatrix: meshPart.worldMatrix                           // stores the local transformation matrix of the mesh part
          });
        }
      }
      catch (e) {
        console.error("Failed to load Leaf.glb:", e);
      }
    }

    // Initializes flowers, if not already done, by loading the .glb file (asynchronously)
    if (flowerParts.length === 0) {
      console.log("Loading flower model...");
      try {
        const gltf = await loader.load('./Flower.glb');                 // asynchronously loads the Flower.glb file
        flowerGLTF = gltf;                                              // stores the loaded gltf data globally
        for (const meshPart of gltf.meshes) {                           // iterates through each mesh part in the loaded gltf file
          const hasTexture = meshPart.material && meshPart.material.image;   // checks if the mesh part has a texture
          let drawer = hasTexture ? texturedDrawer : meshDrawer;             // selects the appropriate drawer based on texture presence
          const handle = drawer.createMesh(meshPart);                        // calls createMesh (see line 1026) to upload the mesh data to the GPU and stores its handle
          const textureHandle = hasTexture ? await glTFImageToTexture(gl, meshPart.material.image) : null;   // creates a texture if available
      
          flowerParts.push({                                           // adds a new flower part object to the flowerParts array
            handle: handle,                                            // stores the mesh handle
            drawer: drawer,                                            // stores the assigned drawer
            texture: textureHandle,                                    // stores the texture handle
            material: meshPart.material,                               // stores the material data from the gltf
            localMatrix: meshPart.worldMatrix,                         // stores the local transformation matrix
            nodeIndex: meshPart.nodeIndex                              // stores the node index for animation
          });
        }
        console.log(`Flower model loaded successfully with ${flowerParts.length} parts, animation ready`);
      }
      catch (e) {
        console.error("Failed to load Red Flower Animated.glb:", e);
      }
    }
  }
  
    // adds a new part to the scene
  function addPlantPart(objDesc, calculatedMatrix) {

    if (objDesc.type === 'flower') {
      if (typeof PERF !== 'undefined') {            // checks if the performance settings object is defined
        if (typeof flowerCount === 'number') {      // checks if the flower counter is a number
          if (flowerCount >= (PERF.maxFlowers ?? Infinity)) return;          // returns if the maximum number of flowers has been reached
          if (Math.random() > (PERF.flowerSpawnProbability ?? 1)) return;    // returns based on the spawn probability
          flowerCount++;
          console.log(`Currently ${flowerCount} flowers in the scene`);      // debug print
        }
      }
    }

    // Flowers
    if (objDesc.type === 'flower') {
      if (!Array.isArray(flowerParts) || flowerParts.length === 0) return;   // returns if the flower parts asset is not ready
    
      sceneObjects.push({                                                    // adds a new flower object to the scene
        meshHandle: null,                                                    // not used: drawn per-part at runtime
        modelMatrix: calculatedMatrix,                                       // the base transformation for the entire flower
        color: [1, 1, 1],                                                    // a neutral color as parts have their own materials
        birthTime: performance.now(),                                        // sets the creation timestamp
        description: objDesc,                                                // stores the original L-System description
        drawer: null,                                                        // not used directly
        texture: null,                                                       // not used directly
        material: null,                                                      // not used directly
        meshParts: flowerParts,                                              // packet of all flower parts
    
        // Animation state (synchronized for all flowers)
        animPhase: "closed",                                                 // "closed" , "opening", "open" , "closing"
        animTime: 0,                                                         // [0...animLen]
        _startReady: false,                                                  // becomes true after the baseStart delay
        _lastIsDay: null                                                     // last seen day/night state
      });
    
      return;
    }

    // Branches and Leaves
    let partsToAdd = [];
    let baseColor = objDesc.color;                                           // gets the base color from the description

    switch (objDesc.type) {
      case 'leaf':
        partsToAdd = leafParts;                                              // parts are already loaded from the leaf glb
        break;
      case 'branch':
      default:
        partsToAdd.push({                                                    // adds a single default part for branches
          handle: defaultBranchMeshHandle,
          drawer: meshDrawer,
          texture: null,
          material: branchMaterial,
          localMatrix: Mat4.identity()
        });
        break;
    }

    if (partsToAdd.length === 0 || !partsToAdd[0].handle) return;           // returns if no valid parts are ready

    for (const part of partsToAdd) {
      const finalPartMatrix = Mat4.multiply(Mat4.identity(), calculatedMatrix, part.localMatrix);    // combines the base matrix with the part's local matrix
      const finalColor = baseColor                                                                   // determines the final color
        || (part.material && part.material.baseColorFactor
              ? part.material.baseColorFactor.slice(0, 3)                                            // slice(0, 3) takes only the first 3 components (R, G, B) and discards any alpha
              : [0.29, 0.61, 0.02]);                                                                 // fallback color if none is defined

      sceneObjects.push({                                                                            // adds the final object to the scene
        meshHandle: part.handle,
        modelMatrix: finalPartMatrix,
        color: finalColor,
        birthTime: performance.now(),
        description: objDesc,
        drawer: part.drawer,
        texture: part.texture,
        material: part.material,
        meshParts: null                                                                              // branches/leaves do NOT have a packet of parts
      });
    }
  }

  function expandLSystem(axiom, rules, iterations) {
    let currentString = axiom;                                                      // initializes the current string with the axiom, it will be updated in each iteration
    // Regex to find modules: 
    // it looks for an uppercase letter ([A-Z]), optionally followed by parentheses containing parameters
    const moduleRegex = /([A-Z])(?:\(([^)]*)\))?/g;                                 // the g flag ensures it finds all matches, not just the first one

    for (let i = 0; i < iterations; i++) {
      let nextString = "";                                                          // initializes an empty string for the next iteration's result
      let lastIndex = 0;                                                            // keeps track of the end of the last processed match

      for (const match of currentString.matchAll(moduleRegex)) {                    // iterates over all modules found in the current string
        // adds all non-module symbols ([, +, ', etc.) that are between the last match and this one
        nextString += currentString.substring(lastIndex, match.index);
        const symbol = match[1];                                                    // extracts the symbol character (e.g., 'F')
        const params = match[2] ? match[2].split(',').map(Number) : [];             // if parameters exist it splits the string by commas and converts each part to a number
        let replacement = match[0];                                                 // sets the default replacement to be the original matched string itself

        if (rules[symbol]) {                                                        // checks if a rule exists for the current symbol
          let rule = rules[symbol];
          if (Array.isArray(rule)) {                                                // handles stochastic rules (if the rule is an array of choices)
            let random = Math.random();                                             // generates a random number between 0.0 and 1.0
            let threshold = 0;                                                      // initializes a probability threshold
            for (const r of rule) {                                                 // iterates through the possible rule choices
              threshold += r.prob;                                                  // adds the probability of the current choice to the threshold
              if (random < threshold) {                                             // checks if the random number falls within this choice's probability range
                rule = r.rule;                                                      // selects this rule
                break;
              }
            }
          }
          if (typeof rule === 'function') {                                         // handles parametric rules (if the rule is a function)
            replacement = rule(...params);                                          // calls the function with the parameters to get the replacement string
          } 
          else {                                                                    // if the rule is a simple string
            replacement = rule;                                                     // uses the string as the replacement
          }
        }
        nextString += replacement;                                                 // appends the replacement string to the result
        lastIndex = match.index + match[0].length;                                 // updates lastIndex to the position right after the currently processed module
      }
      nextString += currentString.substring(lastIndex);                            // adds any remaining characters from the end of the original string that came after the last module
      currentString = nextString;                                                  // sets the result of this iteration as the new current string
    }
    return currentString;
  }

  function interpretLSystem(lString, angleDegrees = 25.7) {
    // This function acts as a "turtle graphics" interpreter:
    // it takes the final expanded L-system string and translates its sequence of symbols into a list of geometric objects (branches, leaves, flowers),
    // each with a calculated world transformation matrix
    // It simulates a "turtle" moving in 3D space, drawing parts and changing its orientation based on the symbols in the string
    const result = [];
    const stack = [];                                                                         // it will be used to save and restore the turtle's state (transformation matrix)
    let currentMatrix  = Mat4.identity();                                                     // initializes the current transformation matrix

    Mat4.translate(currentMatrix , currentMatrix , [0, config.environment.plantBaseY, 0]);    // translates the starting matrix to the plant's base height

    const angle = angleDegrees * (Math.PI / 180);                                             // converts the angle from degrees to radians (Mat4 rotation functions expect radians)

    for (let i = 0; i < lString.length; i++) {
      const symbol = lString[i];                                                              // gets the current character
      let params = [];

      // parameter handling: if the symbol is an uppercase letter and the next character is '(' then read the parameters
      if (symbol >= 'A' && symbol <= 'Z' && i + 1 < lString.length && lString[i+1] === '(') {
        const closingParenIndex = lString.indexOf(')', i);                                    // finds the index of the closing parenthesis
        if (closingParenIndex !== -1) {                                                       // checks if a closing parenthesis was found
          const paramString = lString.substring(i + 2, closingParenIndex);                    // extracts the parameter string (the part between the parentheses)
          if (paramString) params = paramString.split(',').map(Number);                       // parses the parameters into numbers if the string is not empty
          i = closingParenIndex;                                                              // skips the index to the end of the parameters
        }
      }

      switch (symbol) {
        case 'F': {                                                                                  // branch
          const segmentLength = 0.5;
          const segmentWidth = 1.0;
          let localTransform = Mat4.identity();                                                      // creates a local transformation matrix
          Mat4.translate(localTransform, localTransform, [0, segmentLength / 2, 0]);                 // translates the cylinder to its center
          Mat4.scale(localTransform, localTransform, [segmentWidth, segmentLength, segmentWidth]);   // scales the cylinder
          const worldMatrix = Mat4.multiply(Mat4.identity(), currentMatrix , localTransform);        // transforms it into world space
          result.push({                                                                              // adds the new branch to the results
            symbol: 'F',
            matrix: worldMatrix
          });
          Mat4.translate(currentMatrix , currentMatrix , [0, segmentLength, 0]);                     // updates the turtle's position
          break;
        }
        case 'L': {                                                                                  // leaf
          const leafScale = 0.2;
          let leafMatrix = Mat4.clone(currentMatrix);
        
          // move the leaf back by the actual length of the segment
          Mat4.translate(leafMatrix, leafMatrix, [0, -0.7, 0]);
          Mat4.scale(leafMatrix, leafMatrix, [leafScale, leafScale, leafScale]);                    // applies the scale to the leaf
        
          const minAngle = -30 * (Math.PI / 180);                                                   // defines the minimum pitch angle
          const maxAngle = 30 * (Math.PI / 180);                                                    // defines the maximum pitch angle
          const randomRotz = Math.random() * (maxAngle - minAngle) + minAngle;                       // calculates a random pitch
          const randomRotY = Math.random() * Math.PI * 2;                                          // calculates a random yaw
        
          Mat4.rotateY(leafMatrix, leafMatrix, randomRotY);                                        // rotates the leaf around its "stem"
          Mat4.rotateZ(leafMatrix, leafMatrix, randomRotz);                                          // applies the pitch (tilt)
        
          result.push({                                                                             // adds the new leaf to the results
            symbol: 'L',
            matrix: leafMatrix,
          });
          break;
        }
        case 'X': {                                                                                 // flower
          const flowerScale = 0.1;
          const flowerOffsetY = 0;                                                                  // defines the vertical offset
          let flowerMatrix = Mat4.clone(currentMatrix);                                             // clones the current matrix for the flower
          Mat4.translate(flowerMatrix, flowerMatrix, [0, flowerOffsetY, 0]);                        // applies the vertical offset
          Mat4.scale(flowerMatrix, flowerMatrix, [flowerScale, flowerScale, flowerScale])           // applies the scale
          result.push({                                                                             // adds the new flower to the results
            symbol: 'X',
            matrix: flowerMatrix,
          });
          break;
        }
        
        // Turn Left/Right
        case '+': Mat4.rotateZ(currentMatrix , currentMatrix ,  angle); break;
        case '-': Mat4.rotateZ(currentMatrix , currentMatrix , -angle); break;
        
        // Turn Down/Up
        case '&': Mat4.rotateX(currentMatrix , currentMatrix ,  angle); break;
        case '^': Mat4.rotateX(currentMatrix , currentMatrix , -angle); break;
        
        // Tilt Left/Right 
        case '\\': Mat4.rotateY(currentMatrix , currentMatrix ,  angle); break;
        case '/': Mat4.rotateY(currentMatrix , currentMatrix , -angle); break;
        
        // Turn Around
        case '|': Mat4.rotateZ(currentMatrix , currentMatrix , Math.PI); break;
        
        // Stack handling
        case '[': 
          stack.push(Mat4.clone(currentMatrix));                                                       // pushes a copy of the current matrix onto the stack
          break;
        case ']': 
          if (stack.length > 0) {                                                                      // checks if the stack is not empty
            currentMatrix = stack.pop();                                                               // pops the last matrix from the stack, restoring the previous state
          }
          break;
        
        default:
          // ignores unrecognized symbols
          break;
      }
    }
    return result;                                                                                     // returns the final list of transformations
  }

  function runLSystemSymbolicTest() {
    // Debugging function: tests the core string expansion logic of the L-system engine (expandLSystem) with a simple pattern and prints the results of each iteration to the console
    console.log("---- L-System Symbolic Test ----");                                                   // header for the test
    const testAxiom = 'F';                                                                             // starting axiom for the test
    const testRules = { 'F': 'F+F-F' };                                                                // set of rules for the test
    const testIterations = 3;                                                                          // number of iterations for the test
    let currentString = testAxiom;                                                                     // initializes the current string with the axiom
    console.log(`Iteration 0: ${currentString}`);                                                      // logs the initial state (iteration 0)
    for (let i = 1; i <= testIterations; i++) {                                                        // loops for each iteration to show the evolution
      currentString = expandLSystem(currentString, testRules, 1);                                      // calls the expandLSystem function (see line 1689) with a single iteration
      console.log(`Iteration ${i}: (length ${currentString.length})`);                                 // logs the current iteration number and the string length
      console.log(currentString.substring(0, 100) + (currentString.length > 100 ? '...' : ''));        // prints the first 100 characters of the string
    }
    console.log("---- End of Test ----");                                                              // footer for the test
  }

  function LSysToSceneDesc(pureList) {
    // Transforms the raw output of the L-system interpreter to a semantic scene description:
    // interpretLSystem produces a list of objects with symbols ('F', 'L', 'Z') and matrices,
    // this function maps those symbols to meaningful object types ('branch', 'leaf', 'flower') and assigns default properties
  
    return pureList.map(item  => {                                                         // iterates over each item in the pure list
      switch(item.symbol) {                                                                // checks the symbol of the current item
        case 'F':
          return { type: 'branch', color: [0.29, 0.61, 0.02], modelMatrix: item.matrix };  // branch description object
        case 'L':
          return { type: 'leaf', color: [1, 1, 1], modelMatrix: item.matrix };             // neutral color for textured leaves
        case 'X':
          return { type: 'flower', color: null, modelMatrix: item.matrix };                // the color is derived from the material
        default:                                                                           // for any other symbol
          return null;
      }
    }).filter(Boolean);                                                                    // removes any null elements from the final array
  }

  // L-System scene construction
  function buildLSystem() {
    flowerCount = 0;                                                               // resets the cap for the new scene
  
    const preset = config.lSystem.presets[config.lSystem.currentPreset];           // gets the currently selected preset
    console.log(`Using preset: "${preset.name}"`);

    const iterCount = config.lSystem.iterations;                                   // gets the number of iterations
    const angleDegrees = config.lSystem.angle;                                     // gets the angle in degrees

    const expandedString = expandLSystem(preset.axiom, preset.rules, iterCount);   // calls expandLSystem (see line 1689)
    const lSystemOutput = interpretLSystem(expandedString, angleDegrees);          // calls interpretLSystem (see line 1735) to get geometry data
    const sceneDescription = LSysToSceneDesc(lSystemOutput);                       // calls LSysToSceneDesc (see line 1861)
    buildScene(sceneDescription); // builds the scene from the description
  }

  function buildScene(description) {
    // Constructs the scene from its description, preparing it for rendering
    sceneObjects.length = 0;                                                                // clears the scene
    growthQueue = [];                                                                       // clears the previous queue

    // add the ground
    if (!sceneObjects.some(obj => obj.description.type === 'ground')) {                     // checks if the ground is already in the scene
      if (!groundMeshHandle) {                                                              // creates the ground geometry only once
        const groundGeo = createPlane(60, 60);                                              // a 60x60 unit plane
        groundMeshHandle = meshDrawer.createMesh(groundGeo);
      }
      let groundMatrix = Mat4.identity();
      Mat4.translate(groundMatrix, groundMatrix, [0, config.environment.plantBaseY, 0]);    // moves the ground to where the plant starts
      sceneObjects.push({                                                                   // adds the ground to the scene
          meshHandle: groundMeshHandle,
          modelMatrix: groundMatrix,
          color: [0.3, 0.4, 0.2],                                                           // dark green
          birthTime: performance.now(),
          scale: [1,1,1],
          description: {
            type: 'ground',
            transforms: [{type: 'translate', value: [0, config.environment.plantBaseY, 0]}]
          }
      });
    }

    // iterates over each object described in the data structure
    for (const objDesc of description) {
      if (objDesc.type === 'branch' || objDesc.type === 'leaf' || objDesc.type === 'flower') { // checks if the object is a plant part
        let modelMatrix;
        if (objDesc.modelMatrix) {                // checks if the matrix is already calculated (from the L-System)
          modelMatrix = objDesc.modelMatrix;
        } 
        else {                                    // otherwise, calculates it from the list of transforms (for loading from a file)
          modelMatrix = Mat4.identity();
          for (const t of objDesc.transforms) {
            if (t.type === 'translate') Mat4.translate(modelMatrix, modelMatrix, t.value);
            else if (t.type === 'rotateX') Mat4.rotateX(modelMatrix, modelMatrix, t.value);
            else if (t.type === 'rotateY') Mat4.rotateY(modelMatrix, modelMatrix, t.value);
            else if (t.type === 'rotateZ') Mat4.rotateZ(modelMatrix, modelMatrix, t.value);
            else if (t.type === 'scale') Mat4.scale(modelMatrix, modelMatrix, t.value);
          }
        }
        growthQueue.push({ objDesc, modelMatrix });  // adds the parts to the queue, not to the scene
      }
    }
  }

  function handlePlantDeath() { 
    // This function is triggered when the plant "dies" (i.e., is not watered for a full day cycle)
    // It manages the visual feedback for this event
    console.log("Plant has died!");
  
    deadOverlay = [0.4, 0.25, 0.1, 0.6];  // semi-transparent brown overlay
    flowersFrozen = true;                 // freezes the flower animation  (bypasses animation updates)
  
    // displays a message on the screen
    let msg = document.getElementById("deathMsg");                                      // gets the message element
    if (!msg) {
      msg = document.createElement("div");                                              // creates a new div element
      msg.id = "deathMsg";                                                              // sets the id of the message
      msg.textContent = "Your plant should be watered at least once a day! Try again";  // sets the message text
      msg.style.position = "absolute";
      msg.style.top = "50%";
      msg.style.left = "50%";
      msg.style.transform = "translate(-50%, -50%)";                                    // fine-tunes the centering
      msg.style.color = "red";
      msg.style.background = "rgba(0,0,0,0.7)";
      msg.style.padding = "10px 20px";
      msg.style.borderRadius = "8px";                                                    // rounds the corners
      msg.style.fontSize = "18px";
      document.body.appendChild(msg);                                                    // appends the message to the document body
    }
  
    // reset the plant after a short delay
    setTimeout(() => {                                 // sets a timer
      if (msg) msg.remove();                           // removes the message if it exists
      config.environment.isDead = false;               // marks the plant as not dead
      ui.onRegenerate();                               // callls ui.onRegenerate (see line 1256) to generate a new plant
      flowersFrozen = false; // unfreezes the flower animation
      deadOverlay = null; // removes the dead overlay color
    }, 4000); // 4 seconds of pause before the reset
  }

  // initial scene
  buildLSystem()  // calls the buildLSystem function (see line 671)
  
  // function to update the object counter in the UI
  function updateObjCount() {
    document.getElementById("objCount").textContent = sceneObjects.length;
  }

  // Rendering loop
  function render(currentTime) {
    // This function is the main render loop of the application, it is called repeatedly by the browser via requestAnimationFrame
    // In each call it performs all the necessary steps to update the application's state and draw a single frame to the screen:
    // - updating animations
    // - handling the day/night cycle
    // - processing the plant's growth
    // - executing the multi-pass rendering pipeline (shadow pass, main pass)

    // PHASE 1: STATE UPDATE
    const now = performance.now();                                         // timestamp
    if (!window._lastFrameTime) window._lastFrameTime = now;               // initializes the last frame time if it's the first frame
    const deltaTime = now - window._lastFrameTime;                         // time elapsed since the last frame in milliseconds
    window._lastFrameTime = now;                                           // updates the last recorded timestamp for the next frame
    
    // Calculate and updates FPS every half second
    frameCount++; 
    if (currentTime - fpsTime >= 500) {                                    // checks if half a second has passed
      fps = Math.round((frameCount * 1000) / (currentTime - fpsTime));     // calculates the frames per second
      document.getElementById("fps").textContent = fps;                    // updates the FPS display in the UI
      frameCount = 0;                                                      // resets the frame counter
      fpsTime = currentTime;                                               // resets the FPS timer
    }

    // Update the dayTime variable (0.0 to 1.0) to advance the day/night cycle
    if (config.environment.autoCycle) {
      config.environment.dayTime = (config.environment.dayTime + (deltaTime / 1000) / config.environment.dayDuration) % 1.0;
      const slider = document.getElementById("dayTimeSlider");         // gets the time of day slider element
      if (slider) {                                                    // checks if the slider element exists
        slider.value = config.environment.dayTime;                     // sets the slider's value
        // also updates the output next to the slider
        if (slider.nextElementSibling) {                               // checks if there is an element next to the slider
          slider.nextElementSibling.value = config.environment.dayTime.toFixed(3);    // updates its value with 3 decimal places
        }
      }
      
      if (config.environment.dayTime < prevDayTime) {                   // checks if a new day has started (resets from 1 -> 0)
        if (!config.environment.wasWateredToday) {                      // checks if the plant was not watered on the previous day
          config.environment.isDead = true;                             // marks the plant as dead
          handlePlantDeath();                                           // calls handlePlantDeath (see line 1944) to trigger the plant death sequence
        }
        config.environment.wasWateredToday = false;                     // resets the watering status for the new day
      }
      prevDayTime = config.environment.dayTime;                         // saves the current value for comparison in the next frame
    }

    // Sequential growth logic with variable speed
    if (config.animation.growthSpeed > 0) {                            // speed=0 -> pause, speed=2 -> 500ms/branch, speed=5 -> 200ms/branch
        const growthIntervalMs = 1000 / config.animation.growthSpeed;  // calculates the time interval between adding new branches
        timeSinceLastGrowth += deltaTime;                              // accumulates the time since the last branch was added

        while (timeSinceLastGrowth >= growthIntervalMs && growthQueue.length > 0) { // if enough time has passed and if there are items in the queue
          const { objDesc, modelMatrix } = growthQueue.shift();                     // removes the first item from the queue
            
          if (objDesc.type === "leaf" && leafParts.length === 0) {           // checks if it's a leaf and if the leaf assets are not yet loaded
            console.log("⏸ Waiting for leaf asset before growing a leaf...");
            growthQueue.unshift({ objDesc, modelMatrix });                  // puts the object back at the front of the queue
            break;                                                          // exits the while loop for this frame
          }

          if (objDesc.type === "flower" && flowerParts.length === 0) {      // checks if it's a flower and if the flower assets are not yet loaded
            console.log("⏸ Waiting for flower asset before growing a flower...");
            growthQueue.unshift({ objDesc, modelMatrix });                  // puts the object back at the front of the queue
            break;                                                          // exits the while loop for this frame
          }

          if (objDesc) {                                                    // checks if the description object is valid
              addPlantPart(objDesc, modelMatrix);                           // calls addPlantPart (see line 1680) to add the new part to the scene
          }
          timeSinceLastGrowth -= growthIntervalMs;                          // subtracts the interval from the timer for accuracy
        }
    }

    updateObjCount();  // calls the updateObjCount function (see line 1984) to update the number of objects in the scene

    // GLB FLOWER ANIMATION: resets the per-frame cache
    if (flowerGLTF && flowerGLTF.animator) {
      // prevents the system to return the already calculated animation value of a previous flower for a neawly craeted flower
      flowerGLTF.animator._cache = null;
    }

    gl.enable(gl.DEPTH_TEST);                                          // enables depth testing for correct object occlusion
    gl.enable(gl.BLEND);                                               // enables transparency and blending
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);                // sets the standard blending function for transparency

    // DYNAMIC PARAMETERS
    // BACKGROUND: day/night cycle colors, chosen from this page: https://rgbcolorpicker.com/0-1#google_vignette
    // Dawn
    const DawnZ = [0.60, 0.78, 0.89];
    const DawnM = [0.79, 0.78, 0.79];
    const DawnH = [0.95, 0.72, 0.64];
    // Day
    const DayZ = [0.28, 0.51, 0.68];
    const DayM = [0.58, 0.78, 0.89];
    const DayH = [0.56, 0.73, 0.85];
    // Sunset
    const DuskZ = [0.45, 0.43, 0.54];
    const DuskM = [0.94, 0.63, 0.44];
    const DuskH = [0.74, 0.39, 0.35];
    // Night
    const NightZ = [0.00, 0.12, 0.20];
    const NightM = [0.12, 0.22, 0.31];
    const NightH = [0.23, 0.21, 0.32];

    const zenithStops = [
      { time: 0.00, value: NightZ },
      { time: 0.10, value: DawnZ },
      { time: 0.25, value: DayZ },
      { time: 0.50, value: DuskZ },
      { time: 0.75, value: NightZ },
      { time: 1.00, value: NightZ },
    ];
    const midStops = [
      { time: 0.00, value: NightM },
      { time: 0.10, value: DawnM },
      { time: 0.25, value: DayM },
      { time: 0.50, value: DuskM },
      { time: 0.75, value: NightM },
      { time: 1.00, value: NightM },
    ];
    const horizonStops = [
      { time: 0.00, value: NightH },
      { time: 0.10, value: DawnH },
      { time: 0.25, value: DayH },
      { time: 0.50, value: DuskH },
      { time: 0.75, value: NightH },
      { time: 1.00, value: NightH },
    ];

    // Use interpolateMultiStop (see line 1288) to interpolate the colors in time
    const zenithColor = interpolateMultiStop(zenithStops, config.environment.dayTime);
    const horizonColor = interpolateMultiStop(horizonStops, config.environment.dayTime);
    const midColor = interpolateMultiStop(midStops, config.environment.dayTime);    

    // SUN AND MOON
    // timeline of angles (in radians) for sun and moon
    const PI = Math.PI;
    const sunAngleStops = [
      { time: 0.000, value: [PI / 12] },                              // dawn start (15°)
      { time: 0.125, value: [PI / 4] },                               // dawn (45°)
      { time: 0.250, value: [75 * PI / 180] },                        // day start (75°)
      { time: 0.375, value: [105 * PI / 180] },                       // noon (105°)
      { time: 0.500, value: [3 * PI / 4] },                           // sunset start (135°)
      { time: 0.625, value: [165 * PI / 180] },                       // sunset (165°)
      { time: 0.750, value: [195 * PI / 180] },                       // night start (195°)
      { time: 0.825, value: [3 * PI / 2] },                           // night (270°)
      { time: 0.875, value: [2 * PI] },                               // night (360°)
      { time: 1.000, value: [PI / 12] }                               // dawn start (15°)
    
    ];
    const moonAngleStops = [
      { time: 0.000, value: [5 * PI / 6] },                           // dawn start (150°)
      { time: 0.125, value: [PI] },                                   // dawn (180°)
      { time: 0.250, value: [3 * PI / 2] },                           // day start (270°)
      { time: 0.375, value: [2 * PI] },                               // noon (360°/0°)
      { time: 0.500, value: [PI / 6] },                               // sunset start (30°)
      { time: 0.625, value: [PI / 3] },                               // sunset (60°)
      { time: 0.750, value: [PI / 2] },                               // night start (90°)
      { time: 0.875, value: [2 * PI / 3] },                           // night (120°)
      { time: 1.000, value: [5 * PI / 6] },                           // dawn start (150°)
    ];
    // calculate angles and positions
    const sunAngle = interpolateMultiStop(sunAngleStops, config.environment.dayTime, true)[0];
    const moonAngle = interpolateMultiStop(moonAngleStops, config.environment.dayTime, true)[0];
    const orbitRadius = 25.0;
    const sunPosition = [-Math.cos(sunAngle) * orbitRadius, (Math.sin(sunAngle) * orbitRadius) + config.environment.plantBaseY, -5];
    const moonPosition = [-Math.cos(moonAngle) * orbitRadius, ((Math.sin(moonAngle) * orbitRadius) + config.environment.plantBaseY) * 0.9, -5];  // moon slightly lower

    // LIGHT
    let lightPosition, lightColor, lightIntensity;
    const cameraPosition = camera.getEyePosition();        // calls function getEyePosition (see line 382) to obtain the position of th camera

    // checks if the sun is above the horizon (angle between 0 and 180 degrees)
    if (sunPosition[1] > config.environment.plantBaseY) {
      // day
      lightPosition = sunPosition;
      lightColor = [1.0, 1.0, 0.3];
      // intensity is proportional to the sun's height
      lightIntensity = (Math.max(0, sunPosition[1] / orbitRadius)) * 1.5;
    } 
    else {
      // night
      lightPosition = moonPosition;
      lightColor = [1.0, 1.0, 1.0];
      // intensity is proportional to the moon's height
      lightIntensity = Math.max(0, moonPosition[1] / orbitRadius) * 0.25;                        // moon is a bit less intense
    }

    // matrices for the light
    const lightView = Mat4.identity();
    Mat4.lookAt(lightView, sunPosition, [0, config.environment.plantBaseY, 0], [0, 1, 0]);
    const lightProj = Mat4.identity();
    const orthoSize = 10.0;                                                                      // area covered by the shadow map
    Mat4.ortho(lightProj, -orthoSize, orthoSize, -orthoSize, orthoSize, 1.0, 40.0);
    const lightSpaceMatrix = Mat4.multiply(Mat4.identity(), lightProj, lightView);

    // PHASE 2: SHADOW PASS
    // shadow pass
    gl.bindFramebuffer(gl.FRAMEBUFFER, shadowFramebuffer);  // binds the shadow map framebuffer: rendering will write to its depth texture
    gl.viewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);    // sets the viewport to match the shadow map dimensions
    gl.clear(gl.DEPTH_BUFFER_BIT);                          // clears the shadow map's depth buffer
    gl.cullFace(gl.FRONT);                                  // enables front-face culling to reduce artifacts ("peter panning")

    gl.useProgram(depthProgram);                            // uses the depth shader
    gl.uniformMatrix4fv(gl.getUniformLocation(depthProgram, "u_lightSpaceMatrix"), false, lightSpaceMatrix);    // sets the light's matrix

    // draws only the objects that cast shadows (branches, leaves, and flowers)
    for (const obj of sceneObjects) {
      if (!obj.description) continue;
    
      const type = obj.description.type;
        if (type === 'branch' || type === 'leaf' || type === 'flower') {  // only branches, leaves, and flowers cast shadows
    
          // Avoids shadows before the "birth" (consistent with main pass)
          const ageMs = currentTime - (obj.birthTime ?? 0);               // calculates the object's age in milliseconds
          // Checks if the leaf and the flower have a start time defined, falls back to 0
          const startLeaf   = (config.animation && config.animation.leafStartTime   != null) ? config.animation.leafStartTime   : 0;
          const startFlower = (config.animation && config.animation.flowerStartTime != null) ? config.animation.flowerStartTime : 0;
          if (type === 'leaf'   && ageMs < startLeaf)   continue; // skips drawing the shadow for the leaf if it hasn't reached its start time
          if (type === 'flower' && ageMs < startFlower) continue; // skips drawing the shadow for the flower if it hasn't reached its start time
    
          const matrixForShadow = obj.modelMatrix;                // retrieves the object's model matrix for the shadow

          // If the object is a flower, and the flower model has an animator, then we calculate the shadow for each part separately
          if (type === 'flower' && Array.isArray(flowerParts) && flowerGLTF && flowerGLTF.animator) {
            //  shadows synchronized with the main pass
            const animLen = (flowerGLTF.animationDuration ?? 0) || 0;  // gets the animation length from the glTF model, or defaults to 0
            const EPS = 1e-5;                                          // defines a small epsilon value
            // if the object is started (obj._startReady is true) and animTime is a number:
            // clamp the animTime within the animation's time range and otherwise set it to zero
            const tsec = (obj._startReady && typeof obj.animTime === 'number')
                ? Math.max(0, Math.min(obj.animTime, Math.max(0, animLen - EPS)))
                : 0;

            // Iterate through each part of the flower
            for (const part of obj.meshParts) {
              // animated world if time is over 0 and the animation exists, otherwise use the initial pose
              const world =
                  (tsec > 0 && animLen > 0)            // checks if the current time is positive and if there's an animation duration
                      ? (flowerGLTF.animator.getNodeWorldAt(tsec, part.nodeIndex) || part.localMatrix || Mat4.identity())  // gets the world matrix at the current time from the animator
                      : (part.localMatrix || Mat4.identity());                                                             // if there is no animation, uses the local matrix
              const model = Mat4.multiply(Mat4.identity(), matrixForShadow, world);                                        // model matrix for the current part of the flower
              shadowDrawer.draw(part.handle, { u_model: model, u_lightSpaceMatrix: lightSpaceMatrix });                    // draws the shadow of the current flower part
            }
          }
          else {
              // branches and leaves (or a flower without an animator): standard shadow
              shadowDrawer.draw(obj.meshHandle, {
                u_model: matrixForShadow,
                u_lightSpaceMatrix: lightSpaceMatrix
              });                                                                                                          // draws the shadow of the object
          }
        }
    }

    // restores the default buffer state
    gl.cullFace(gl.BACK);

    // PHASE 4: MAIN PASS
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);          // sets the viewport to match the canvas dimensions

    // background
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);     // clears the framebuffer's color and depth buffers 
    backgroundDrawer.draw(zenithColor, midColor, horizonColor, config.environment.horizonY);       // draws the background (gradient)

    // camera
    const proj = Mat4.projection(Mat4.identity(), Math.PI/3, canvas.width/canvas.height, 0.1, 100);
    const view = camera.getViewMatrix(Mat4.identity());

    // activates the shadow map on TEXTURE UNIT 0 to make it available to the shaders
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, shadowDepthTexture);

    const sunMatrix = Mat4.translate(Mat4.identity(), Mat4.identity(), sunPosition);
    const moonMatrix = Mat4.translate(Mat4.identity(), Mat4.identity(), moonPosition);
    
    const dynamicObjects = [...sceneObjects];      // copies the plant objects into the list of dynamic objects to be drawn

    // draws all objects in the scene framebuffer
    for (const obj of dynamicObjects) {
      if (obj.description && obj.description.type === 'ground') {
        meshDrawer.draw(obj.meshHandle, {
          u_model: obj.modelMatrix,
          u_view: view,
          u_proj: proj,
          u_alpha: 1.0,
          u_lightPosition: lightPosition,
          u_lightColor: lightColor,
          u_lightIntensity: lightIntensity,
          u_viewPosition: cameraPosition,
          u_diffuseColor: groundMaterial.diffuse,
          u_specularColor: groundMaterial.specular,
          u_shininess: groundMaterial.shininess,
          u_lightSpaceMatrix: lightSpaceMatrix,
          u_shadowMap: 0
        });
      }
      else { // logic for branches, leaves and flowers
        // visibility logic of the plant's lifecycle
        const age = currentTime - (obj.birthTime || currentTime); // age in ms
        if (obj.description.type === 'leaf'   && age < (config.animation.leafStartTime   ?? 0)) continue;   // skips the leaf if it hasn't reached its start time
        if (obj.description.type === 'flower' && age < (config.animation.flowerStartTime ?? 0)) continue;   // skips the flower if it hasn't reached its start time
    
        // default variables for "adult" objects
        let finalAlpha = 1.0;
        let finalColor = obj.color;                  // sets the default color to the object's base color
        let finalModelMatrix = obj.modelMatrix;      // default matrix is the one calculated by the L-System
    
        // applies youth effects only to branches/leaves (NOT to flowers: avoids squashing)
        // initial highlight (both branches and leaves)
        if ((obj.description.type === 'branch' || obj.description.type === 'leaf') &&
            age < config.animation.highlightDuration) {
        
          const highlightRatio = age / config.animation.highlightDuration;             // calculates the progress of the highlight animation
          const intensity = 1.5 - 0.5 * highlightRatio;                                // interpolates the color intensity from 1.5 down to 1.0
          finalColor = obj.color.map(c => Math.min(1.0, c * intensity));               // applies the intensity to the color, clamping at 1.0
          finalAlpha = 0.5 + 0.5 * highlightRatio;                                     // interpolates the alpha from 0.5 up to 1.0
        }
        
        // growth: branches and leaves with different anchors
        if (obj.description.type === 'branch' && age < config.animation.growthDuration) { // checks if the object is a branch and is still growing
          // branch: centered geometry -> use halfHeight=0.5
          const scaleY = Math.min(1.0, age / config.animation.growthDuration);                            // calculates the growth scale factor
          const halfHeight = 0.5;                                                                         // half the height of the base cylinder
          const moveToTipMatrix = Mat4.translate(Mat4.identity(), Mat4.identity(), [0, halfHeight, 0]);   // creates a matrix to move the pivot to the tip
          const scaleMatrix     = Mat4.scale    (Mat4.identity(), Mat4.identity(), [1, scaleY, 1]);       // creates a matrix for the scaling animation
          const moveBackMatrix  = Mat4.translate(Mat4.identity(), Mat4.identity(), [0,-halfHeight, 0]);   // creates a matrix to move the pivot back
          const growthTransform = Mat4.multiply(                                                          // combines the transformations
            Mat4.identity(),                                                                              // the output matrix
            moveBackMatrix,                                                                               // applies the move back transformation last
            Mat4.multiply(Mat4.identity(), scaleMatrix, moveToTipMatrix)                                  // applies scale, then move to tip first
          );
          finalModelMatrix = Mat4.multiply(Mat4.identity(), obj.modelMatrix, growthTransform);            // applies the final growth transform to the object's model matrix
        }
        
        if (obj.description.type === 'leaf' && age < config.animation.growthDuration) {                   // checks if the object is a leaf and is still growing
          // leaf: pivot on the stem at the base -> scale directly, no halfHeight
          const scaleY = Math.min(1.0, age / config.animation.growthDuration);                            // calculates the growth scale factor
          const scaleMatrix = Mat4.scale(Mat4.identity(), Mat4.identity(), [1, scaleY, 1]);               // creates a matrix for direct scaling
          finalModelMatrix = Mat4.multiply(Mat4.identity(), obj.modelMatrix, scaleMatrix);                // applies the growth transform to the object's model matrix
        }
    
        if (obj.description.type === 'flower') {
          const flowerScale = 0.8;
          const scaleMatrix = Mat4.scale(Mat4.identity(), Mat4.identity(), [flowerScale, flowerScale, flowerScale]);   // creates a uniform scaling matrix
          finalModelMatrix = Mat4.multiply(Mat4.identity(), finalModelMatrix, scaleMatrix);                            // applies the scaling to the final model matrix
        }
    
        const drawer = obj.drawer || meshDrawer;                                              // selects the appropriate drawer, falling back to the default meshDrawer
        const material = obj.material || branchMaterial;                                      // selects the appropriate material, falling back to the branch material
        const specularColor = material.specular || [0.1, 0.1, 0.1];                           // gets the specular color, with a fallback
        const shininess = material.shininess !== undefined ? material.shininess : 32.0;       // gets the shininess value, with a fallback
    
        // common uniforms
        let uniforms = {
          u_model: finalModelMatrix,
          u_view: view,
          u_proj: proj,
          u_alpha: finalAlpha,
          u_lightPosition: lightPosition,
          u_lightColor: lightColor,
          u_lightIntensity: lightIntensity,
          u_viewPosition: cameraPosition,
          u_lightSpaceMatrix: lightSpaceMatrix,        // matrix for shadow mapping
          u_shadowMap: 0,                              // tells the shader to use texture unit 0 for the shadow map
          u_specularColor: specularColor,
          u_shininess: shininess
        };
    
        // specific uniforms for drawer type
        if (drawer === texturedDrawer) {                    // checks if the object should be drawn with the textured shader
          if (obj.texture) {                                // checks if the object has a texture
            gl.activeTexture(gl.TEXTURE1);                  // activates texture unit 1
            gl.bindTexture(gl.TEXTURE_2D, obj.texture);     // binds the object's texture
            uniforms.u_diffuseTexture = 1;                  // tells the shader to use texture unit 1
            uniforms.u_tintColor = [1,1,1];                 // sets a neutral tint color (no color change)
          }
        } 
        else { // for non-textured objects
          uniforms.u_diffuseColor = finalColor;             // sets the calculated diffuse color
        }

        if (config.environment.isDead && deadOverlay && uniforms.u_diffuseColor) {      // checks if the plant is dead and an overlay color is set
          uniforms.u_diffuseColor = [                                                   // modulates the diffuse color with the overlay
            uniforms.u_diffuseColor[0] * deadOverlay[0],
            uniforms.u_diffuseColor[1] * deadOverlay[1],
            uniforms.u_diffuseColor[2] * deadOverlay[2],
          ];
          uniforms.u_alpha *= deadOverlay[3];                                           // modulates the alpha with the overlay
        }
    
        // ANIMATED FLOWERS: part by part with delay + nightly rewind
        if (obj.description && obj.description.type === 'flower' && Array.isArray(obj.meshParts) && flowerGLTF && flowerGLTF.animator) { // checks if the object is a valid, animatable flower
          // per-flower state with delay and dayRange to avoid global re-triggers
          const animLen = flowerGLTF.animationDuration || 2.0;                                                   // gets the animation length, with a fallback
          const extraD = (FLOWER_ANIM?.extraStartDelayMs != null) ? FLOWER_ANIM.extraStartDelayMs : 2000;        // gets the extra delay for animation start
          const speed = (FLOWER_ANIM?.speed != null) ? FLOWER_ANIM.speed : 1.0;                                  // gets the animation speed multiplier
        
          const objDelay  = (obj.animDelayMs != null) ? obj.animDelayMs : 0;                                     // per-flower delay
          const baseStart = (config.animation.flowerStartTime ?? 0) + extraD + objDelay;                         // calculates the absolute start time for the animation
          const ageMs     = currentTime - (obj.birthTime ?? 0);                                                  // calculates the object's age in milliseconds
          
          // per-flower window: if missing, fallback to global
          const fallbackDR = FLOWER_ANIM?.dayRange ?? [0.25, 0.75];                                              // gets the day range for animation, with a fallback
          const activeRange= (obj.dayRange && obj.dayRange.length === 2) ? obj.dayRange : fallbackDR;            // determines the active range for this specific flower
          
          // per-flower day/night state
          const tDay  = (config.environment.timeOfDay !== undefined)                                             // gets the current time of day
                         ? config.environment.timeOfDay
                         : (config.environment.dayTime !== undefined ? config.environment.dayTime : 0.5);
          const isDay = (tDay >= activeRange[0] && tDay <= activeRange[1]);                                      // determines if it is currently "day" for this flower

          // becomes "ready" only when it has reached its start time
          const becameReady = (!obj._startReady) && (ageMs >= baseStart);                                        // checks if the flower just became ready to animate
          if (becameReady) {
            obj._startReady = true;                                                                              // sets its ready flag
            // starts ONCE based on the current day/night state
            if (isDay) {
              obj.animPhase = "opening";                                                                         // starts opening
              obj.animTime = 0;                                                                                  // from the beginning of the animation
            }
            else { //night
              obj.animPhase = "closing";                                                                         // starts closing
              obj.animTime = animLen;                                                                            // from the end of the animation
            }
            obj._lastIsDay = isDay;                                                                              // records the current day/night state
          }
        
          // day/night changes after the start -> a single transition
          if (obj._startReady) {                                                                                 // if the flower is ready to animate
            if (obj._lastIsDay === null) obj._lastIsDay = isDay;                                                 // initializes the last day state if needed
            if (isDay !== obj._lastIsDay) {                                                                      // checks if the day/night state has changed
              if (isDay && obj.animPhase !== "opening" && obj.animPhase !== "open") {                            // if it's now day and the flower isn't already opening/open
                obj.animPhase = "opening";
              } 
              else if (!isDay && obj.animPhase !== "closing" && obj.animPhase !== "closed") {                    // if it's now night and the flower isn't already closing/closed
                obj.animPhase = "closing";
              }
              obj._lastIsDay = isDay;                                                                            // updates the last seen day/night state
            }
        
            // progress
            if (obj.animPhase === "opening") {
              obj.animTime += (deltaTime/1000) * speed;                                                          // advances the animation time
              if (obj.animTime >= animLen) { obj.animTime = animLen; obj.animPhase = "open"; }                   // checks if the animation is finished
            } 
            else if (obj.animPhase === "closing") {
              obj.animTime -= (deltaTime/1000) * speed;                                                          // rewinds the animation time
              if (obj.animTime <= 0) { obj.animTime = 0; obj.animPhase = "closed"; }                             // checks if the animation is finished
            }
          }
        
          // time to pass to the animator:
          // - if not yet "ready", stay in the initial pose (t = 0)
          // - otherwise clamp between [0, animLen)
          const EPS  = 1e-5;                                        // a small epsilon to avoid floating point issues at the end of the animation
          const tsec = obj._startReady ? Math.max(0, Math.min(obj.animTime, Math.max(0, animLen - EPS))) : 0;    // calculates the final clamped animation time in seconds
        
          const finalModelMatrixForFlower = finalModelMatrix;                                                    // gets the base matrix for the flower
          for (const part of obj.meshParts) {
            const localAnimated =                                                                                // gets the animated local matrix for this part
              flowerGLTF.animator.getNodeWorldAt(tsec, part.nodeIndex) ||                                        // from the animator
              part.localMatrix || Mat4.identity();                                                               // or falls back to its static local matrix
        
            const model = Mat4.multiply(Mat4.identity(), finalModelMatrixForFlower, localAnimated);              // calculates the final world matrix for this part
            uniforms.u_model = model;                                                                            // updates the model uniform for this part's draw call
        
            const partDrawer = part.drawer || meshDrawer;                                                        // selects the drawer for this part
            if (partDrawer === texturedDrawer && part.texture) {                                                 // if the part is textured
              gl.activeTexture(gl.TEXTURE1);                                                                     // activates texture unit 1
              gl.bindTexture(gl.TEXTURE_2D, part.texture);                                                       // binds the part's texture
              uniforms.u_diffuseTexture = 1;                                                                     // sets the texture uniform
              uniforms.u_tintColor = [1, 1, 1];                                                                  // sets a neutral tint
              partDrawer.draw(part.handle, uniforms);                                                            // draws the textured part
            } 
            else { // if the part is not textured
              const bc = (part.material && part.material.baseColorFactor) ? part.material.baseColorFactor : [1.0, 0.4, 0.6, 1.0];   // gets the color from the material or a fallback
              uniforms.u_diffuseColor = [bc[0], bc[1], bc[2]];                                                                      // sets the diffuse color uniform
              meshDrawer.draw(part.handle, uniforms);                                                                               // draws the part with a solid color
            }
          }
        
        } 
        else { // if the object is not an animatable flower
          // non-flower objects or flower without animator -> standard drawing
          drawer.draw(obj.meshHandle, uniforms, gl.TRIANGLES);
        }
      }
    }

    const isAboveGround = (cameraPosition[1] > config.environment.plantBaseY + 0.01);

    // Add sun and sun ruys only if above the horizon (Y > plantBaseY)
    if (sunPosition[1] > config.environment.plantBaseY) {
      const sunScaleFactor = 1.7;                                                                                              // the sun will be 1.7 times larger than the moon
      const scaleMatrix = Mat4.scale(Mat4.identity(), Mat4.identity(), [sunScaleFactor, sunScaleFactor, sunScaleFactor]);      // creates a uniform scaling matrix
      const finalSunMatrix = Mat4.multiply(Mat4.identity(), sunMatrix, scaleMatrix);

      // rendering matrices
      let modelView = Mat4.multiply(Mat4.identity(), view, finalSunMatrix);                                                    // calculates the model-view matrix for the sun
      let mvp = Mat4.multiply(Mat4.identity(), proj, modelView);                                                               // calculates the final model-view-projection matrix

      sunMoonDrawer.draw(sunMoonMeshHandle, {
        u_mvp: mvp,
        u_color: [1.0, 1.0, 0.7]
      });

      if (config.weather.showSunRays && isAboveGround) {                                                               // checks if the sun rays effect should be shown
        // clip = VP * sunW  (column-major)
        const sunWorldPos = [sunPosition[0], sunPosition[1], sunPosition[2], 1.0];                                     // defines the sun's position in homogeneous coordinates (w=1)

        // view-projection matrix VP = P * V
        const VP = Mat4.multiply(Mat4.identity(), proj, view);
      
        // VP * sunWorldPos multiplication (column-major)
        const clipX = VP[0]*sunWorldPos[0] + VP[4]*sunWorldPos[1] + VP[8]*sunWorldPos[2]  + VP[12]*sunWorldPos[3];     // calculates the x coordinate in clip space
        const clipY = VP[1]*sunWorldPos[0] + VP[5]*sunWorldPos[1] + VP[9]*sunWorldPos[2]  + VP[13]*sunWorldPos[3];     // calculates the y coordinate in clip space
        const clipW = VP[3]*sunWorldPos[0] + VP[7]*sunWorldPos[1] + VP[11]*sunWorldPos[2] + VP[15]*sunWorldPos[3];     // calculates the w component for perspective division
      
        if (clipW > 0.0) {                                                                                             // if clipW > 0.0, the sun is in front of the camera
          // from clip space to Normalized Device Coordinates (-1 to 1)
          const ndcX = clipX / clipW;                                                                                  // performs the perspective divide for x
          const ndcY = clipY / clipW;                                                                                  // performs the perspective divide for y
          // from Normalized Device Coordinates to screen space [0..1] with inverted Y
          const sunPos = [ ndcX * 0.5 + 0.5, ndcY * 0.5 + 0.5 ];                                                       // converts NDC to the [0,1] range expected by the shader
  
          gl.enable(gl.BLEND);                                                                                         // enables blending for transparency
          gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);                                                          // sets the standard alpha blending function
          sunRaysDrawer.draw(sunRaysMeshHandle, {                                                                      // draws the sun rays effect
            u_sunPos: sunPos,
            u_time: currentTime / 1000.0,
            u_color: [1.0, 0.95, 0.7],
            u_intensity: 0.6 * Math.max(0.0, Math.min(1.0, lightIntensity)),                                           // modulates intensity to make the rays more transparent
            u_aspect: canvas.width / canvas.height                                                                     // passes the screen aspect ratio to correct distortion
          }, gl.TRIANGLES);
  
          gl.disable(gl.BLEND);                                                                                        // disables blending for cleanup
        }
      }
    }

    // Add moon only if above the horizon (Y > plantBaseY)
    if (moonPosition[1] > config.environment.plantBaseY) {
      let modelView = Mat4.multiply(Mat4.identity(), view, moonMatrix);
      let mvp = Mat4.multiply(Mat4.identity(), proj, modelView);

      sunMoonDrawer.draw(sunMoonMeshHandle, {
        u_mvp: mvp,
        u_color: [0.9, 0.9, 1.0]
      });
    }

    // Draw stars and fireflies only at night
    if (isAboveGround && 0.75 < config.environment.dayTime && config.environment.dayTime < 0.99) {
      let fireflyModelMatrix = Mat4.identity();                                // initializes the model matrix for the fireflies
      let starMatrix = Mat4.clone(view);                                       // clones the view matrix to make the stars follow the camera's rotation
      starMatrix[12] = 0; starMatrix[13] = 0; starMatrix[14] = 0;              // zeroes out the translation components of the star matrix
      // we move the fireflies to the height of the plant's base, plus a small offset to make them float above the ground
      Mat4.translate(fireflyModelMatrix, fireflyModelMatrix, [0, config.environment.plantBaseY + 4.0, 0]);         // applies the translation to the firefly model matrix
      let firefliesModelView = Mat4.multiply(Mat4.identity(), view, fireflyModelMatrix);                           // calculates the model-view matrix for the fireflies
      let firefliesMVP = Mat4.multiply(Mat4.identity(), proj, firefliesModelView);                                 // calculates the final MVP matrix for the fireflies
      let starsMVP = Mat4.multiply(Mat4.identity(), proj, starMatrix);                                             // calculates the final MVP matrix for the stars

      const timeInSeconds = currentTime / 1000.0;                                                                  // converts the current time to seconds for animations
      const twinkle = Math.sin(timeInSeconds * 2.0) * 0.1 + 0.9;                                                   // varies between 0.8 and 1.0
      const starColor = [twinkle, twinkle, twinkle];                                                               // creates a pulsating white/grey color for the stars

      // fireflies
      gl.depthMask(false);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);  
      fireflyDrawer.draw(fireflyMeshHandle, {                                                                      // draws the fireflies with their specific uniforms
        u_mvp: firefliesMVP,
        u_time: timeInSeconds,
        u_color: [1.0, 0.85, 0.2]   // bright yellow
      }, gl.POINTS); // specifies to draw them as points
      
      // stars with glow
      starDrawer.draw(starMeshHandle, {                                                                            // draws the stars with their specific uniforms
        u_mvp: starsMVP,
        u_color: starColor
      });
      
      gl.depthMask(true);                                                                                         // enables writing to the depth buffer
      gl.disable(gl.BLEND);
    }

    // Draw the rain if active
    if (config.weather.isRaining) {
      config.environment.wasWateredToday = true;                                      // watering logic
    
      let rainModelMatrix = Mat4.identity(); 
      let modelView = Matá.multiply(Mat4.identity(), view, rainModelMatrix);
      let mvp = Mat4.multiply(Mat4.identity(), proj, modelView);

      const timeInSeconds = currentTime / 1000.0;                                     // converts the current time to seconds for animations

      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);                                             // sets additive blending for a bright, semi-transparent effect
      gl.depthMask(false);                                                            // disables writing to the depth buffer to avoid sorting issues with transparent particles

      const rainTop = 20.0;                                                           // defines the top y-coordinate from which the rain falls
      const rainBottom = -8.0;                                                        // defines the bottom y-coordinate where the rain disappears

      rainDrawer.draw(rainMeshHandle, {                                               // draws the rain particles
        u_mvp: mvp,
        u_proj: proj,
        u_time: timeInSeconds,
        u_color: [0.7, 0.8, 0.9],                                                     //  blueish-white color 
        u_rainTop: rainTop,
        u_rainBottom: rainBottom
      }, gl.POINTS);                                                                  // specifies to draw them as points
          
      gl.depthMask(true);                                                             // re-enables writing to the depth buffer
      gl.disable(gl.BLEND);                                                           // disables blending for cleanup
    }

    // Draw clouds if they are active
    if (config.weather.showClouds) {
      let cloudModelMatrix = Mat4.identity();
      Mat4.translate(cloudModelMatrix, cloudModelMatrix, [0, 12, 0]);                 // translates the clouds to be high up in the sky
      
      const timeInSeconds = currentTime / 1000.0;                                     // converts the current time to seconds for animation
      gl.enable(gl.BLEND);                                                            // enables blending for the semi-transparent clouds
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);                             // sets the standard alpha blending function
      gl.depthMask(false);                                                            // disables writing to the depth buffer for correct transparency sorting
      cloudDrawer.draw(cloudMeshHandle, {                                             // draws the clouds
        u_model: cloudModelMatrix,
        u_view: view,
        u_proj: proj,
        u_time: timeInSeconds                                                         // passes the current time to animate the cloud noise
      }, gl.TRIANGLES);                                                               // specifies to draw them as triangles
      
      gl.depthMask(true);                                                             // re-enables writing to the depth buffer for the rest of the scene
      gl.disable(gl.BLEND);                                                           // disables blending for cleanup
    }

    // Restore the viewport to full screen for the next frame
    gl.viewport(0, 0, canvas.width, canvas.height);

    requestAnimationFrame(render);                     // asks the browser to call 'render' again on the next available frame
  }
  
  // LAUNCHING THE APPLICATION
  await initBaseGeometries();     // load geometries
  buildLSystem();                 // calls the buildLSystem functions (see line 1881) to construct the initial scene
  requestAnimationFrame(render);  // calls the render function (see line 1989) to start the rendering loop
});
