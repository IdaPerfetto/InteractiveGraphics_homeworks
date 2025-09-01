/*
This file contains the necessary code to load and parse 3D models stored in the binary glTF (.glb) format
It is composed of two main classes: 
1) GLTFLoader, which handles fetching the file and unpacking its main components
2) GLTFParser, which processes the unpacked data to extract geometry, materials, and animations into a format usable by the WebGL application
*/

/*
For the GLTFLoader the primary responsibilities are to 
1) fetch the file from a URL
2) separate its binary content into the JSON scene description and the binary buffer containing geometry data
*/
class GLTFLoader {
  async load(url) {
    // Asynchronously fetches a .glb file from a specified URL and initiates the parsing process
    // The async keyword allows the use of await inside, making it easier to handle asynchronous operations like network requests
    const response = await fetch(url);                                            // make a network request to the provided url
    if (!response.ok) throw new Error(`HTTP ${response.status} loading ${url}`);  // error if the response is not successfull
    const arrayBuffer = await response.arrayBuffer();                             // returns the body of the response as an ArrayBuffer
    return this.parse(arrayBuffer);                                               // calls the parse method (see line 23) and returns the fully processed scene object
  }
  
  parse(arrayBuffer) {
    // Unpacks a raw ArrayBuffer containing .glb data into its constituent JSON and binary chunks
    const dataView = new DataView(arrayBuffer);                              //  creates a DataView for the ArrayBuffer
    // Reads the first 4 bytes of the file as a 32-bit unsigned integer, 0x46546c67 corresponds to the ASCII string "glTF"
    if (dataView.getUint32(0, true) !== 0x46546c67) throw new Error("Not a GLB");  // true specifies little-endian byte order required by the glTF 2.0 specification
    
    const version = dataView.getUint32(4, true);                             // reads the next 4 bytes (starting at offset 4) to get the glTF version number
    if (version !== 2) {
      console.warn("GLB version is not 2; attempting anyway");               // this parser is designed for glTF 2.0 (the version 2.0 is assigned by blender)
    }
    
    const length = dataView.getUint32(8, true);                              // reads the next 4 bytes (starting at offset 8) to get the length of the .glb file in bytes
    let offset = 12;                                                         // byte position right after the 12-byte .glb header, where the first data chunk begins
    let json = null;                                                         // this variable will hold the extracted JSON object
    let binary = null;                                                       // this variable will hold the binary data chunk
    while (offset < length) {                                                // process one chunk at a time
      const chunkLength = dataView.getUint32(offset, true);
      offset += 4;
      const chunkType = dataView.getUint32(offset, true);
      offset += 4;
      const chunkData = arrayBuffer.slice(offset, offset + chunkLength);     // new buffer (copy of the specified section of the main buffer) to extract the data of the chunk
      offset += chunkLength;                                                 // advance the offset to the beginning of the next chunk of data
      
      if (chunkType === 0x4E4F534A) {                                        // 0x4E4F534A corresponds to the ASCII string "JSON"
        // If the chunk is a JSON:
        // 1) decode the binary data into a UTF-8 string using TextDecoder
        // 2) parse the string into a JavaScript object using JSON.parse
        json = JSON.parse(new TextDecoder().decode(new Uint8Array(chunkData)));
      }
      else if (chunkType === 0x004E4942) {                                   // 0x004E4942 corresponds to "BIN" followed by a null terminator
        binary = chunkData;                                                  // this is the binary buffer containing geometry, animation, and image data.
      }
    }
    if (!json || !binary) throw new Error("Invalid GLB: missing JSON or BIN chunk");
    return new GLTFParser(json, binary).parse();            // creates a new instance of GLTFParser (see line 69) and calls its parse method (see line 235) to process the model data
  }
}

/*
The GLTFParser takes the separated JSON structure and binary buffer from a .glb file extracts all the scene information:
- geometry
- materials
- scene graph hierarchy
- animations
into a structured format that the bloomscape.js application can directly use for rendering with WebGL
*/
class GLTFParser {
  constructor(json, binary) {
    this.json = json;                    // entire scene description (nodes, meshes, materials, animations, etc.)
    this.binary = binary;                // raw binary data chunk
    this.bufferViews = new Map();        // initializes a Map to use as a cache for bufferView data to avoid redundant data extraction for views used by multiple accessors
    this.parents = null;                 // this will later be populated with an array for quickly looking up the parent of any node in the scene graph
  }

  getBufferView(index) {
    // Gets a specific slice of the main binary buffer (takes the index of the desired bufferView as an argument) as defined by a bufferView in the JSON
    // It uses a cache to avoid re-slicing the same data multiple times:
    // if the data for this bufferView index has already been extracted and cached in the bufferViews map return the cached data
    if (this.bufferViews.has(index)) return this.bufferViews.get(index);
    const bufferView = this.json.bufferViews[index];                                              // this object contains properties like byteOffset and byteLength
    // Create a Uint8Array view onto the main binary buffer that:
    // - starts at the specified byteOffset (or 0 if not defined) 
    // - has the specified byteLength
    const data = new Uint8Array(this.binary, bufferView.byteOffset || 0, bufferView.byteLength);  // creates a reference to the part of the buffer without copying the data
    this.bufferViews.set(index, data);                                                            // stores the Uint8Array view in the cache using its index as the key for future requests
    return data;                                                                                  // extracted Uint8Array data view
  }

  getAccessorData(index) {
    // Interprets the data from a bufferView according to an accessor's definition:
    // - an accessor specifies the data type, component type, count, and layout (interleaved or not)
    // - returns a correctly typed and formatted TypedArray ready for WebGL
    const acc = this.json.accessors[index];                     // accessor: definition object that describes how to interpret the binary data
    const bvj = this.json.bufferViews[acc.bufferView];          // bufferView: definition object that the accessor points to
    const src = this.getBufferView(acc.bufferView);             // calls the getBufferView method (see line 77) to get the raw binary data (Uint8Array) for the bufferView

    const cmap = {                                              // maps glTF component type constants to their corresponding JavaScript TypedArray constructors
      5120:Int8Array,   5121:Uint8Array,                        // BYTE and UNSIGNED_BYTE
      5122:Int16Array,  5123:Uint16Array,                       // SHORT and UNSIGNED_SHORT
      5125:Uint32Array, 5126:Float32Array                       // UNSIGNED_INT and FLOAT
    };
    const tsize = {SCALAR:1, VEC2:2, VEC3:3, VEC4:4, MAT2:4, MAT3:9, MAT4:16};  // maps glTF type strings to the number of components they have
    const Comp = cmap[acc.componentType];                                       // gets the correct TypedArray constructor from the cmap based on the accessor's componentType
    const n = tsize[acc.type];                                                  // gets the number of components per element from the tsize based on the accessor's type
    const compB = Comp.BYTES_PER_ELEMENT;                                       // size in bytes of a single component
    const tight = n * compB;                                                    // "tight" or non-interleaved (see line 112) stride: total size of one complete component in bytes
    const stride = bvj.byteStride || tight;                                     // stride: number of bytes from the start of one element to the start of the next (byteStride if defined, otherwise tight)
    const base = (src.byteOffset||0) + (acc.byteOffset||0);                     // calculates the starting byte offset for this accessor's data within the main binary buffer

    // Non-interleaved: in a non-interleaved buffer all attributes of a vertex (location, normal, UV, etc.) are stored in separate arrays
    if (stride === tight) return new Comp(src.buffer, base, acc.count * n);     // new TypedArray view over the section of the buffer (acc.count * n = number of components to include in the view)

    // Interleaved: in an interleaved buffer all attributes of a vertex (location, normal, UV, etc.) are stored one after the other in the same array, with a stride
    const out = new Comp(acc.count * n);                                        // new empty TypedArray of the correct size to hold the de-interleaved data
    const dv = new DataView(src.buffer);                                        // creates a DataView on the source buffer to allow reading specific data types at arbitrary offsets
    const le = true;                                                            // little-endian byte order (least significant byte at the lowest memory address), the standard for glTF

    function readAt(off, T) {
      // Reads a single value of a specific type T from a given offset off
      switch (T) {
        case Int8Array:    return dv.getInt8(off);                              // if the type is Int8Array, read a signed 8-bit integer
        case Uint8Array:   return dv.getUint8(off);                             // if the type is Uint8Array, read an unsigned 8-bit integer
        case Int16Array:   return dv.getInt16(off, le);                         // if the type is Int16Array, read a signed 16-bit integer in little-endian format
        case Uint16Array:  return dv.getUint16(off, le);                        // if the type is Uint16Array, read an unsigned 16-bit integer in little-endian format
        case Uint32Array:  return dv.getUint32(off, le);                        // if the type is Uint32Array, read an unsigned 32-bit integer in little-endian format
        case Float32Array: return dv.getFloat32(off, le);                       // if the type is Float32Array, read a 32-bit floating-point number in little-endian format
      }
    }
    let k = 0;
    for (let i=0; i<acc.count; i++) {                // for each element defined by the accessor
      const e = base + i * stride;                   // calculates the starting byte position of the current element using the stride
      for (let c = 0; c < n; c++) {                  // for each component of the element
        out[k++] = readAt(e + c * compB, Comp);      // calls the readAt function (see line 120) with the specific offset to write the component value sequentially into the out array
      }
    }
    return out;
  }

  static composeTRS(out, t, r, s){
    // Creates a 4x4 transformation matrix from Translation (t), Rotation (r, quaternion), and Scale (s) components
    const tx = t ? t[0] : 0,   ty = t ? t[1] : 0,   tz = t ? t[2] : 0;                       // extract the translation: if t is not provided the no translation
    const sx = s ? s[0] : 1,   sy = s ? s[1] : 1,   sz = s ? s[2] : 1;                       // extract the scale: if s is not provided the scale = 1
    const  x = r ? r[0] : 0,    y = r ? r[1] : 0,    z = r ? r[2] : 0,    w = r ? r[3] : 1;  // extract the rotation (quaternion): if r is not provided the no rotation
    // pre-calculate several intermediate products from the quaternion components to optimize the main calculation
    const x2 = x + x,          y2 = y + y,          z2 = z + z;
    const xx = x * x2,         xy = x * y2,         xz = x * z2;
    const yy = y * y2,         yz = y * z2,         zz = z * z2;
    const wx = w * x2,         wy = w * y2,         wz = w * z2;
    // final 4x4 matrix: 
    // - the top-left 3x3 submatrix is the rotation matrix derived from the quaternion, with each column scaled by the corresponding scale factor (sx, sy, sz)
    // - the last column (out[12], out[13], out[14]) is set to the translation components (tx, ty, tz)
    out[0] = (1 - (yy + zz)) * sx;    out[1] = (xy + wz) * sx;          out[2] = (xz - wy) * sx;         out[3] = 0;
    out[4] = (xy - wz) * sy;          out[5] = (1 - (xx + zz)) * sy;    out[6] = (yz + wx) * sy;         out[7] = 0;
    out[8] = (xz + wy) * sz;          out[9] = (yz - wx) * sz;          out[10] = (1 - (xx + yy)) *sz;   out[11] = 0;
    out[12] = tx;                     out[13] = ty;                     out[14] = tz;                    out[15] = 1;
    return out;
  }
  static v3lerp(a, b, t) {
    // Linear Interpolation between two 3D vectors a and b
    return [
      a[0] * (1 - t) + b[0] * t,
      a[1] * (1 - t) + b[1] * t,
      a[2] * (1 - t) + b[2] * t
    ];
  }
  static slerp(a, b, t) {
    // Spherical Linear Interpolation between two quaternions a and b (shortest path rotation)
    // Assigns the components of the input quaternions a and b to local variables for semplicity
    let ax = a[0],  ay = a[1],  az = a[2],  aw = a[3];
    let bx = b[0],  by = b[1],  bz = b[2],  bw = b[3];
    let cos = ax * bx + ay * by + az * bz + aw * bw;       // dot product of the two quaternions
    if (cos < 0) {
    // inverts the second quaternion b (since quaternions q and -q represent the same rotation) and its dot product to use the shorter path
      bx = - bx;
      by = - by;
      bz = - bz;
      bw = - bw;
      cos = - cos;
    }
    if (cos > 0.9995) {                                   // if the two quaternions are very close together
      // slerp is numerically unstable so this block falls back to simple linear interpolation (lerp), which is a good approximation for small angles and avoids division by zero
      return [
        ax + t * (bx - ax),
        ay + t * (by - ay),
        az + t * (bz - az),
        aw + t * (bw - aw)
      ];
    }
    const theta = Math.acos(cos);                        // calculates the angle (theta) between the quaternions from their dot product
    const sin = Math.sin(theta); 
    const s0 = Math.sin((1 - t) * theta) / sin;          // calculates the scaling factor for the first quaternion a
    const s1 = Math.sin(t * theta) / sin;                // calculates the scaling factor for the second quaternion b
    return [                                             // calculates and returns each component of the resulting quaternion by applying the scaling factors
      ax * s0 + bx * s1,
      ay * s0 + by * s1,
      az * s0 + bz * s1,
      aw * s0 + bw * s1
    ];
  }

  ensureParents() {
    // Builds a flat array that maps each node's index to its parent's index
    //  This is done once and cached to make hierarchical transformations (calculating world matrices) more efficient
    if (this.parents) return;                            // if the parent map (this.parents) has already been built, the function returns immediately to avoid redundant work
    const n = this.json.nodes || [];                     // array of nodes from the JSON, or an empty array if no nodes are defined
    this.parents = new Array(n.length).fill(undefined);  // new array with the same size as the number of nodes, initializing all parent references to undefined
    for (let i = 0; i < n.length; i++) {
      if (n[i].children) {                               // if the current node at index i has a children array property
        for (const c of n[i].children) {                 // for each child index c
          this.parents[c]=i;                             // sets the corresponding entry in the parents array to i, the index of the current (parent) node
        }
      }
    }
  }

  getNodeWorld(nodeIndex, nodeMatrices) {
    // Recursively calculates the final world-space transformation matrix for a given node, takes the index of the node and an array to use for caching results
    // It combines the node's local transformation with its parent's world transformation
    // This is used for static scenes without animation (for the case with animation see getNodeWorldAt at line 380)
    if(nodeMatrices[nodeIndex]) return nodeMatrices[nodeIndex];     // returns the cached world matrix for this nodeIndex if it has already been calculated and stored
    const n = this.json.nodes[nodeIndex];
    const local = GLTFParser.composeTRS(Mat4.identity, n.translation, n.rotation, n.scale);  // calls composeTRS (see line 141) to compose the node's local transformation matrix
    const p = this.parents[nodeIndex];
    if(p === undefined) {                                           // if the node has no parent it's a root node of the scene graph
      nodeMatrices[nodeIndex] = local;                              // if it's a root node, its world matrix is simply its local matrix
      return local; 
    }
    const parentWorld = this.getNodeWorld(p,nodeMatrices);          // if there is a parent, it recursively calls itself to get the parent's world matrix
    const world = Mat4.multiply(Mat4.identity,parentWorld,local);   // multiplies the parent's world matrix by the node's local matrix to get the final world matrix for the current node
    nodeMatrices[nodeIndex] = world;
  }

  parse(){
    // Converts the raw JSON and binary data into a structured scene object containing drawable meshes and animation data
    const scene = {               // initializes the main scene object that will be built and returned
      meshes: [],
      materials: [], 
      animations: null,           // object to hold the parsed animation keyframe data
      animator: null,             // object with methods to calculate animated transformations at runtime
      _imageBlobs: []             // temporary internal array to store image data as Blob objects before they are converted to WebGL textures
    };

    // MATERIALS PARSING SECTION
    if (this.json.materials) {                                // if the glTF file defines any materials
      for (const m of this.json.materials) {                  // for each material definition in the JSON
        const p = (m.pbrMetallicRoughness || {});             // accesses the pbrMetallicRoughness property of the material, provides an empty object as a fallback if it doesn't exist    
        scene.materials.push({
          baseColorFactor: p.baseColorFactor || [1, 1, 1, 1]  // extracts the baseColorFactor (a 4-component RGBA color multiplier), provides a default white if it's not defined
        });
      }
    }

    const meshes = this.json.meshes || [];                           // gets the array of mesh definitions from the JSON, or an empty array if none exist
    const nodes  = this.json.nodes  || [];                           // gets the array of node definitions (the scene graph) from the JSON
    const defaultScene = this.json.scenes[this.json.scene || 0];     // gets the definition for the default scene to be rendered
    this.ensureParents();                                            // calls the method ensureParents (see line 203) to build the parent lookup table for the scene graph nodes
    const nodeMatrices = new Array(nodes.length);                    // creates an empty array to be used for caching the static world matrices of the nodes

    // collect meshes per node with material (+ optional image)
    const pushPrimitive = (nodeIndex, prim, worldMatrix) => {
      // Processes a single drawable part of a mesh
      // nodeIndex: the index of the node in the glTF scene (needed to know which "logical node" the geometry belongs to)
      // prim: the glTF primitive, i.e. a sub-block of the mesh (a node can have a mesh with multiple materials, each piece is a primitive)
      // worldmatrix: the world matrix calculated for the node (via getNodeWorld (see line 218) or getNodeWorldAt (see line 380) if animated)
      const attrs = {};                                                           // initializes an empty object to store the extracted vertex attributes (position, normal, etc.)
      
      for (const key in prim.attributes) {                                        // for each attribute defined for this primitive ('POAITION', 'NORMAL', etc.)
        attrs[key.toLowerCase()] = this.getAccessorData(prim.attributes[key]);    // calls getAccessorData (see line 91) to get the typed array and stores it in the attrs object with a lowercase key
      }
      
      const indices = this.getAccessorData(prim.indices);            // calls getAccessorData (see line 91) to get the typed array for the vertex indices
      const matIdx = prim.material ?? -1;                            // gets the index of the material used by this primitive, defaulting to -1 (no material)
      let baseColorFactor = [1, 1, 1, 1];                            // initializes the base color to a default of white
      let image = null;
      
      if (matIdx >= 0) {                                             // if a material is assigned
        const matDef = this.json.materials[matIdx];                  // retrieves the full material definition from the JSON
        const p = matDef && matDef.pbrMetallicRoughness || {};       // accesses the PBR properties of the material if defined
        baseColorFactor = p.baseColorFactor || [1, 1, 1, 1];         // updates baseColorFactor with the value from the material definition
        
        if (p.baseColorTexture) {                                    // if the material has a base color texture
          const tex = this.json.textures[p.baseColorTexture.index];  // retrieves the texture definition
          const img = this.json.images[tex.source];                  // retrieves the image definition associated with the texture
          
          if (img && img.bufferView !== undefined) {                 // if the image data is embedded in the binary buffer
            const bv = this.getBufferView(img.bufferView);           // extracts the binary data for the image
            image = new Blob([bv], {type: img.mimeType});            // creates a Blob object from the binary data (it can be read by createImageBitmap or used to create an HTML Image)
            scene._imageBlobs.push(image);                           // adds the blob to the scene's temporary list for later processing
          }
        }
      }
      scene.meshes.push({                                            // pushes the new fully processed mesh object into the scene.meshes array
        nodeIndex,
        positions: attrs.position,
        normals: attrs.normal,
        uvs: attrs.texcoord_0,
        indices,
        worldMatrix,
        material: { baseColorFactor, image }
      });
    };

    const traverse = (nodeIndex) => {
      // Traverses the scene graph
      const node = nodes[nodeIndex];                                     // gets the current node's definition
      if (node.mesh !== undefined) {                                     // if this node is associated with a mesh
        const m = meshes[node.mesh];                                     // gets the mesh definition
        const worldMatrix = this.getNodeWorld(nodeIndex, nodeMatrices);  // calls getNodeWorld (see line 218) to alculate the static world matrix for this node
        for (const prim of m.primitives) {                               // for each primitive (sub-meshe) of the mesh
          pushPrimitive(nodeIndex, prim, worldMatrix);                   // calls pushPrimitive (see line 262) to process and store this primitive
        }
      }
      
      if (node.children) {                                               // if the current node has children
        for (const c of node.children) {
          traverse(c);                                                   // recursively calls itself to process each child node
        }
      }
    };

    for (const root of defaultScene.nodes) {                             // starts the traversal by looping through the root nodes of the default scene
      traverse(root);                                                    // calls traverse (see line 305) for each root node
    }

    // ANIMATIONS PARSING SECTION:
    const anims = this.json.animations || [];                            // gets the array of animation definitions
    const samplers = [];                                                 // initializes an array to hold the processed animation samplers
    // A sampler in glTF describes the time curve of a parameter, it contains:
    // - input = timing array (keyframes)
    // - output = array of corresponding values (positions, rotations, scales, target morph weights)
    // - interpolation = type of interpolation between keyframes (LINEAR, STEP, CUBICSPLINE, for the flower it was set to LINEAR in blender)
    const channels = [];                                                 // initializes an array to hold the processed animation channels
    // A channel indicates where to apply the values produced by a sampler, it contains:
    // - sampler = index to the sampler that generates the values
    // - target:
    //    - node = index of the node to animate (e.g. the petal)
    //    - path = what we are animating (translation, rotation, scale, weights)
    
    for (let ai = 0; ai < anims.length; ai++) {                          // for each animation defined in the glTF file
      const sArr = anims[ai].samplers.map(s => ({                        // maps over its samplers array to create a new array of processed samplers
        input: this.getAccessorData(s.input),                            // extracts the keyframe times (input)
        output: this.getAccessorData(s.output),                          // extracts the keyframe values (output)
        interp: s.interpolation || "LINEAR"                              // stores the interpolation mode, defaulting to "LINEAR"
      }));
      samplers.push(sArr);                                               // pushes the array of processed samplers for this animation into the main samplers array
      
      for (const ch of anims[ai].channels) {                             // for each channel of the current animation
        channels.push({                                                  // pushes the processed channel into the main channels array
          animIndex: ai,
          samplerIndex: ch.sampler,
          targetNode: ch.target.node,
          path: ch.target.path
        });
      }
    }
    scene.animations = { samplers, channels };                           // stores the processed samplers and channels in the main scene object

    // Calculate and save the maximum animation duration (in seconds) considering all samplers in all animations
    let maxDur = 0;
    for (let ai = 0; ai < samplers.length; ai++) {                       // for each processed animation
      const sArr = samplers[ai];                                         // gets the samplers for the current animation
      for (let si = 0; si < sArr.length; si++) {                         // for each sampler
        const times = sArr[si].input;                                    // gets the array of keyframe times
        if (times && times.length) {                                     // if the times array is valid
          const dur = times[times.length - 1];                           // the duration of this sampler is the time of the last keyframe
          if (dur > maxDur) maxDur = dur;                                // updates maxDur if the current duration is greater
        }
      }
    }
    scene.animationDuration = maxDur;                                    // stores the calculated maximum duration in the scene object for use by the renderer

    // Animator
    const self = this;                                                   // stores a reference to the GLTFParser instance so it can be accessed inside the nested animator functions
    this.ensureParents();                                                // calls ensureParents (see line 203) to make sure the parent lookup table is built before defining the animator

    scene.animator = {
      _cache: null,                                                      // this will be used to store calculated matrices within a single frame's animation update to avoid re-computation
      getNodeWorldAt(t, nodeIndex){
        //  It calculates the world matrix for a given nodeIndex at a specific time t
        // memoize per-call
        if (!this._cache) this._cache = new Map();                       // initializes the per-call cache if it doesn't exist yet
        const key = nodeIndex|0;                                         // creates a simple key for the cache map from the node's index
        if (this._cache.has(key)) return this._cache.get(key);           // if the matrix for this node at this time has already been calculated, return it from the cache

        // default TRS
        const n = self.json.nodes[nodeIndex];                            // gets the definition for the current node
        let T = n.translation ? n.translation.slice() : [0,0,0];         // gets the node's default translation, creating a copy to avoid modifying the original JSON
        let R = n.rotation ? n.rotation.slice() : [0,0,0,1];             // gets the node's default rotation
        let S = n.scale ? n.scale.slice() : [1,1,1];                     // gets the node's default scale

        // apply channels
        const {samplers, channels} = scene.animations;
        for (const ch of channels) {
          if (ch.targetNode !== nodeIndex) continue;                     // skips this channel if it doesn't target the current node
          const samp = samplers[ch.animIndex][ch.samplerIndex];          // gets the sampler data associated with this channel
          const times = samp.input;                                      // gets the keyframe times from the sampler
          if (!times || times.length === 0) continue;                    // skips if there are no keyframes
          const duration = times[times.length - 1];                      // gets the duration of this specific animation track
          const tt = (duration > 0) ? (t % duration) : 0;                // calculates the local time tt within the animation loop using the modulo operator

          let i = 0; while (i < times.length - 1 && tt > times[i + 1]) i++;      // finds the index i of the keyframe that comes just before the current time tt
          const t0 = times[i], t1 = times[Math.min(i + 1,times.length - 1)];     // gets the times of the two keyframes to interpolate between (t0 and t1)
          const a = (t1 === t0) ? 0 : (tt - t0) / (t1 - t0);                     // calculates the interpolation factor a (from 0 to 1) between the two keyframes

          // Find keyframes i and i+1
          if (ch.path === "translation" || ch.path === "scale") {                // handles animation for translation and scale properties
            const stride = 3;                                                    // VEC3: 3 components for the keyframe (x,y,z)
            const idx0 = i * stride;                                             // offset for keyframe i
            const idx1 = Math.min(i + 1, times.length - 1) * stride;             // offset for keyframe i+1 (clamp to last), Math.min avoids exiting the array when i is the last keyframe
            
            const readV3 = (arr, ofs) => [ arr[ofs], arr[ofs+1], arr[ofs+2] ];   // small helper to read a 3-component vector from an array at a given offset

            const A = readV3(samp.output, idx0);                                 // value at keyframe i
            const B = readV3(samp.output, idx1);                                 // value at keyframe i+1
            const V = GLTFParser.v3lerp(A, B, a);                                // calls v3lerp (see line 160) to interpolate between A and B witch factor a

            if (ch.path === "translation") T = V;                                // if tha channel is a translation updates T, otherwise it's a scale so updates S
            else S = V;
          }
          else if (ch.path === "rotation") {                                     // handles animation for rotation (quaternion) properties
            const stride = 4;                                                    // VEC4: 4 components for the keyframe (x,y,z,w) (quaternion)
            const idx0 = i * stride;                                             // offset for keyframe i
            const idx1 = Math.min(i + 1, times.length - 1) * stride;             // offset for keyframe i+1 (clamp to last), Math.min avoids exiting the array when i is the last keyframe

            const readQ = (arr, ofs) => [ arr[ofs], arr[ofs+1], arr[ofs+2], arr[ofs+3] ];  // small helper to read a 4-component vector from an array at a given offset

            const A = readQ(samp.output, idx0);                                  // quaternion at keyframe i
            const B = readQ(samp.output, idx1);                                  // quaternion at keyframe i+1
            R = GLTFParser.slerp(A, B, a);                                       // calls slerp (see line 168) to sferically interpolate between A and B witch factor a
          }
        }

        // compose local
        const local = GLTFParser.composeTRS(new Float32Array(16), T, R, S);      // calls composeTRS (see line 141) to composes the final animated local transformation matrix
        // parent world
        if (!self.parents) {
          self.ensureParents();                                                  // calls ensureParents (see line 203) to make sure the parent lookup table is available
        }
        const p = self.parents[nodeIndex];                                       // gets the parent index
        if (p === undefined) {                                                   // if it's a root node
          this._cache.set(key, local);                                           // its animated world matrix is just its animated local matrix (cached and returned)
          return local;
        } 
        else {                                                                   // if it has a parent
          const parentWorld = this.getNodeWorldAt(t, p);                         // it recursively calls itself to get the parent's animated world matrix at the same time t
          const w = new Float32Array(16);
          Mat4.multiply(w, parentWorld, local);                                  // multiplies the parent's world matrix by the node's local matrix to get the final animated world matrix
          this._cache.set(key, w);                                               // the final matrix is cached and returned
          return w;
        }
      }
    }

    if (scene.animations && scene.animations.channels.length>0) {                // re-calculate the total duration of the animation, ensuring it's accurate after all processing
      scene.animationDuration = 0;
      for (const anim of this.json.animations) {
        for (const sampler of anim.samplers) {
          const times = this.getAccessorData(sampler.input);                     // retrieves the times array of that sampler
          // updatee the duration with the maximum between the current duration and the found duration
          scene.animationDuration = Math.max(scene.animationDuration, times[times.length-1]);  // times[times.length-1] corresponds to the last time, i.e. the final time of the channel
        }
      }
    }
    return scene;
  }
}