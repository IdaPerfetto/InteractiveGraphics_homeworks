var raytraceFS = `
struct Ray {
	vec3 pos;  // origin of the ray
	vec3 dir;  // direction of the ray
};

struct Material {
	vec3  k_d;	// diffuse coefficient
	vec3  k_s;	// specular coefficient
	float n;	// specular exponent
};

struct Sphere {         // scene object
	vec3     center;
	float    radius;
	Material mtl;
};

struct Light {
	vec3 position;  // position of the light
	vec3 intensity;  // RGB brightness values
};

struct HitInfo {       // stores data about a ray-object intersection
	float    t;
	vec3     position;  // position of the hit point
	vec3     normal;  // normal surface
	Material mtl;  // material at the intersection
};

uniform Sphere spheres[ NUM_SPHERES ];
uniform Light  lights [ NUM_LIGHTS  ];
uniform samplerCube envMap;
uniform int bounceLimit;

// check if ray hits any sphere in the scene
bool IntersectRay( inout HitInfo hit, Ray ray );  // inout HitInfo hit gets filled with data from the closest hit

// Shades the given point and returns the computed color.
vec3 Shade( Material mtl, vec3 position, vec3 normal, vec3 view ){
    // Inputs: 
	// material at the hit point (mtl), 
	// world-space position where the ray hit the object (position) and surface normal at that position (normal), 
	// direction pointing towards the viewer or camera (view)
	// Output:
	// RGB color at that point

	// start with black 
	vec3 color = vec3(0,0,0);

    // accumulate light contributions from each point light source
	for ( int i=0; i<NUM_LIGHTS; ++i ) {
		vec3 lightDir = lights[i].position - position;  // vector from the surface point to the light
		float lightDist = length(lightDir);  // used to compare against shadow hits
		lightDir = normalize(lightDir);  // normalization for correct lighting math

		// Cast a shadow ray to the light
		Ray shadowRay;
		shadowRay.pos = position + 0.001 * normal;  // avoid self-intersection
		shadowRay.dir = lightDir;

		HitInfo shadowHit;
		bool shadowed = false;

		// if the ray hits something before reaching the light, the point is in shadow
		if (IntersectRay(shadowHit, shadowRay))
		{
			if (shadowHit.t < lightDist)  // ensure the blocker is between surface and light
				shadowed = true;
		}

		// only compute lighting if the point is not in shadow
		if (!shadowed)
		{
			// Diffuse term
			float NdotL = max(dot(normal, lightDir), 0.0);
			vec3 diffuse = mtl.k_d * lights[i].intensity * NdotL;  // reflects light evenly in all directions, k_d is the diffuse coefficient

			// Specular term (Blinn-Phong)
			vec3 h = normalize(lightDir + view);  // halfway vector between the light direction and the view direction
			float NdotH = max(dot(normal, h), 0.0);
			vec3 specular = mtl.k_s * lights[i].intensity * pow(NdotH, mtl.n);  // k_s is the specular coefficient 

			color += diffuse + specular;  // add contribution from this light to the total
		}
	}
	return color;  // final shaded color from all light sources
}

// Intersects the given ray with all spheres in the scene
// Updates the given HitInfo using the information of the sphere that first intersects with the ray
// Returns true if an intersection is found
bool IntersectRay( inout HitInfo hit, Ray ray ){
	
    hit.t = 1e30;  // initialize hit.t to a very large number, we'll search for the smallest valid t (closest intersection)
	bool foundHit = false;  // tracks whether we hit anything
	vec3 d = ray.dir;
	vec3 p = ray.pos;
	float a = dot(d, d);  // a is part of the quadratic equation for ray-sphere intersection

	for ( int i=0; i<NUM_SPHERES; ++i ) {  // for each sphere
		
	    vec3 c = spheres[i].center;  // c is its center
		float r = spheres[i].radius;
		vec3 oc = p - c;  // oc is the vector from the sphere center to the ray origin

		float b = 2.0 * dot(d, oc);
		float c_term = dot(oc, oc) - r * r;
		float discr = b * b - 4.0 * a * c_term;  // if the discriminant is lower than 0 there's no real root -> no intersection

		if (discr >= 0.0)  // if there is an intersection
		{
		    // compute the smaller root (nearest point)
			float t;
			if (a != 0.0)
				t = (-b - sqrt(discr)) / (2.0 * a);
			else
				t = -c_term / b;  // fallback to linear

			if (t > 0.0 && t < hit.t) // only accept positive t values (in front of camera) and keep the closest hit only (i.e., t < hit.t)
			{
			    // populate hit with the intersection data
				hit.t = t;
				hit.position = p + t * d;
				hit.normal = normalize(hit.position - c);  // normal is the direction from the center to the hit point
				hit.mtl = spheres[i].mtl;
				foundHit = true;
			}
		}
	}
	return foundHit;
}

// Given a ray, returns the shaded color where the ray intersects a sphere
// If the ray does not hit a sphere, returns the environment color
vec4 RayTracer( Ray ray )
{
	HitInfo hit;
	if ( IntersectRay( hit, ray ) ) {
	// Input: a ray shot from the camera (or reflection)
	// Output: an RGBA color value (vec4) for a pixel
	
		vec3 view = normalize( -ray.dir );  // direction pointing from hit point to the camera
		vec3 clr = Shade( hit.mtl, hit.position, hit.normal, view );  // base color from direct illumination (diffuse + specular)
		
		// Compute reflections
		vec3 k_s = hit.mtl.k_s;  // current reflection intensity (starts with the k_s of the material)
		for ( int bounce=0; bounce<MAX_BOUNCES; ++bounce ) {
			if ( bounce >= bounceLimit ) break;
			if ( hit.mtl.k_s.r + hit.mtl.k_s.g + hit.mtl.k_s.b <= 0.0 ) break; // if reflectivity is  zero (black) stop
			
			Ray r;	// this is the reflection ray
			r.pos = hit.position + 0.001 * hit.normal;  // a tiny step away from the surface to avoid self-hit
			r.dir = reflect(-view, hit.normal);  // mirror reflection of the view direction across the surface normal
			HitInfo h;	// reflection hit info
			
			if ( IntersectRay( h, r ) ) {  // if the reflected ray hits another object

				view = normalize(-r.dir);
				vec3 reflection = Shade(h.mtl, h.position, h.normal, view);  // compute the color at the reflection hit point
				clr += k_s * reflection;  // add its contribution to the total color modulated by k_s

				k_s *= h.mtl.k_s; // dimish k_s: a reflective surface bouncing off another reflective surface produces a weaker reflection
				hit = h;  // set hit to the new point for the next bounce
			} 
			else {
				// The reflection ray did not intersect with anything,
				// so we are using the environment color
				clr += k_s * textureCube( envMap, r.dir.xzy ).rgb;  // multiply by current k_s to scale reflection strength
				break;	// no more reflections
			}
		}
		return vec4( clr, 1 );	// return the accumulated color, including the reflections
	} else {
		return vec4( textureCube( envMap, ray.dir.xzy ).rgb, 0 );	// return the environment color
	}
}
`;
