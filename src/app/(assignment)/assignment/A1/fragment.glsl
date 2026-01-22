/////////////////////////////////////////////////////
//// CS 8803/4803 CGAI: Computer Graphics in AI Era
//// Assignment a: Ray Tracing
/////////////////////////////////////////////////////

// Added to work with M1
precision highp float;
precision highp int;

varying vec2 vUv; // UV (screen) coordinates in [0,1]^2


uniform vec2 iResolution;
uniform float iTime;
uniform int iFrame;

uniform sampler2D floorTex;

#define M_PI 3.1415925585
#define Epsilon 1e-6
vec3 sampleDiffuse(int matId, vec3 p);

//============================================================================
// Primitive data types.
//============================================================================

struct Camera 
{
    vec3 origin;
    vec3 lookAt;
    vec3 up;
    vec3 right;
    float aspectRatio;
};

struct Ray 
{
    vec3 ori;
    vec3 dir;
};

struct Plane 
{
    vec3 n;
    vec3 p;
    int matId;
};

struct Sphere 
{
    vec3 ori;
    float r;
    int matId;
};

struct Box 
{
    vec3 ori;
    vec3 halfWidth;
    mat3 rot;
    int matId;
};

struct Light 
{
    vec3 position;
    vec3 Ia;
    vec3 Id;
    vec3 Is;
};

struct Hit 
{
    float t;
    vec3 p;
    vec3 normal;
    int matId;
};

struct Material 
{
    vec3 ka;   // Ambient coefficient.
    vec3 kd;   // Diffuse coefficient.
    vec3 ks;   // Reflected specular coefficient.
    float shininess; // Shininess of the material.

    vec3 kr;   // Reflected coefficient.
};

//============================================================================
// Global scene data.
//============================================================================
Camera camera;
Light lights[2];
Material materials[6];

Sphere spheres[2];
Box boxes[3];
Plane planes[1];
//////////// Random functions ///////////
float g_seed = 0.;

vec3 gamma2(vec3 col) {
    return vec3(sqrt(col.r), sqrt(col.g), sqrt(col.b));
}

float deg2rad(float deg) 
{
    return deg * M_PI / 180.0;
}

uint base_hash(uvec2 p) 
{
    p = 1103515245U * ((p >> 1U) ^ (p.yx));
    uint h32 = 1103515245U * ((p.x) ^ (p.y >> 3U));
    return h32 ^ (h32 >> 16);
}

void initRand(in vec2 frag_coord, in float time) 
{
    g_seed = float(base_hash(floatBitsToUint(frag_coord))) / float(0xffffffffU) + time;
}

vec2 rand2(inout float seed) 
{
    uint n = base_hash(floatBitsToUint(vec2(seed += .1, seed += .1)));
    uvec2 rz = uvec2(n, n * 48271U);
    return vec2(rz.xy & uvec2(0x7fffffffU)) / float(0x7fffffff);
}

/////////////////////////////////////////
const Hit noHit = Hit(
                 /* time or distance */ -1.0, 
                 /* hit position */     vec3(0), 
                 /* hit normal */       vec3(0), 
                 /* hit material id*/   -1);

Hit hitPlane(const Ray r, const Plane pl) 
{
    Hit hit = noHit;
	
    float t = dot(pl.p - r.ori, pl.n) / dot(r.dir, pl.n);

    if (t <= 0.0)
        return noHit;

    vec3 hitP = r.ori + t * r.dir;
    hit = Hit(t, hitP, pl.n, pl.matId);
    
	return hit;
}

// TODO Step 1.1: Implement the sphere intersection
Hit hitSphere(const Ray r, const Sphere s) 
{
    Hit hit = noHit;
	
    /* your implementation starts */
    // Ref: https://raytracing.github.io/books/RayTracingInOneWeekend.html
    vec3 oc = s.ori - r.ori; // Vector from the ray origin O to sphere center C

    // float a = dot(r.dir, r.dir);
    // float b = -2.0 * dot(r.dir, oc); 
    // float c = dot(oc, oc,) - s.r * s.r
    // float discriminant = b * b - 4 * a * c;

    float a = dot(r.dir, r.dir);
    float h = dot(r.dir, oc); // half of b
    float c = dot(oc, oc) - s.r * s.r;
    float discriminant = h * h - a * c;

    if (discriminant < 0.0) {
        return hit;
    }

    float sqrtd = sqrt(discriminant);

    // Two roots of the quadratic equation
    // (-h +- sqrt(d)) / a
    float root1 = (h - sqrtd) / a;
    float root2 = (h + sqrtd) / a;

    // Pick smallest positive root
    float t = root1;
    if (t <= Epsilon) t = root2;
    if (t <= Epsilon) return hit;

    // Hit point
    vec3 hitP = r.ori + t * r.dir;

    // Hit normal
    vec3 hitNormal = normalize(hitP - s.ori);

    // First arg: time or distance
    // Second arg: hit position vec3
    // Third arg: hit normal vect3
    // Fourth arg: hit material

    hit = Hit(t, hitP, hitNormal, s.matId);

	/* your implementation ends */
    
	return hit;
}

// TODO Step 1.2: Implement the box intersection
Hit hitBox(const Ray r, const Box b) 
{
    Hit hit = noHit;
	
    /* your implementation starts */

    /// Step 2/2

    // r.ori: Position of ori relative to world origin resolved in world frame
    // b.ori: Position of box center relative to world origin resolved in world frame
    // r.ori - b.ori = Position of ray relative to box origin resolved in world frame
    // b.rot is rotation matrix from box frame to world frame

    // Resolve the ray origin and direction in the box coordinate system
    mat3 R_WorldToBox = transpose(b.rot);
    vec3 o = R_WorldToBox * (r.ori - b.ori);
    vec3 d = R_WorldToBox * r.dir;

    /// Step 1/2

    // Find the bounding box of the box in box frame
    vec3 bMin = -b.halfWidth; // [b_x_min, b_y_min, b_z_min]
    vec3 bMax = b.halfWidth; // [b_x_max, b_y_max, b_z_max]

    // Compute the intersection inverals along each axis
    // p = o + td, t > 0
    // p_x \in [b_x_min, b_x_max]
    // p_y \in [b_y_min, b_y_max]
    // p_z \in [b_z_min, b_z_max]
    //
    // For example, consider p_x
    // b_x_min <= p_x <= b_x_max
    // b_x_min <= o_x + t d_x <= b_x_max
    // (b_x_min - o_x ) / d_x <= t <= (b_x_max - o_x) / d_x

    // Valid region for t
    float tMax = 1e8;
    float tMin = -1e8;

    // Surface normals
    vec3 nMin = vec3(0.0);
    vec3 nMax = vec3(0.0);

    // X-axis
    // Case when ray X is parallel to X-axis
    if (abs(d.x) <= 1e-8) {
        if (o.x < bMin.x || o.x > bMax.x) {
            // No hit
            return hit;
        } 
    } else {
        float txMin = (bMin.x - o.x) / d.x;
        float txMax = (bMax.x - o.x) / d.x;

        vec3 nxMin = vec3(-1.0, 0.0, 0.0);
        vec3 nxMax = vec3(1.0, 0.0, 0.0);

        // Negative ray direction
        if (txMin > txMax) {
            float tmp = txMin;
            txMin = txMax;
            txMax = tmp;

            vec3 ntmp = nxMin;
            nxMin = nxMax;
            nxMax = ntmp;
        }

        if (txMin > tMin) {
            tMin = txMin;
            nMin = nxMin;
        }

        if (txMax < tMax) {
            tMax = txMax;
            nMax = nxMax;
        }

        if (tMin > tMax) return hit; // No hit
    }

    // Y-axis
    // Case when ray Y is parallel to Y-axis
    if (abs(d.y) <= 1e-8) {
        if (o.y < bMin.y || o.y > bMax.y) {
            // No hit
            return hit;
        } 
    } else {
        float tyMin = (bMin.y - o.y) / d.y;
        float tyMax = (bMax.y - o.y) / d.y;

        vec3 nyMin = vec3(0.0, -1.0, 0.0);
        vec3 nyMax = vec3(0.0, 1.0, 0.0);

        // Negative ray direction
        if (tyMin > tyMax) {
            float tmp = tyMin;
            tyMin = tyMax;
            tyMax = tmp;

            vec3 ntmp = nyMin; 
            nyMin = nyMax; 
            nyMax = ntmp;
        }

        if (tyMin > tMin) { 
            tMin = tyMin; 
            nMin = nyMin; 
        } 
        
        if (tyMax < tMax) { 
            tMax = tyMax; 
            nMax = nyMax; 
        }

        if (tMin > tMax) return hit; // No hit
    }

    // Z-axis
    // Case when ray Z is parallel to Z-axis
    if (abs(d.z) <= 1e-8) {
        if (o.z < bMin.z || o.z > bMax.z) {
            // No hit
            return hit;
        } 
    } else {
        float tzMin = (bMin.z - o.z) / d.z;
        float tzMax = (bMax.z - o.z) / d.z;

        vec3 nzMin = vec3(0.0, 0.0, -1.0); 
        vec3 nzMax = vec3(0.0, 0.0,  1.0);

        // Negative ray direction
        if (tzMin > tzMax) {
            float tmp = tzMin;
            tzMin = tzMax;
            tzMax = tmp;

            vec3 ntmp = nzMin;
            nzMin = nzMax;
            nzMax = ntmp;
        }

        if (tzMin > tMin) { 
            tMin = tzMin;
            nMin = nzMin; 
        } 
        
        if (tzMax < tMax) {
            tMax = tzMax;
            nMax = nzMax; 
        }

        if (tMin > tMax) return hit; // No hit
    }

    if (tMax <= 0.0) {
        // Whole intersection is behind ray
        return hit;
    }

    float t = (tMin > 0.0) ? tMin : tMax; // closest valid t

    // Hit point
    vec3 hitPinBoxFrame = o + t * d;
    vec3 hitPinWorldFrame = b.ori + (b.rot * hitPinBoxFrame);

    // Hit normal
    vec3 hitNormalinBoxFrame = (tMin > 0.0) ? nMin : nMax;
    vec3 hitNormal = normalize(b.rot * hitNormalinBoxFrame);

    // First arg: time or distance
    // Second arg: hit position vec3
    // Third arg: hit normal vect3
    // Fourth arg: hit material

    hit = Hit(t, hitPinWorldFrame, hitNormal, b.matId);

	/* your implementation ends */
    
	return hit;
}

Hit findHit(Ray r) 
{
    Hit h = noHit;
    
	for(int i = 0; i < spheres.length(); i++) {
        Hit tempH = hitSphere(r, spheres[i]);
        if(tempH.t > Epsilon && (h.t < 0. || h.t > tempH.t))
            h = tempH;
    }
	
    for(int i = 0; i < planes.length(); i++) {
        Hit tempH = hitPlane(r, planes[i]);
        if(tempH.t > Epsilon && (h.t < 0. || h.t > tempH.t))
            h = tempH;
    }

    for(int i = 0; i < boxes.length(); i++) {
        Hit tempH = hitBox(r, boxes[i]);
        if(tempH.t > Epsilon && (h.t < 0. || h.t > tempH.t))
            h = tempH;
    }

    return h;
}

// TODO Step 2: Implement the Phong shading model
vec3 shading_phong(Light light, int matId, vec3 e, vec3 p, vec3 s, vec3 n) 
{
	//// default color: return dark red for the ground and dark blue for spheres
    vec3 color = matId == 0 ? vec3(0.2, 0, 0) : vec3(0, 0, 0.3);
	
    /* your implementation starts */
    // 3 points: s (light source point), e (eye point), p (point on surface)
    // 4 vectors: l (light direction), v (viewer direction), n (normal), r (reflection direction)

    // Phong shading model:
    // L_phong = \sum_{j \in light} (k_a I^j_a + k^j_d max(0, 1^j \dot n) + k_s I^j_s max(0, v \dot r)^p)

    vec3 ka = materials[matId].ka;
    vec3 kd = materials[matId].kd;
    vec3 ks = materials[matId].ks;

    vec3 Ia = light.Ia;
    vec3 Id = light.Id;
    vec3 Is = light.Is;

    vec3 l = normalize(s - p);
    vec3 v = normalize(e - p);
    vec3 r = reflect(-l, normalize(n));

    float shininess = materials[matId].shininess;

    vec3 ambient = ka * Ia;
    vec3 lambertian = kd * Id * max(0.0, dot(l, n));
    vec3 specular = ks * Is * pow(max(0.0, dot(v, r)), shininess);

    color = ambient + lambertian + specular;
	/* your implementation ends */
    
	return color;
}

// TODO Step 3: Implement the shadow test 
bool isShadowed(Light light, Hit h) 
{
    bool shadowed = false;
	
    /* your implementation starts */
    
	/* your implementation ends */
    
	return shadowed;
}

// TODO Step 4: Implement the texture mapping
vec3 sampleDiffuse(int matId, vec3 p) 
{
    if(matId == 0) {
        vec3 color = materials[matId].kd;
		
        /* your implementation starts */
        
		/* your implementation ends */
        
		return color;
    }
    return materials[matId].kd;
}

vec3 rayTrace(in Ray r, out Hit hit) 
{
    vec3 col = vec3(0);
    Hit h = findHit(r);
    hit = h;
    if(h.t > 0. && h.t < 1e8) {
        // shading
        for(int i = 0; i < lights.length(); i++) {
            if(isShadowed(lights[i], h)) {
                col += materials[h.matId].ka * lights[i].Ia;
            } else {
                vec3 e = camera.origin;
                vec3 p = h.p;
                vec3 s = lights[i].position;
                vec3 n = h.normal;
                col += shading_phong(lights[i], h.matId, e, p, s, n);
            }
        }
    }
    return col;
}

Ray getPrimaryRay(vec2 uv) 
{
    return Ray(camera.origin, 
               normalize(camera.lookAt + 
                        (uv.x - 0.5) * camera.right * camera.aspectRatio + 
                        (uv.y - 0.5) * camera.up));
}

mat3 getRotXYZ(float pitch, float yaw, float roll) 
{
    
    mat3 rotX = mat3(
        vec3(1, 0, 0),
        vec3(0, cos(pitch), sin(pitch)),
        vec3(0, -sin(pitch), cos(pitch))
    );
    mat3 rotY = mat3(
        vec3(cos(yaw), 0, -sin(yaw)),
        vec3(0, 1, 0),
        vec3(sin(yaw), 0, cos(yaw))
    );
    mat3 rotZ = mat3(
        vec3(cos(roll), sin(roll), 0),
        vec3(-sin(roll), cos(roll), 0),
        vec3(0, 0, 1)
    );
    
    return rotZ * rotY * rotX;
}

void initScene() 
{
    float aspectRatio = iResolution.x / iResolution.y;
    vec3 origin = vec3(0., 2.8, 3);
    vec3 lookAt = normalize(vec3(0.,0.45,0.) - origin);
    vec3 up = vec3(0, 1, 0);
    vec3 right = normalize(cross(lookAt, up));
    up = normalize(cross(right, lookAt));
    camera = Camera(origin, lookAt, up, right, aspectRatio);

    // Floor Material 
    materials[0].ka = vec3(0.05);
    materials[0].kd = vec3(0.5);
    materials[0].ks = vec3(0.8);
    materials[0].shininess = 10.0;
    materials[0].kr = 0.3 * materials[0].ks;

    materials[1].ka = vec3(0.0);
    materials[1].kd = vec3(0.0);
    materials[1].ks = vec3(0.95);
    materials[1].shininess = 512.;
    materials[1].kr = 0.8 * materials[1].ks;

    materials[2].ka = vec3(0.0);
    materials[2].kd = vec3(0.5);
    materials[2].ks = vec3(0.5);
    materials[2].shininess = 128.;
    materials[2].kr = 0.5 * materials[2].ks;

    materials[3].ka = vec3(0.0);
    materials[3].kd = vec3(13, 71, 161) / 255.;
    materials[3].ks = vec3(0.3);
    materials[3].shininess = 128.;
    materials[3].kr = 0.4 * materials[3].ks;

    materials[4].ka = vec3(0.0);
    materials[4].kd = vec3(183, 28, 28) / 255.;
    materials[4].ks = 1.2 * materials[4].kd;
    materials[4].shininess = 128.;
    materials[4].kr = 0.6 * materials[4].ks;

    materials[5].ka = vec3(0.0);
    materials[5].kd = vec3(27, 94, 32) / 255.;
    materials[5].ks = 0.2 * materials[5].kd;
    materials[5].shininess = 128.;
    materials[5].kr = 0.5 * materials[5].ks;

    lights[0] = Light(vec3(-4., 5., 2.5), 
                            /*Ia*/ vec3(0.1, 0.1, 0.1), 
                            /*Id*/ vec3(1.0, 1.0, 1.0), 
                            /*Is*/ vec3(0.8, 0.8, 0.8));
    lights[1] = Light(vec3(1.5, 4., 3.), 
                            /*Ia*/ vec3(0.1, 0.1, 0.1), 
                            /*Id*/ vec3(0.9, 0.9, 0.9), 
                            /*Is*/ vec3(0.5, 0.5, 0.5));
    planes[0] = Plane(vec3(0., 1., 0.), vec3(0., 0., 0.), 0);

    spheres[0] = Sphere(vec3(0., 1.5, 0.), 0.5, 1);
    spheres[1] = Sphere(vec3(-0.6, 0.4, 1.1), 0.4, 2);

    boxes[0] = Box(vec3(0., 0.5, 0.), vec3(0.5), getRotXYZ(0., 0., 0.), 3);
    boxes[1] = Box(vec3(-1.2, 0.85, 0.0), vec3(0.4, 0.85, 0.4), getRotXYZ(0., 0.4 * M_PI, 0.), 4);
    boxes[2] = Box(vec3(0.8, 0.3, 0.8), vec3(0.75, 0.3, 0.3), getRotXYZ(0., 0.2 * M_PI, 0.), 5);
}

// TODO Step 5: Change the value of numberOfSampling to 50

/* your implementation starts */

const int numberOfSampling = 1;

/* your implementation ends */

void main() 
{
    initScene();
    initRand(gl_FragCoord.xy, iTime); 
    vec2 uv = gl_FragCoord.xy / iResolution.xy;

    vec3 resultCol = vec3(0.);
    vec3 compounded_kr = vec3(1.0); // cumulative reflection coefficient

    Ray recursive_ray = getPrimaryRay(uv + rand2(g_seed) / iResolution.xy);
    for(int i = 0; i < numberOfSampling; i++) {
        Hit hit;
        vec3 col = rayTrace(recursive_ray, hit);

        resultCol += compounded_kr * col;

        if(hit.t < 0.0 || hit.t > 1e8 || length(compounded_kr) < 0.001)
            break;

        compounded_kr *= materials[hit.matId].kr;
		
        // TODO Step 5: Define the reflected ray and assign this ray to recursive_ray
        
		/* your implementation starts */
        
		/* your implementation ends */
    }

    resultCol = gamma2(resultCol);
    gl_FragColor = vec4(resultCol, 1.);
}