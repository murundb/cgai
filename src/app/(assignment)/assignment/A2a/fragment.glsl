/////////////////////////////////////////////////////
//// CS 8803/4803 CGAI: Computer Graphics in AI Era
//// Assignment 2A: SDF and Ray Marching
/////////////////////////////////////////////////////

// Added to work with M1
precision highp float;              //// set default precision of float variables to high precision
precision highp int;

varying vec2 vUv;                   //// screen uv coordinates (varying, from vertex shader)
uniform vec2 iResolution;           //// screen resolution (uniform, from CPU)
uniform float iTime;                //// time elapsed (uniform, from CPU)

const vec3 CAM_POS = vec3(-0.35, 1.0, -3.0);
float sdf2(vec3 p);

/////////////////////////////////////////////////////
//// sdf functions
/////////////////////////////////////////////////////

/////////////////////////////////////////////////////
//// Step 1: sdf primitives
//// You are asked to implement sdf primitive functions for sphere, plane, and box.
//// In each function, you will calculate the sdf value based on the function arguments.
/////////////////////////////////////////////////////

//// sphere: p - query point; c - sphere center; r - sphere radius
float sdfSphere(vec3 p, vec3 c, float r)
{
    //// your implementation starts
    float sdf = distance(p, c) - r;
    return sdf;
    //// your implementation ends
}

//// plane: p - query point; h - height
float sdfPlane(vec3 p, float h)
{
    //// your implementation starts
    // The plane is on XZ since it is described by its height
    float sdf = p.y - h;
    return sdf;
    //// your implementation ends
}

//// box: p - query point; c - box center; b - box half size (i.e., the box size is (2*b.x, 2*b.y, 2*b.z))
float sdfBox(vec3 p, vec3 c, vec3 b)
{
    //// your implementation starts
    // First compute the d vector
    vec3 d = abs(p - c) - b;

    float outsideDist = length(max(d, 0.0));
    float insideDist = min(max(d.x, max(d.y, d.z)), 0.0);

    float sdf = outsideDist + insideDist;
    return sdf;
    
    //// your implementation ends
}

/////////////////////////////////////////////////////
//// boolean operations
/////////////////////////////////////////////////////

/////////////////////////////////////////////////////
//// Step 2: sdf boolean operations
//// You are asked to implement sdf boolean operations for intersection, union, and subtraction.
/////////////////////////////////////////////////////

float sdfIntersection(float s1, float s2)
{
    //// your implementation starts
    float s = max(s1, s2);
    return s;

    //// your implementation ends
}

float sdfUnion(float s1, float s2)
{
    //// your implementation starts
    float s = min(s1, s2);
    return s;

    //// your implementation ends
}

float sdfSubtraction(float s1, float s2)
{
    //// your implementation starts
    float s = max(s1, -s2);
    return s;

    //// your implementation ends
}

/////////////////////////////////////////////////////
//// sdf calculation
/////////////////////////////////////////////////////

/////////////////////////////////////////////////////
//// Step 3: scene sdf
//// You are asked to use the implemented sdf boolean operations to draw the following objects in the scene by calculating their CSG operations.
/////////////////////////////////////////////////////

//// sdf: p - query point
float sdf(vec3 p)
{
    float s = 0.;

    //// 1st object: plane
    float plane1_h = -0.1;
    
    //// 2nd object: sphere
    vec3 sphere1_c = vec3(-2.0, 1.0, 0.0);
    float sphere1_r = 0.25;

    //// 3rd object: box
    vec3 box1_c = vec3(-1.0, 1.0, 0.0);
    vec3 box1_b = vec3(0.2, 0.2, 0.2);

    //// 4th object: box-sphere subtraction
    vec3 box2_c = vec3(0.0, 1.0, 0.0);
    vec3 box2_b = vec3(0.3, 0.3, 0.3);

    vec3 sphere2_c = vec3(0.0, 1.0, 0.0);
    float sphere2_r = 0.4;

    //// 5th object: sphere-sphere intersection
    vec3 sphere3_c = vec3(1.0, 1.0, 0.0);
    float sphere3_r = 0.4;

    vec3 sphere4_c = vec3(1.3, 1.0, 0.0);
    float sphere4_r = 0.3;

    //// calculate the sdf based on all objects in the scene
    
    //// your implementation starts
    s = 1e9;

    // 1st object: plane
    float sdfObject1 = sdfPlane(p, plane1_h);
    s = min(s, sdfObject1);

    // 2nd object: sphere
    float sdfObject2 = sdfSphere(p, sphere1_c, sphere1_r);
    s = min(s,sdfObject2 );

    // 3rd object: box
    float sdfObject3 = sdfBox(p, box1_c, box1_b);
    s = min(s,sdfObject3 );

    // 4th object: box-sphere subtraction
    float sdf_box2 = sdfBox(p, box2_c, box2_b);
    float sdf_sphere2 = sdfSphere(p, sphere2_c, sphere2_r);
    float sdfObject4 = sdfSubtraction(sdf_box2, sdf_sphere2);
    s = min(s,sdfObject4 );

    // 5th object: sphere-sphere intersection
    float sdf_sphere3 = sdfSphere(p, sphere3_c, sphere3_r);
    float sdf_sphere4 = sdfSphere(p, sphere4_c, sphere4_r);
    float sdfObject5 = sdfIntersection(sdf_sphere3, sdf_sphere4);
    s = min(s,sdfObject5);
    //// your implementation ends

    return s;
}

/////////////////////////////////////////////////////
//// ray marching
/////////////////////////////////////////////////////

/////////////////////////////////////////////////////
//// Step 4: ray marching
//// You are asked to implement the ray marching algorithm within the following for-loop.
/////////////////////////////////////////////////////

//// ray marching: origin - ray origin; dir - ray direction 
float rayMarching(vec3 origin, vec3 dir)
{
    float s = 0.0;
    for(int i = 0; i < 100; i++)
    {
        //// your implementation starts
        vec3 p = origin + s * dir;
        float ds = sdf(p);

        if (ds < 1e-3) {
            break; // hit
        }

        s += ds;

        if (s > 1e2) {
            break;
        }
        //// your implementation ends
    }
    return s;
}

/////////////////////////////////////////////////////
//// normal calculation
/////////////////////////////////////////////////////

/////////////////////////////////////////////////////
//// Step 5: normal calculation
//// You are asked to calculate the sdf normal based on finite difference.
/////////////////////////////////////////////////////

//// normal: p - query point
vec3 normal(vec3 p)
{
    // float s = sdf(p);          //// sdf value in p
    float dx = 0.01;           //// step size for finite difference

    //// your implementation starts

    // Perturb in x
    vec3 p_plus_dx = p + vec3(1.0, 0.0, 0.0) * dx; 
    vec3 p_minus_dx = p - vec3(1.0, 0.0, 0.0) * dx;
    float s_plus_dx = sdf(p_plus_dx);
    float s_minus_dx = sdf(p_minus_dx);
    float grad_x = (s_plus_dx - s_minus_dx) / (2.0 * dx);

    // Perturb in y
    vec3 p_plus_dy = p + vec3(0.0, 1.0, 0.0) * dx; 
    vec3 p_minus_dy = p - vec3(0.0, 1.0, 0.0) * dx;
    float s_plus_dy = sdf(p_plus_dy);
    float s_minus_dy = sdf(p_minus_dy);
    float grad_y = (s_plus_dy - s_minus_dy) / (2.0 * dx);

    // Perturb in z
    vec3 p_plus_dz = p + vec3(0.0, 0.0, 1.0) * dx; 
    vec3 p_minus_dz = p - vec3(0.0, 0.0, 1.0) * dx;
    float s_plus_dz = sdf(p_plus_dz);
    float s_minus_dz = sdf(p_minus_dz);
    float grad_z = (s_plus_dz - s_minus_dz) / (2.0 * dx);

    vec3 n = normalize(vec3(grad_x, grad_y, grad_z));

    return n;

    //// your implementation ends
}

/////////////////////////////////////////////////////
//// Phong shading
/////////////////////////////////////////////////////

/////////////////////////////////////////////////////
//// Step 6: lighting and coloring
//// You are asked to specify the color for each object in the scene.
//// Each object must have a separate color without mixing.
//// Notice that we have implemented the default Phong shading model for you.
/////////////////////////////////////////////////////

vec3 phong_shading(vec3 p, vec3 n)
{
    //// background
    if(p.z > 10.0){
        return vec3(0.9, 0.6, 0.2);
    }

    //// phong shading
    vec3 lightPos = vec3(4.*sin(iTime), 4., 4.*cos(iTime));  
    vec3 l = normalize(lightPos - p);               
    float amb = 0.1;
    float dif = max(dot(n, l), 0.) * 0.7;
    vec3 eye = CAM_POS;
    float spec = pow(max(dot(reflect(-l, n), normalize(eye - p)), 0.0), 128.0) * 0.9;

    vec3 sunDir = vec3(0, 1, -1);
    float sunDif = max(dot(n, sunDir), 0.) * 0.2;

    //// shadow
    float s = rayMarching(p + n * 0.02, l);
    if(s < length(lightPos - p)) dif *= .2;

    vec3 color = vec3(1.0, 1.0, 1.0);

    //// your implementation for coloring starts

    // Re-evaluate the SDFS at the hit point to determine which object was hit
    //// 1st object: plane
    float plane1_h = -0.1;
    float sdf_plane = sdfPlane(p, plane1_h);
    
    //// 2nd object: sphere
    vec3 sphere1_c = vec3(-2.0, 1.0, 0.0);
    float sphere1_r = 0.25;
    float sdf_sphere1 = sdfSphere(p, sphere1_c, sphere1_r);

    //// 3rd object: box
    vec3 box1_c = vec3(-1.0, 1.0, 0.0);
    vec3 box1_b = vec3(0.2, 0.2, 0.2);
    float sdf_box1 = sdfBox(p, box1_c, box1_b);

    //// 4th object: box-sphere subtraction
    vec3 box2_c = vec3(0.0, 1.0, 0.0);
    vec3 box2_b = vec3(0.3, 0.3, 0.3);

    vec3 sphere2_c = vec3(0.0, 1.0, 0.0);
    float sphere2_r = 0.4;
    float sdfObject4 = sdfSubtraction(sdfBox(p, box2_c, box2_b), sdfSphere(p, sphere2_c, sphere2_r));

    //// 5th object: sphere-sphere intersection
    vec3 sphere3_c = vec3(1.0, 1.0, 0.0);
    float sphere3_r = 0.4;
    vec3 sphere4_c = vec3(1.3, 1.0, 0.0);
    float sphere4_r = 0.3;
    float sdfObject5 = sdfIntersection(sdfSphere(p, sphere3_c, sphere3_r), sdfSphere(p, sphere4_c, sphere4_r));

    float EPS = 1e-2;

    if (sdf_plane < EPS) {
        color = vec3(0.6, 0.6, 0); // yellowish green
    } else if (sdf_sphere1 < EPS) {
        color = vec3(1.0, 0.0, 0.0); // red
    } else if (sdf_box1 < EPS) {
        color = vec3(0.0, 1.0, 0.0); // green
    } else if (sdfObject4 < EPS) {
        color = vec3(0.0, 0.0, 1.0); // blue
    } else if (sdfObject5 < EPS) {
        color = vec3(0.0, 1.0, 1.0); // cyan
    }

    //// your implementation for coloring ends

    return (amb + dif + spec + sunDif) * color;
}

/////////////////////////////////////////////////////
//// Step 7: creative expression
//// You will create your customized sdf scene with new primitives and CSG operations in the sdf2 function.
//// Call sdf2 in your ray marching function to render your customized scene.
/////////////////////////////////////////////////////

//// sdf2: p - query point
float sdf2(vec3 p)
{
    float s = 0.;

    //// your implementation starts

    //// your implementation ends

    return s;
}

/////////////////////////////////////////////////////
//// main function
/////////////////////////////////////////////////////

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = (fragCoord.xy - .5 * iResolution.xy) / iResolution.y;         //// screen uv
    vec3 origin = CAM_POS;                                                  //// camera position 
    vec3 dir = normalize(vec3(uv.x, uv.y, 1));                              //// camera direction
    float s = rayMarching(origin, dir);                                     //// ray marching
    vec3 p = origin + dir * s;                                              //// ray-sdf intersection
    vec3 n = normal(p);                                                     //// sdf normal
    vec3 color = phong_shading(p, n);                                       //// phong shading
    fragColor = vec4(color, 1.);                                            //// fragment color
}

void main() 
{
    mainImage(gl_FragColor, gl_FragCoord.xy);
}