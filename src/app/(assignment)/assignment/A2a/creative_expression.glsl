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

// const vec3 CAM_POS = vec3(-0.35, 1.0, -3.0); 
const vec3 CAM_POS = vec3(0.0, 0.0, -3.0); 
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

float sdfCutHollowSphere(vec3 p, float r, float h, float t)
{
    // Reference: https://iquilezles.org/articles/distfunctions/
    float w = sqrt(r * r - h * h);
    vec2 q = vec2( length(p.xz), p.y );
    return ((h * q.x < w * q.y) ? length(q - vec2(w,h)) : 
                            abs(length(q) - r) ) - t;
}


float sdfCappedCylinder( vec3 p, float r, float h )
{
  // Reference: https://iquilezles.org/articles/distfunctions/
  vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(r,h);
  return min(max(d.x, d.y), 0.0) + length(max(d,0.0));
}

float sdfCappedCone( vec3 p, vec3 a, vec3 b, float ra, float rb )
{
  // Reference: https://iquilezles.org/articles/distfunctions/
  float rba  = rb-ra;
  float baba = dot(b-a,b-a);
  float papa = dot(p-a,p-a);
  float paba = dot(p-a,b-a)/baba;
  float x = sqrt( papa - paba*paba*baba );
  float cax = max(0.0,x-((paba<0.5)?ra:rb));
  float cay = abs(paba-0.5)-0.5;
  float k = rba*rba + baba;
  float f = clamp( (rba*(x-ra)+paba*baba)/k, 0.0, 1.0 );
  float cbx = x-ra - f*rba;
  float cby = paba - f;
  float s = (cbx<0.0 && cay<0.0) ? -1.0 : 1.0;
  return s*sqrt( min(cax*cax + cay*cay*baba,
                     cbx*cbx + cby*cby*baba) );
}

float sdfTorus( vec3 p, vec2 t )
{
  // Reference: https://iquilezles.org/articles/distfunctions/
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}

float sdfSmoothMin(float s1, float s2)
{
    return s1 + (s2 - s1) / (1.0 - pow(2.0, (s2 - s1) / (0.03 * 0.693)));
}

float sdfSmoothMin(float s1, float s2, float k)
{
    return s1 + (s2 - s1) / (1.0 - pow(2.0, (s2 - s1) / (k * 0.693)));
}

float sdfBlobbyTable(vec3 p, float h)
{
    float size = 0.75;
    float thickness = 0.01;

    vec3 topCenter = vec3(0.0, h - thickness, 0.0);
    float sdf_top = sdfBox(p, topCenter, vec3(size, thickness, size));
    
    float legRadius = 0.04;
    float legHeight = 1.0;
    float legInset = size - 0.1;  // how far from edge
    
    // Leg positions
    vec3 leg1Pos = vec3( legInset, h - legHeight * 0.5 - thickness,  legInset);
    vec3 leg2Pos = vec3(-legInset, h - legHeight * 0.5 - thickness,  legInset);
    vec3 leg3Pos = vec3( legInset, h - legHeight * 0.5 - thickness, -legInset);
    vec3 leg4Pos = vec3(-legInset, h - legHeight * 0.5 - thickness, -legInset);
    
    // Table legs cylinders
    float sdf_leg1 = sdfCappedCylinder(p - leg1Pos, legRadius, legHeight * 0.5);
    float sdf_leg2 = sdfCappedCylinder(p - leg2Pos, legRadius, legHeight * 0.5);
    float sdf_leg3 = sdfCappedCylinder(p - leg3Pos, legRadius, legHeight * 0.5);
    float sdf_leg4 = sdfCappedCylinder(p - leg4Pos, legRadius, legHeight * 0.5);

    // Add spheres along Y for each table
    float halfH = legHeight * 0.5;

    // Leg 1 
    float s1 = sdfSphere(p, leg1Pos + vec3(0.0, -halfH*0.9, 0.0), legRadius*1.8);
    float s2 = sdfSphere(p, leg1Pos + vec3(0.0, -halfH*0.3, 0.0), legRadius*1.4);
    float s3 = sdfSphere(p, leg1Pos + vec3(0.0, halfH*0.3, 0.0), legRadius*1.5);
    float s4 = sdfSphere(p, leg1Pos + vec3(0.0, halfH*0.9, 0.0), legRadius*2.0);
    sdf_leg1 = sdfSmoothMin(sdf_leg1, s1, 0.03);
    sdf_leg1 = sdfSmoothMin(sdf_leg1, s2, 0.03);
    sdf_leg1 = sdfSmoothMin(sdf_leg1, s3, 0.03);
    sdf_leg1 = sdfSmoothMin(sdf_leg1, s4, 0.03);

    // Leg 2
    s1 = sdfSphere(p, leg2Pos + vec3(0.0, -halfH*0.9, 0.0), legRadius*1.8);
    s2 = sdfSphere(p, leg2Pos + vec3(0.0, -halfH*0.3, 0.0), legRadius*1.4);
    s3 = sdfSphere(p, leg2Pos + vec3(0.0, halfH*0.3, 0.0), legRadius*1.5);
    s4 = sdfSphere(p, leg2Pos + vec3(0.0, halfH*0.9, 0.0), legRadius*2.0);
    sdf_leg2 = sdfSmoothMin(sdf_leg2, s1, 0.03);
    sdf_leg2 = sdfSmoothMin(sdf_leg2, s2, 0.03);
    sdf_leg2 = sdfSmoothMin(sdf_leg2, s3, 0.03);
    sdf_leg2 = sdfSmoothMin(sdf_leg2, s4, 0.03);

    // Leg 3
    s1 = sdfSphere(p, leg3Pos + vec3(0.0, -halfH*0.9, 0.0), legRadius*1.8);
    s2 = sdfSphere(p, leg3Pos + vec3(0.0, -halfH*0.3, 0.0), legRadius*1.4);
    s3 = sdfSphere(p, leg3Pos + vec3(0.0, halfH*0.3, 0.0), legRadius*1.5);
    s4 = sdfSphere(p, leg3Pos + vec3(0.0, halfH*0.9, 0.0), legRadius*2.0);
    sdf_leg3 = sdfSmoothMin(sdf_leg3, s1, 0.03);
    sdf_leg3 = sdfSmoothMin(sdf_leg3, s2, 0.03);
    sdf_leg3 = sdfSmoothMin(sdf_leg3, s3, 0.03);
    sdf_leg3 = sdfSmoothMin(sdf_leg3, s4, 0.03);

    // Leg 4
    s1 = sdfSphere(p, leg4Pos + vec3(0.0, -halfH*0.9, 0.0), legRadius*1.8);
    s2 = sdfSphere(p, leg4Pos + vec3(0.0, -halfH*0.3, 0.0), legRadius*1.4);
    s3 = sdfSphere(p, leg4Pos + vec3(0.0, halfH*0.3, 0.0), legRadius*1.5);
    s4 = sdfSphere(p, leg4Pos + vec3(0.0, halfH*0.9, 0.0), legRadius*2.0);
    sdf_leg4 = sdfSmoothMin(sdf_leg4, s1, 0.03);
    sdf_leg4 = sdfSmoothMin(sdf_leg4, s2, 0.03);
    sdf_leg4 = sdfSmoothMin(sdf_leg4, s3, 0.03);
    sdf_leg4 = sdfSmoothMin(sdf_leg4, s4, 0.03);

    float blobK = 0.06;
    float sdf_table = sdf_top;
    sdf_table = sdfSmoothMin(sdf_table, sdf_leg1, blobK);
    sdf_table = sdfSmoothMin(sdf_table, sdf_leg2, blobK);
    sdf_table = sdfSmoothMin(sdf_table, sdf_leg3, blobK);
    sdf_table = sdfSmoothMin(sdf_table, sdf_leg4, blobK);
    
    return sdf_table;
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

vec3 rotateZ(vec3 p, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return vec3(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
}

vec3 rotateX(vec3 p, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return vec3(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
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
        float ds = sdf2(p);

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
    // float s = sdf2(p);          //// sdf value in p
    float dx = 0.01;           //// step size for finite difference

    //// your implementation starts

    // Perturb in x
    vec3 p_plus_dx = p + vec3(1.0, 0.0, 0.0) * dx; 
    vec3 p_minus_dx = p - vec3(1.0, 0.0, 0.0) * dx;
    float s_plus_dx = sdf2(p_plus_dx);
    float s_minus_dx = sdf2(p_minus_dx);
    float grad_x = (s_plus_dx - s_minus_dx) / (2.0 * dx);

    // Perturb in y
    vec3 p_plus_dy = p + vec3(0.0, 1.0, 0.0) * dx; 
    vec3 p_minus_dy = p - vec3(0.0, 1.0, 0.0) * dx;
    float s_plus_dy = sdf2(p_plus_dy);
    float s_minus_dy = sdf2(p_minus_dy);
    float grad_y = (s_plus_dy - s_minus_dy) / (2.0 * dx);

    // Perturb in z
    vec3 p_plus_dz = p + vec3(0.0, 0.0, 1.0) * dx; 
    vec3 p_minus_dz = p - vec3(0.0, 0.0, 1.0) * dx;
    float s_plus_dz = sdf2(p_plus_dz);
    float s_minus_dz = sdf2(p_minus_dz);
    float grad_z = (s_plus_dz - s_minus_dz) / (2.0 * dx);

    vec3 n = normalize(vec3(grad_x, grad_y, grad_z));

    return n;

    //// your implementation ends
}

vec3 grapeCenter(vec3 ref, vec3 dir, float grapeSpacing)
{
    return ref + normalize(dir) * grapeSpacing;
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
    // background
    if(length(p) > 40.0){
        return vec3(0.0, 0.0, 0.0);
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
    /// Table
    float sdf_table = sdfBlobbyTable(p, -0.1);

    /// Bowl
    vec3 boxCenter = vec3(0.0, -0.05, 0.0);
    vec3 boxB = vec3(0.1, 0.05, 0.1);

    float sdf_box = sdfBox(p, boxCenter, boxB);

    // Cut hollow sphere to make plate
    float cutHollowSphereR = 0.5;
    float cutHollowSphereH = -0.4;
    float cCutHollowSphereT = 0.01;

    vec3 cutHollowCenter = vec3(0.0, cutHollowSphereR, 0.0);
    vec3 cutHollowP = p - cutHollowCenter;
    vec3 cutHollowBottom = vec3(0.0, cutHollowCenter.y + cutHollowSphereH, 0.0);

    float sdf_cut_hollow_sphere = sdfCutHollowSphere(cutHollowP, cutHollowSphereR, cutHollowSphereH, cCutHollowSphereT);

    float sdf_bowl = sdfSmoothMin(sdf_cut_hollow_sphere, sdf_box, 0.03);

    // Torus dougned on top of plate
    vec3 torusCenter = cutHollowBottom + vec3(0.0, 0, 0.0);
    vec3 torusP = p - torusCenter;
    float torusR = 0.3;  // ring size
    float torusT = 0.02;  // thickness
    float sdf_torus = sdfTorus(torusP, vec2(torusR, torusT));

    // Add bumps to torius
    float bumps = 1e9;
    for (float i = 0.0; i < 6.28; i += 0.5) {
        vec3 bumpPos = torusCenter + vec3(torusR * cos(i), 0.0, torusR * sin(i));
        bumps = min(bumps, sdfSphere(p, bumpPos, 0.001));
    }

    float sdf_blobby_torus = sdfSmoothMin(sdf_torus, bumps, 0.02);
    sdf_bowl = min(sdf_bowl, sdf_blobby_torus);

    /// Grapes
    // Main top thick stem
    vec3 stemBase = cutHollowCenter / 3.0 + vec3(0.2, 0.05, 0.0);

    vec3 stemStart = stemBase + vec3(-0.4, 0.35, 0.0);
    vec3 stemEnd = stemBase + vec3(-0.35, 0.275, 0.0);
    float sdfStem = sdfCappedCone(p, stemStart, stemEnd, 0.007, 0.003);

    // Fork two branches splitting
    vec3 fork1Start = stemEnd;
    vec3 fork1EndA = fork1Start + vec3(0.15, -0.12, 0.05) * 0.5; 
    float branch1A = sdfCappedCone(p, fork1Start, fork1EndA, 0.005, 0.003);

    vec3 fork1EndB = fork1Start + vec3(0.08, -0.10, -0.08) * 0.4;
    float branch1B = sdfCappedCone(p, fork1Start, fork1EndB, 0.004, 0.002);

    // Wanna have the sharp edges
    sdfStem = min(sdfStem, branch1A);
    sdfStem = min(sdfStem, branch1B);

    //  Fork two branches from branch1A
    vec3 fork2Start = fork1EndA;
    vec3 fork2EndA = fork2Start + vec3(0.12, -0.08, 0.06) * 0.5;
    vec3 fork2EndB = fork2Start + vec3(0.02, -0.12, 0.08) * 0.5;
    float branch2A = sdfCappedCone(p, fork2Start, fork2EndA, 0.005, 0.003);
    float branch2B = sdfCappedCone(p, fork2Start, fork2EndB, 0.004, 0.002);
    sdfStem = min(sdfStem, branch2A);
    sdfStem = min(sdfStem, branch2B);

    // Place grapes
    float grapeR = 0.05;
    float grapeSpacing = 2.0 * grapeR;

    // -------------------- Grapes hanging from fork1EndB area ---------------------
    vec3 grapeBase = fork1EndB;

    // Top grape
    vec3 f1b_g1 = grapeCenter(grapeBase, vec3(0.0, -1.0, 0.0), grapeR);
    float sdfGrapes = sdfSphere(p, f1b_g1, grapeR);

    // Second layer
    vec3 f1b_g2 = grapeCenter(f1b_g1, vec3(0.5, -0.8, 0.4), grapeSpacing);
    vec3 f1b_g3 = grapeCenter(f1b_g1, vec3(-0.6, -0.7, 0.3), grapeSpacing);
    vec3 f1b_g4 = grapeCenter(f1b_g1, vec3(0.1, -0.8, -0.6), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g2, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g3, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g4, grapeR));

    // Third layer
    vec3 f1b_g5 = grapeCenter(f1b_g2, vec3(0.3, -0.9, 0.2), grapeSpacing);
    vec3 f1b_g6 = grapeCenter(f1b_g3, vec3(-0.2, -0.9, 0.3), grapeSpacing);
    vec3 f1b_g7 = grapeCenter(f1b_g4, vec3(0.0, -0.95, -0.2), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g5, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g6, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g7, grapeR));

    vec3 f1b_g8 = (f1b_g5 + f1b_g6) * 0.5 + vec3(0.0, -grapeR * 0.5, 0.0);  // Between g5 and g6
    vec3 f1b_g9 = (f1b_g6 + f1b_g7) * 0.5 + vec3(0.0, -grapeR * 0.5, 0.0);  // Between g6 and g7
    vec3 f1b_g10 = (f1b_g5 + f1b_g7) * 0.5 + vec3(0.0, -grapeR * 0.5, 0.0); // Between g5 and g7
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g8, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g9, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g10, grapeR));

    // Fourth layer
    vec3 f1b_g11 = grapeCenter(f1b_g8, vec3(0.2, -0.95, 0.1), grapeSpacing);
    vec3 f1b_g12 = grapeCenter(f1b_g9, vec3(-0.1, -0.95, -0.2), grapeSpacing);
    vec3 f1b_g13 = grapeCenter(f1b_g10, vec3(0.1, -0.95, -0.1), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g11, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g12, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g13, grapeR));

    // ------------- Grapes hanging from fork2EndA -------------------------
    grapeBase = fork2EndA;

    // Top grape
    vec3 f2a_g1 = grapeCenter(grapeBase, vec3(1.1, -1.4, 0.0), 0.03);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f2a_g1, grapeR));

    // Layer around top grape
    vec3 f2a_g2 = grapeCenter(f2a_g1, vec3(-0.5, -0.7, 0.5), grapeSpacing);
    vec3 f2a_g3 = grapeCenter(f2a_g1, vec3(0.0, -0.7, -1.5), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f2a_g2, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f2a_g3, grapeR));
    
    // Spill chain to the right and down
    vec3 spillStart = grapeCenter(f2a_g1, vec3(0.5, -0.7, 0.5), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillStart, grapeR));

    vec3 spill1 = grapeCenter(spillStart, vec3(0.7, -1.5, 0.4), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spill1, grapeR));

    vec3 spill2 = grapeCenter(spill1, vec3(0.8, -0.8, 0.2), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spill2, grapeR));
    
    vec3 spill3 = grapeCenter(spill2, vec3(0.7, 0.2, 0.3), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spill3, grapeR));

    vec3 spill4 = grapeCenter(spill3, vec3(0.8, -0.5, 0.2), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spill4, grapeR));

    vec3 spill5 = grapeCenter(spill4, vec3(0.2, -0.8, 0.1), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spill5, grapeR));

    vec3 spill6 = grapeCenter(spill5, vec3(0.7, -0.7, 0.0), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spill6, grapeR));

    vec3 spill7 = grapeCenter(spill6, vec3(0.7, 0.0, 0.0), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spill7, grapeR));

    // More thickness, grapes on the other side in +z direction
    vec3 spillSide7 = grapeCenter(spill1, vec3(0.1, -0.5, 0.85), grapeSpacing);
    vec3 spillSide8 = grapeCenter(spill2, vec3(0.2, -0.4, 0.9), grapeSpacing);
    vec3 spillSide9 = grapeCenter(spill3, vec3(0.1, -0.3, 0.9), grapeSpacing);
    vec3 spillSide10 = grapeCenter(spill4, vec3(0.2, -0.5, 0.85), grapeSpacing);
    vec3 spillSide11 = grapeCenter(spill5, vec3(0.1, -0.4, 0.9), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillSide7, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillSide8, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillSide9, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillSide10, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillSide11, grapeR));


    // Front side thickness (-Z direction)
    vec3 spillFront1 = grapeCenter(spillStart, vec3(0.2, -0.5, -0.85), grapeSpacing);
    vec3 spillFront2 = grapeCenter(spill1, vec3(0.1, -0.5, -0.85), grapeSpacing);
    vec3 spillFront3 = grapeCenter(spill2, vec3(0.2, -0.4, -0.9), grapeSpacing);
    vec3 spillFront4 = grapeCenter(spill3, vec3(0.1, -0.3, -0.9), grapeSpacing);
    vec3 spillFront5 = grapeCenter(spill4, vec3(0.2, -0.5, -0.85), grapeSpacing);
    vec3 spillFront6 = grapeCenter(spill5, vec3(0.1, -0.4, -0.9), grapeSpacing);
    vec3 spillFront7 = grapeCenter(spill6, vec3(0.2, 0.0, -0.85), grapeSpacing);

    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillFront1, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillFront2, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillFront3, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillFront4, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillFront5, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillFront6, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillFront7, grapeR));

    // Add more graps to fill the front
    vec3 front_g1 = grapeCenter(f2a_g1, vec3(0.3, -0.8, -0.5), grapeSpacing);
    vec3 front_g2 = grapeCenter(f2a_g3, vec3(0.4, -0.7, -0.5), grapeSpacing);

    sdfGrapes = min(sdfGrapes, sdfSphere(p, front_g1, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, front_g2, grapeR));

    vec3 front_g3 = grapeCenter(front_g2, vec3(0.0, -0.7, 0.2), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, front_g3, grapeR));

    vec3 front_g4 = grapeCenter(front_g3, vec3(0.4, -0.7, -0.5), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, front_g4, grapeR));

    vec3 front_g5 = grapeCenter(front_g3, vec3(-0.6, -0.7, -0.5), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, front_g5, grapeR));

    float EPS = 1e-2;

    if (sdf_table < EPS) {
        color = vec3(0.55, 0.4, 0.25);      // Wooden ground
    } else if (sdf_bowl < EPS) {
        color = vec3(1.0, 1.0, 1.0);     // White bowl
    } else if (sdfStem < EPS) {
        color = vec3(0.45, 0.32, 0.18);    // Brown stem
    } else if (sdfGrapes < EPS) {
        color = vec3(0.7, 0.85, 0.4);      // Green grapes
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
    s = 1e9;

    float k = 0.03;

    /// Table
    float sdf_table = sdfBlobbyTable(p, -0.1);
    s = min(s, sdf_table);

    /// Bowl
    vec3 boxCenter = vec3(0.0, -0.05, 0.0);
    vec3 boxB = vec3(0.1, 0.05, 0.1);

    float sdf_box = sdfBox(p, boxCenter, boxB);

    // Cut hollow sphere to make plate
    float cutHollowSphereR = 0.5;
    float cutHollowSphereH = -0.4;
    float cCutHollowSphereT = 0.01;

    vec3 cutHollowCenter = vec3(0.0, cutHollowSphereR, 0.0);
    vec3 cutHollowP = p - cutHollowCenter;
    vec3 cutHollowBottom = vec3(0.0, cutHollowCenter.y + cutHollowSphereH, 0.0);

    float sdf_cut_hollow_sphere = sdfCutHollowSphere(cutHollowP, cutHollowSphereR, cutHollowSphereH, cCutHollowSphereT);

    float sdf_bowl = sdfSmoothMin(sdf_cut_hollow_sphere, sdf_box, 0.03);

    // Torus dougned on top of plate
    vec3 torusCenter = cutHollowBottom + vec3(0.0, 0, 0.0);
    vec3 torusP = p - torusCenter;
    float torusR = 0.3;  // ring size
    float torusT = 0.02;  // thickness
    float sdf_torus = sdfTorus(torusP, vec2(torusR, torusT));

    // Add bumps to torius
    float bumps = 1e9;
    for (float i = 0.0; i < 6.28; i += 0.5) {
        vec3 bumpPos = torusCenter + vec3(torusR * cos(i), 0.0, torusR * sin(i));
        bumps = min(bumps, sdfSphere(p, bumpPos, 0.001));
    }

    float sdf_blobby_torus = sdfSmoothMin(sdf_torus, bumps, 0.02);
    sdf_bowl = min(sdf_bowl, sdf_blobby_torus);

    s = min(s, sdf_bowl);

    /// Grapes
    // Main top thick stem
    vec3 stemBase = cutHollowCenter / 3.0 + vec3(0.2, 0.05, 0.0);

    vec3 stemStart = stemBase + vec3(-0.4, 0.35, 0.0);
    vec3 stemEnd = stemBase + vec3(-0.35, 0.275, 0.0);
    float sdfStem = sdfCappedCone(p, stemStart, stemEnd, 0.007, 0.003);

    // Fork two branches splitting
    vec3 fork1Start = stemEnd;
    vec3 fork1EndA = fork1Start + vec3(0.15, -0.12, 0.05) * 0.5; 
    float branch1A = sdfCappedCone(p, fork1Start, fork1EndA, 0.005, 0.003);

    vec3 fork1EndB = fork1Start + vec3(0.08, -0.10, -0.08) * 0.4;
    float branch1B = sdfCappedCone(p, fork1Start, fork1EndB, 0.004, 0.002);

    // Wanna have the sharp edges
    sdfStem = min(sdfStem, branch1A);
    sdfStem = min(sdfStem, branch1B);

    //  Fork two branches from branch1A
    vec3 fork2Start = fork1EndA;
    vec3 fork2EndA = fork2Start + vec3(0.12, -0.08, 0.06) * 0.5;
    vec3 fork2EndB = fork2Start + vec3(0.02, -0.12, 0.08) * 0.5;
    float branch2A = sdfCappedCone(p, fork2Start, fork2EndA, 0.005, 0.003);
    float branch2B = sdfCappedCone(p, fork2Start, fork2EndB, 0.004, 0.002);
    sdfStem = min(sdfStem, branch2A);
    sdfStem = min(sdfStem, branch2B);
    s = min(s, sdfStem);

    // Place grapes
    float grapeR = 0.05;
    float grapeSpacing = 2.0 * grapeR;

    // -------------------- Grapes hanging from fork1EndB area ---------------------
    vec3 grapeBase = fork1EndB;

    // Top grape
    vec3 f1b_g1 = grapeCenter(grapeBase, vec3(0.0, -1.0, 0.0), grapeR);
    float sdfGrapes = sdfSphere(p, f1b_g1, grapeR);

    // Second layer
    vec3 f1b_g2 = grapeCenter(f1b_g1, vec3(0.5, -0.8, 0.4), grapeSpacing);
    vec3 f1b_g3 = grapeCenter(f1b_g1, vec3(-0.6, -0.7, 0.3), grapeSpacing);
    vec3 f1b_g4 = grapeCenter(f1b_g1, vec3(0.1, -0.8, -0.6), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g2, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g3, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g4, grapeR));

    // Third layer
    vec3 f1b_g5 = grapeCenter(f1b_g2, vec3(0.3, -0.9, 0.2), grapeSpacing);
    vec3 f1b_g6 = grapeCenter(f1b_g3, vec3(-0.2, -0.9, 0.3), grapeSpacing);
    vec3 f1b_g7 = grapeCenter(f1b_g4, vec3(0.0, -0.95, -0.2), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g5, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g6, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g7, grapeR));

    vec3 f1b_g8 = (f1b_g5 + f1b_g6) * 0.5 + vec3(0.0, -grapeR * 0.5, 0.0);  // Between g5 and g6
    vec3 f1b_g9 = (f1b_g6 + f1b_g7) * 0.5 + vec3(0.0, -grapeR * 0.5, 0.0);  // Between g6 and g7
    vec3 f1b_g10 = (f1b_g5 + f1b_g7) * 0.5 + vec3(0.0, -grapeR * 0.5, 0.0); // Between g5 and g7
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g8, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g9, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g10, grapeR));

    // Fourth layer
    vec3 f1b_g11 = grapeCenter(f1b_g8, vec3(0.2, -0.95, 0.1), grapeSpacing);
    vec3 f1b_g12 = grapeCenter(f1b_g9, vec3(-0.1, -0.95, -0.2), grapeSpacing);
    vec3 f1b_g13 = grapeCenter(f1b_g10, vec3(0.1, -0.95, -0.1), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g11, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g12, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f1b_g13, grapeR));

    // ------------- Grapes hanging from fork2EndA -------------------------
    grapeBase = fork2EndA;

    // Top grape
    vec3 f2a_g1 = grapeCenter(grapeBase, vec3(1.1, -1.4, 0.0), 0.03);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f2a_g1, grapeR));

    // Layer around top grape
    vec3 f2a_g2 = grapeCenter(f2a_g1, vec3(-0.5, -0.7, 0.5), grapeSpacing);
    vec3 f2a_g3 = grapeCenter(f2a_g1, vec3(0.0, -0.7, -1.5), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f2a_g2, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, f2a_g3, grapeR));
    
    // Spill chain to the right and down
    vec3 spillStart = grapeCenter(f2a_g1, vec3(0.5, -0.7, 0.5), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillStart, grapeR));

    vec3 spill1 = grapeCenter(spillStart, vec3(0.7, -1.5, 0.4), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spill1, grapeR));

    vec3 spill2 = grapeCenter(spill1, vec3(0.8, -0.8, 0.2), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spill2, grapeR));
    
    vec3 spill3 = grapeCenter(spill2, vec3(0.7, 0.2, 0.3), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spill3, grapeR));

    vec3 spill4 = grapeCenter(spill3, vec3(0.8, -0.5, 0.2), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spill4, grapeR));

    vec3 spill5 = grapeCenter(spill4, vec3(0.2, -0.8, 0.1), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spill5, grapeR));

    vec3 spill6 = grapeCenter(spill5, vec3(0.7, -0.7, 0.0), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spill6, grapeR));

    vec3 spill7 = grapeCenter(spill6, vec3(0.7, 0.0, 0.0), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spill7, grapeR));

    // More thickness, grapes on the other side in +z direction
    vec3 spillSide7 = grapeCenter(spill1, vec3(0.1, -0.5, 0.85), grapeSpacing);
    vec3 spillSide8 = grapeCenter(spill2, vec3(0.2, -0.4, 0.9), grapeSpacing);
    vec3 spillSide9 = grapeCenter(spill3, vec3(0.1, -0.3, 0.9), grapeSpacing);
    vec3 spillSide10 = grapeCenter(spill4, vec3(0.2, -0.5, 0.85), grapeSpacing);
    vec3 spillSide11 = grapeCenter(spill5, vec3(0.1, -0.4, 0.9), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillSide7, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillSide8, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillSide9, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillSide10, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillSide11, grapeR));


    // Front side thickness (-Z direction)
    vec3 spillFront1 = grapeCenter(spillStart, vec3(0.2, -0.5, -0.85), grapeSpacing);
    vec3 spillFront2 = grapeCenter(spill1, vec3(0.1, -0.5, -0.85), grapeSpacing);
    vec3 spillFront3 = grapeCenter(spill2, vec3(0.2, -0.4, -0.9), grapeSpacing);
    vec3 spillFront4 = grapeCenter(spill3, vec3(0.1, -0.3, -0.9), grapeSpacing);
    vec3 spillFront5 = grapeCenter(spill4, vec3(0.2, -0.5, -0.85), grapeSpacing);
    vec3 spillFront6 = grapeCenter(spill5, vec3(0.1, -0.4, -0.9), grapeSpacing);
    vec3 spillFront7 = grapeCenter(spill6, vec3(0.2, 0.0, -0.85), grapeSpacing);

    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillFront1, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillFront2, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillFront3, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillFront4, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillFront5, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillFront6, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, spillFront7, grapeR));


    // Add more graps to fill the front
    vec3 front_g1 = grapeCenter(f2a_g1, vec3(0.3, -0.8, -0.5), grapeSpacing);
    vec3 front_g2 = grapeCenter(f2a_g3, vec3(0.4, -0.7, -0.5), grapeSpacing);

    sdfGrapes = min(sdfGrapes, sdfSphere(p, front_g1, grapeR));
    sdfGrapes = min(sdfGrapes, sdfSphere(p, front_g2, grapeR));

    vec3 front_g3 = grapeCenter(front_g2, vec3(0.0, -0.7, 0.2), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, front_g3, grapeR));

    vec3 front_g4 = grapeCenter(front_g3, vec3(0.4, -0.7, -0.5), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, front_g4, grapeR));

    vec3 front_g5 = grapeCenter(front_g3, vec3(-0.6, -0.7, -0.5), grapeSpacing);
    sdfGrapes = min(sdfGrapes, sdfSphere(p, front_g5, grapeR));

    s = min(s, sdfGrapes);
    //// your implementation ends

    return s;
}

/////////////////////////////////////////////////////
//// main function
/////////////////////////////////////////////////////

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{

    vec2 uv = (fragCoord.xy - .5 * iResolution.xy) / iResolution.y;
    
    //// Video view rotating around y axis////////////
    vec3 target = vec3(0.0, 0.0, 0.0);
    
    
    float camDist = 3.0;
    float camHeight = 1.0;
    float angle = iTime * 0.5;
    
    vec3 origin = vec3(
        camDist * sin(angle),
        camHeight,
        camDist * cos(angle)
    );
    
    vec3 forward = normalize(target - origin);
    vec3 worldUp = vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(forward, worldUp));
    vec3 up = cross(right, forward);
    
    vec3 dir = normalize(forward + uv.x * right + uv.y * up);

    //////////////////
    /////// Angled view ///////////
    // vec3 origin = vec3(0.0, 1.0, -2.0); 
    // vec3 target = vec3(0.0, 0.0, 0.0);

    // // 45 pitch foraward direction
    // vec3 forward = normalize(target - origin);
    // vec3 worldUp = vec3(0.0, 1.0, 0.0);


    // vec3 right = normalize(cross(forward, worldUp));
    // vec3 up    = cross(right, forward);
    // vec3 dir = normalize(forward + uv.x * right + uv.y * up);
   //////////////

    // vec2 uv = (fragCoord.xy - .5 * iResolution.xy) / iResolution.y;         //// screen uv
    // vec3 origin = CAM_POS;                                                  //// camera position 
    // vec3 dir = normalize(vec3(uv.x, uv.y, 1));                              //// camera direction
    
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