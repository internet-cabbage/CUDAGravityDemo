#include "initialConditions.h"

/*
After solving the probability density equation for a galaxy with an exponential distribution, I could not perform a basic inversion due to the form of the equation.

Instead of that, I can perform a random sample over the equation, in a rectangle surrounding the curve. Then if the random sample lies outside of the curve, I can reject the coordinate,
and vice-versa.

The equation I am sampling over is: P(R) = R exp(-a R)
*/




// -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
// Compile time constants
// -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

// This determines the number of scale lengths, at which point no more stars will be spanwed in the exponential distance distribution
#define TRUNCATIONCONSTANT 5.0f

// The maximum number of attempts you can try to produce a star position, at which point the code gives up and assumes there
// is some manner of error in the distribution
#define MAXSAMPLEATTEMPTS 200

// This is just e^(-1)
#define invE 0.367879441171f
// This is my attempt to be funny
#define pie 3.141592653589793f

float randDecimal() {
    // A constant factor is added to the randomiser, so that 0.0 cannot be returned
    float randVal = (float) ( rand() + 1.0f)/(RAND_MAX + 3.0f);
    return randVal;
}

// -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
// Sampling functions (and a new cool comment block style!)
// -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-


// Alpha is a scaling constant which determines how fast the star density decreases as a function of radial distance R

float sampleRadius(float scaleLength) {
    // Maximum values of function to perform sampling over
    float alpha = 1.0 / scaleLength;
    float yMax = (1.0/alpha) * invE;

    float rMax = TRUNCATIONCONSTANT/alpha;
    int attempts = 0;

    while (attempts < MAXSAMPLEATTEMPTS) {
        float r = rMax * randDecimal();
        float y = yMax * randDecimal();
        float fVal = r * expf(-alpha * r);

        if (y < fVal) {
            return r;
        }
        else {
            attempts++;
        }
    }
    if (attempts >= MAXSAMPLEATTEMPTS) {
        printf("ERROR: The sampling algorithm has failed (%d) times to generate a valid star position.\nThere is likely an error with the chosen distribution.\n", MAXSAMPLEATTEMPTS);
        exit(-1);
    }
    return -1;
}

float sampleAngle(){
    float angle = 2.0 * pie * randDecimal();
    return angle;
}

float gaussian(float standardDev, float mean, float xVal) {
    float multConst = 1.0f/(sqrt(2.0f*pie*standardDev*standardDev));
    float exponent = -((xVal - mean)*(xVal - mean))/(2 * standardDev * standardDev);
    return multConst * expf(exponent);
}

/*
The vertical star density profile is being modelled as an inverse exponential distribution, which is independent of radius.
In the future I will try adapt the equation to be one which accounts for the radial distance, but I have not done that yet.

Density: P ∝ e^(-τ * |Z|), where P is density, τ is the thickness scaling constant, and Z is the height above\below the galactic plane

P = K * e^(-τ * |Z|)

Integrating over the density from negative infinity to positive infinity, gives us the total probability which should be one. So we first only integrate over the positive side, and doule it.

1.0 = \int_{-\infty}^{\infty} K * e^(-τ * |Z|) dZ = 2 K \int_0^{\infty} e^(-τ * Z) dZ

RHS = 2K [e^(-τ * Z)]_0^{\infty} = \frac{2K}{\tau} which is the total probability.

Since the total probability equals 1.0, we know that K = \frac{\tau}{2}

================

Therefore the density function is:

P = τ/2 * e^(-τ * |Z|)

The CDF should tend towards 1 as Z tends towards infinity, and also tend towards zreo as Z tends towards negative infinity.

Since it is a symmetric distribution about Z, at Z = 0 the CDF should be 0.5.

So to calculate the CDF I simple integrate between 0 and Z;

C(Z) = 0.5 + \int_0^{Z} τ/2 * e^(-τ * |Z|) dZ

C(Z) = 0.5 + 0.5 - 0.5 e^(-τ * |Z|)

C(Z) = 1.0 - 0.5 e^(-τ * |Z|)


// --------------

Setting up the sampling

u = C(Z), Z = C^-1 (u)

therefore u = 1 - 0.5 e^(-τ * Z) for Z > 0 and u > 0.5

2(1-u) = e^(-τ * Z)
ln(2-2u) = -τ * Z
-1/τ * (ln(2-2u)) = Z for Z > 0

// --------------

Now for the negative branch. We know that C(-Z) + C(Z) = 1 as the distribution is symmetric about zero

Therefore since C(Z) = 1 - 0.5 e^(-τ * Z) for Z > 0. Therefore C(-Z) = 0.5 e^(-τ * Z) for Z <0

Now to perform the inverse sample

u = 0.5e^(-τ * Z)
ln(2u) = -τ * Z
ln(2u) / -τ = Z

*/



float sampleHeight(float scaleThickness) {
    float tau = 1.0f / scaleThickness;
    float u = randDecimal();

    if (u >= 0.5f) {
        float heightZ = -logf(2.0f - 2.0f * u) / tau;
        return heightZ;
    }
    else {
        float heightZ = logf(2.0f * u) / tau;
        return heightZ;
    }
}

float galacticMassEnclosed(float scaleLength, float totalGalacticMass, float radius) {
    float expConst = expf(- radius / scaleLength);
    float enclosedMass = totalGalacticMass * (1 - (1  + radius/scaleLength) * expConst);
    return enclosedMass;
}

float sampleCircularSpeed(float scaleLength, float totalMass, float R) {
    if (R <= 0.0f) {printf("ERROR: Negative radius given.\n"); exit(-1);}
    float mass = galacticMassEnclosed(scaleLength,totalMass,R);
    float speed = sqrtf(G * mass / R);

    return speed;
}

float sampleVerticalSpeed(float netSpeed, float scaleFactor,float enclosedMass,float totalMass) {
    float speed = scaleFactor * netSpeed * randDecimal();
    float massEnclosedRatio = enclosedMass/totalMass;
    speed *= massEnclosedRatio;
    if (randDecimal() > 0.5) {
        return speed;
    }
    else {
        return -speed;
    }
}


// -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
// Rotation matrix code
// -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-


// Building the rotation matrix
// rotationAngles = [yaw, pitch, roll]

void createRotationMatrix(sVec3 rotationAngles, mat3 destination) {
    // Angles from struct
    float yaw = rotationAngles.x;
    float pitch = rotationAngles.y;
    float roll = rotationAngles.z;

    // Trig functions of angles
    float sinY = sin(yaw);
    float cosY = cos(yaw);

    float sinP = sin(pitch);
    float cosP = cos(pitch);

    float sinR = sin(roll);
    float cosR = cos(roll);

    // Rotation matrices

    mat3 yawMat =  {{cosY,  -sinY,  0},
                    {sinY,  cosY,   0},
                    {0,     0,      1}};
    
    mat3 pitchMat ={{cosP,  0,      sinP},
                    {0,     1,      0},
                    {-sinP, 0,      cosP}};
    
    mat3 rollMat = {{1,     0,      0},
                    {0,     cosR,   -sinR},
                    {0,      sinR,   cosR}};

    mat3 rotationMatrix;
    glm_mat3_mul(yawMat,pitchMat,rotationMatrix);
    glm_mat3_mul(rotationMatrix,rollMat,destination);
}

void rotateVector(mat3 rotationMatrix, sVec3* vectorToRotate) {
    vec3 vecVals; // used to convert to cglm vector format
    vecVals[0] = vectorToRotate->x;
    vecVals[1] = vectorToRotate->y;
    vecVals[2] = vectorToRotate->z;

    glm_mat3_mulv(rotationMatrix,vecVals,vecVals);
    vectorToRotate->x = vecVals[0];
    vectorToRotate->y = vecVals[1];
    vectorToRotate->z = vecVals[2];
}

void generateLocalCoordsDisk(galaxyDiskParameters* parameters,float* massVals,float4* posMassVals, float3* velVals, int spawnNumber) {

    // Alpha is the rate at which the surface density decreases as a function of length
    float scaleThickness = parameters->scaleThickness;
    float scaleLength = parameters->scaleLength;
    float totalMass = parameters->totalDiskMass;

    sVec3 positionOffset = parameters->galacticCentreLocation;
    sVec3 velocityOffset = parameters->galacticNetVelocity;

    // I need to work on calculating this analytically, as there is a method I forgot how to use
    // float verticalVelocityDispersion = 0.0;

    mat3 rotateMatrix;
    createRotationMatrix(parameters->rotationAngles, rotateMatrix);

    for (int i = 0; i < spawnNumber; i++) {
        float rVal = sampleRadius(scaleLength);
        float angleVal = sampleAngle();
        float cosA = cos(angleVal);
        float sinA = sin(angleVal);

        // Converts the polar coordinates to cartesian coordiantes in the x-y plame
        sVec3 pos;
        pos.x = rVal * cosA;
        pos.y = rVal * sinA;
        pos.z = sampleHeight(scaleThickness);

        float circularSpeed = sampleCircularSpeed(scaleLength,totalMass,rVal);
        sVec3 vel;
        vel.x = - circularSpeed * sinA;
        vel.y = circularSpeed * cosA;
        // Z velocity distribution
        float enclosedMass = galacticMassEnclosed(scaleLength,totalMass,rVal);
        vel.z = sampleVerticalSpeed(circularSpeed,0.2,enclosedMass,totalMass);

        // Rotate in the local frame
        rotateVector(rotateMatrix,&pos);
        rotateVector(rotateMatrix,&vel);

        posMassVals[i].x = pos.x + positionOffset.x;
        posMassVals[i].y = pos.y + positionOffset.y;
        posMassVals[i].z = pos.z + positionOffset.z;
        posMassVals[i].w = massVals[i];

        velVals[i].x = vel.x + velocityOffset.x;
        velVals[i].y = vel.y + velocityOffset.y;
        velVals[i].z = vel.z + velocityOffset.z;
    }
}