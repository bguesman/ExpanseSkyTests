#ifndef EXPANSE_SKY_COMMON_INCLUDED
#define EXPANSE_SKY_COMMON_INCLUDED

/********************************************************************************/
/****************************** UNIFORM VARIABLES *******************************/
/********************************************************************************/

/* All of these things have to be in a cbuffer, so we can access them across
 * different shaders. */
CBUFFER_START(ExpanseSky)
float _atmosphereThickness;
float _planetRadius;
TEXTURECUBE(_groundColorTexture);
float4 _groundTint;
TEXTURECUBE(_groundEmissiveTexture);
float _groundEmissiveMultiplier;
TEXTURECUBE(_nightSkyHDRI);
float4 _nightTint;
float _nightIntensity;
float _aerosolCoefficient;
float _scaleHeightAerosols;
float _aerosolAnisotropy;
float _aerosolDensity;
float4 _airCoefficients;
float _scaleHeightAir;
float _airDensity;
float4 _ozoneCoefficients;
float _ozoneThickness;
float _ozoneHeight;
float _ozoneDensity;
float4 _skyTint;
float _starAerosolScatterMultiplier;
int _numberOfSamples;
bool _useImportanceSampling;
bool _useCubicApproximation;
float _ditherAmount;

float3   _WorldSpaceCameraPos1;
float4x4 _ViewMatrix1;
#undef UNITY_MATRIX_V
#define UNITY_MATRIX_V _ViewMatrix1

/* Precomputed tables. */

/* Transmittance table. Leverages spherical symmetry of the atmosphere,
 * parameterized by:
 * h (x dimension): the height of the camera.
 * phi (y dimension): the zenith angle of the viewing direction. */
TEXTURE2D(_TransmittanceTable);

/* Single scattering tables. Leverage spherical symmetry of the atmosphere,
 * parameterized by:
 * h (x dimension): the height of the camera.
 * phi (y dimension): the zenith angle of the viewing direction.
 * phi_l (z dimension): the zenith angle of the light source.
 * nu (w dimension): the azimuth angle of the light source. */
TEXTURE3D(_SingleScatteringTableAir);
TEXTURE3D(_SingleScatteringTableAerosol);

CBUFFER_END

/* Redefine colors to float3's for efficiency, since Unity can only set
 * float4's. */
#define _groundTintF3 _groundTint.xyz
#define _nightTintF3 _nightTint.xyz
#define _airCoefficientsF3 _airCoefficients.xyz
#define _ozoneCoefficientsF3 _ozoneCoefficients.xyz
#define _skyTintF3 _skyTint.xyz

/* Easier to type... */
#define g _aerosolAnisotropy

/* Set up a sampler for the cubemaps. */
SAMPLER(sampler_Cubemap);

/* Sampler for tables. */
#ifndef UNITY_SHADER_VARIABLES_INCLUDED
    SAMPLER(s_linear_clamp_sampler);
#endif

/* Some mathematical constants that are good to have pre-computed. */
#define PI 3.1415926535
#define SQRT_PI 1.77245385091
#define E 2.7182818285
#define GOLDEN_RATIO 1.6180339887498948482
#define GOLDEN_ANGLE 2.39996322972865332

#define FLT_EPSILON 0.00001 /* TODO: good value for this? */

/********************************************************************************/
/**************************** END UNIFORM VARIABLES *****************************/
/********************************************************************************/

float random(float2 uv) {
  return frac(sin(dot(uv,float2(12.9898,78.233)))*43758.5453123);
}

float clampCosine(float c) {
  return clamp(c, -1.0, 1.0);
}

/* True if a is greater than b within tolerance FLT_EPSILON, false
 * otherwise. */
bool floatGT(float a, float b) {
  return a > b - FLT_EPSILON;
}

/* True if a is less than b within tolerance FLT_EPSILON, false
 * otherwise. */
bool floatLT(float a, float b) {
  return a < b + FLT_EPSILON;
}

/* Returns t values of ray intersection with sphere. Third value indicates
 * if there was an intersection at all; if negative, there was no
 * intersection. */
float3 intersectSphere(float3 p, float3 d, float r) {
  float A = dot(d, d);
  float B = 2.f * dot(d, p);
  float C = dot(p, p) - (r * r);
  float det = (B * B) - 4.f * A * C;
  if (floatGT(det, 0.0)) {
    det = sqrt(max(0.0, det));
    return float3((-B + det) / (2.f * A), (-B - det) / (2.f * A), 1.0);
  }
  return float3(0, 0, -1.0);
}

/* This parameterization was taken from Bruneton and Neyer's model. */
/* Returns u_r, u_mu. */
float2 mapTransmittanceCoordinates(float r, float mu, float atmosphereRadius,
  float planetRadius, bool groundHit) {
  float rho = sqrt(max(0.0, r * r - planetRadius * planetRadius));
  float H = sqrt(max(0.0, atmosphereRadius * atmosphereRadius - planetRadius * planetRadius));
  float delta = r * r * mu * mu - rho * rho;

  float u_mu = 0.0;
  float discriminant = r * r * mu * mu - r * r + planetRadius * planetRadius;
  if (groundHit) {
    float d = -r * mu - sqrt(max(0.0, discriminant));
    float d_min = r - planetRadius;
    float d_max = rho;
    /* Use lower half of [0, 1] range. */
    u_mu = 0.5 - 0.5 * (d_max == d_min ? 0.0 : (d - d_min) / (d_max - d_min));
  } else {
    float d = -r * mu + sqrt(max(0.0, discriminant + H * H));
    float d_min = atmosphereRadius - r;
    float d_max = rho + H;
    /* Use upper half of [0, 1] range. */
    u_mu = 0.5 + 0.5 * (d_max == d_min ? 0.0 : (d - d_min) / (d_max - d_min));
  }

  float u_r = rho / H;

  // naive mapping:
  //float u_mu = (mu + 1) * 0.5;

  return float2(u_r, u_mu);
}

/* Returns r, mu. */
float2 unmapTransmittanceCoordinates(float u_r, float u_mu,
  float atmosphereRadius, float planetRadius) {
  float H = sqrt(max(0.0, atmosphereRadius * atmosphereRadius - planetRadius * planetRadius));
  float rho = u_r * H;

  float r = sqrt(max(0.0, rho * rho + planetRadius * planetRadius));

  float mu = 0.0;
  if (u_mu < 0.5) {
    float d_min = r - planetRadius;
    float d_max = rho;
    float d = d_min + (d_max - d_min) * (1.0 - 2.0 * u_mu);
    mu = (d == 0.0) ? -1.0 : clampCosine(-(rho * rho + d * d) / (2 * r * d));
  } else {
    float d_min = atmosphereRadius - r;
    float d_max = rho + H;
    float d = d_min + (d_max - d_min) * (2.0 * u_mu - 1.0);
    mu = (d == 0.0) ? 1.0 : clampCosine((H * H - rho * rho - d * d) / (2 * r * d));
  }

  // naive mapping:
  //float mu = u_mu * 2 - 1;

  return float2(r, mu);
}

/* Follow the strategy in physically based sky and lerp between 2 4D
 * texture lookups to solve the issue of uv-mapping for a deep texture. */
struct TexCoord5D {
  float x, y, z, w, a;
};

/* Returns u_r, u_mu, u_mu_l/u_nu bundled into z. */
TexCoord5D mapSingleScatteringCoordinates(float r, float mu, float mu_l, float nu,
  float atmosphereRadius, float planetRadius, float d, bool groundHit) {
  float rho = sqrt(max(0.0, r * r - planetRadius * planetRadius));
  float H = sqrt(max(0.0, atmosphereRadius * atmosphereRadius - planetRadius * planetRadius));

  float u_mu = 0.0;
  // float discriminant = r * r * mu * mu - r * r + planetRadius * planetRadius;
  if (groundHit) {
    // float d = -r * mu - sqrt(max(0.0, discriminant));
    float d_min = r - planetRadius;
    float d_max = rho;
    /* Use lower half of [0, 1] range. */
    // u_mu = 0.5 - 0.5 * (d_max == d_min ? 0.0 : (d - d_min) / (d_max - d_min));
    u_mu = 0.49 - 0.49 * (d_max == d_min ? 0.0 : (d - d_min) / (d_max - d_min));
  } else {
    // float d = -r * mu + sqrt(max(0.0, discriminant + H * H));
    float d_min = atmosphereRadius - r;
    float d_max = rho + H;
    /* Use upper half of [0, 1] range. */
    u_mu = 0.51 + 0.49 * (d_max == d_min ? 0.0 : (d - d_min) / (d_max - d_min));
  }

  float u_r = rho / H;

  /* TODO: this is an ad hoc mapping. Could be improved! */

  float u_mu_l = saturate((1.0 - exp(-3 * mu_l - 0.6)) / (1 - exp(-3.6)));

  // float u_mu_l = (1.0 + mu_l) / 2.0;

  float u_nu = saturate((1.0 + nu) / 2.0);

  // naive mapping of u_mu:
  // u_mu = (mu + 1) * 0.5;

  // float u_mu_l_floor = floor(u_mu * 32) / 32.0;
  // float u_mu_l_ceil = ceil(u_mu * 32) / 32.0;
  // float u_mu_a = ((u_mu * 32.0) - u_mu_l_floor);
  // float u_nu_deep_coord = u_nu/32.0;

  // float u_nu_deep_coord = (u_nu * (64.0 - 1.0));
  // float z = ((floor(u_mu_l * (16.0 - 1.0)) * 64.0) + u_nu_deep_coord) / (64.0 * 16.0);
  // float w = ((ceil(u_mu_l * (16.0 - 1.0)) * 64.0) + u_nu_deep_coord) / (64.0 * 16.0);
  // float a = (u_mu_l * (16.0 - 1.0)) - floor(u_mu_l * (16.0 - 1.0));
  // TexCoord5D toRet = {u_r, u_mu, saturate(z), saturate(w), saturate(a)};

  float zTexSize = 16;
  float zTexCount = 64;
  float w = (0.5 + u_mu_l * (zTexSize - 1)) * (1.0/zTexSize);
  float k = u_nu * (zTexCount - 1);
  float w0 = (floor(k) + w) * (1.0/zTexCount);
  float w1 = (ceil(k) + w) * (1.0/zTexCount);
  float a = frac(k);
  TexCoord5D toRet = {u_r, u_mu, w0, w1, a};
  return toRet;
}

/* Returns r, mu, mu_l, and nu. */
float4 unmapSingleScatteringCoordinates(float u_r, float u_mu, float u_mu_l,
  float u_nu, float atmosphereRadius, float planetRadius) {
  float H = sqrt(max(0.0, atmosphereRadius * atmosphereRadius - planetRadius * planetRadius));
  float rho = u_r * H;

  float r = sqrt(max(0.0, rho * rho + planetRadius * planetRadius));

  float mu = 0.0;
  if (floatLT(u_mu, 0.5)) {
    float d_min = r - planetRadius;
    float d_max = rho;
    float d = d_min + (d_max - d_min) * (1.0 - (1.0 / 0.49) * u_mu);
    mu = (d == 0.0) ? -1.0 : clampCosine(-(rho * rho + d * d) / (2.0 * r * d));
  } else {
    float d_min = atmosphereRadius - r;
    float d_max = rho + H;
    float d = d_min + (d_max - d_min) * (2.0 * u_mu - 1.02);
    mu = (d == 0.0) ? 1.0 : clampCosine((H * H - rho * rho - d * d) / (2.0 * r * d));
  }

  /* TODO: may want to check. */
  float mu_l = clampCosine((log(1.0 - (u_mu_l * (1 - exp(-3.6)))) + 0.6) / -3.0);
  // float mu_l = (u_mu_l * 2.0) - 1.0;

  float nu = clampCosine((u_nu * 2.0) - 1.0);

  nu = clamp(nu, mu * mu_l - sqrt((1.0 - mu * mu) * (1.0 - mu_l * mu_l)),
      mu * mu_l + sqrt((1.0 - mu * mu) * (1.0 - mu_l * mu_l)));

  // naive mapping:
  // mu = u_mu * 2 - 1;

  return float4(r, mu, mu_l, nu);
}

/* Computes density at a point for exponentially distributed atmosphere.
 * Assumes the planet is centered at the origin. */
float computeDensityExponential(float3 p, float planetR, float scaleHeight,
  float density) {
  return density * exp((planetR - length(p))/scaleHeight);
}

/* Computes density at a point for tent distributed atmosphere.
 * Assumes the planet is centered at the origin. */
float computeDensityTent(float3 p, float planetR, float height,
  float thickness, float density) {
  return density * max(0.0,
    1.0 - abs(length(p) - planetR - height) / (0.5 * thickness));
}

/* TODO: set up importance sampling for these optical depth functions? */
/* Computes the optical depth for an exponentially distributed layer. */
float computeOpticalDepthExponential(float3 originPoint, float3 samplePoint,
  float planetR, float scaleHeight, float density) {
  // Evaluate integral with curved planet with a midpoint integrator.
  float3 d = samplePoint - originPoint;
  float acc = 0.0;
  float ds = length(d) / ((float) _numberOfSamples);
  for (int i = 0; i < _numberOfSamples; i++) {
    float t = (((float) i + 0.5)) / ((float) _numberOfSamples);
    float3 pt = originPoint + (d * t);
    acc += computeDensityExponential(pt, planetR, scaleHeight, density) * ds;
  }
  return acc;
}

/* Computes the optical depth for a layer distributed as a tent
 * function at specified height with specified thickness. */
float computeOpticalDepthTent(float3 originPoint, float3 samplePoint,
  float planetR, float height, float thickness, float density) {
  // Evaluate integral with curved planet with a midpoint integrator.
  float3 d = samplePoint - originPoint;
  float acc = 0.0;
  float ds = length(d) / ((float) _numberOfSamples);
  for (int i = 0; i < _numberOfSamples; i++) {
    float t = (((float) i + 0.5)) / ((float) _numberOfSamples);
    float3 pt = originPoint + (d * t);
    acc += computeDensityTent(pt, planetR, height, thickness, density) * ds;
  }
  return acc;
}

#endif // EXPANSE_SKY_COMMON_INCLUDED
