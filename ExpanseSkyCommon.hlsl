#ifndef EXPANSE_SKY_COMMON_INCLUDED
#define EXPANSE_SKY_COMMON_INCLUDED

/******************************************************************************/
/***************************** GLOBAL VARIABLES *******************************/
/******************************************************************************/

/* All of these things have to be in a cbuffer, so we can access them across
 * different shaders. */
CBUFFER_START(ExpanseSky)
float _atmosphereThickness;
float _planetRadius;
float _atmosphereRadius;
float _aerosolCoefficient;
float _scaleHeightAerosols;
float _aerosolDensity;
float4 _airCoefficients;
float _scaleHeightAir;
float _airDensity;
float4 _ozoneCoefficients;
float _ozoneThickness;
float _ozoneHeight;
float _ozoneDensity;
int _numberOfSamples;
bool _useImportanceSampling;
bool _useCubicApproximation;

/* Redefine colors to float3's for efficiency, since Unity can only set
 * float4's. */
#define _airCoefficientsF3 _airCoefficients.xyz
#define _ozoneCoefficientsF3 _ozoneCoefficients.xyz

/* Set up a sampler for the cubemaps. */
SAMPLER(sampler_Cubemap);

/* Precomputed tables. */

/* Transmittance table. Leverages spherical symmetry of the atmosphere,
 * parameterized by:
 * h (x dimension): the height of the camera.
 * phi (y dimension): the zenith angle of the viewing direction. */
TEXTURE2D(_TransmittanceTable);
/* Table dimensions. Must match those in ExpanseSkyRenderer.cs. TODO: would
 * be great to be able to set these as constants. I don't want to make them
 * accessible because I haven't set up any way to reallocate these on the
 * fly. */
#define TRANSMITTANCE_TABLE_SIZE_H 32
#define TRANSMITTANCE_TABLE_SIZE_PHI 128

/* Single scattering tables. Leverage spherical symmetry of the atmosphere,
 * parameterized by:
 * h (x dimension): the height of the camera.
 * phi (y dimension): the zenith angle of the viewing direction.
 * phi_l (z dimension): the zenith angle of the light source.
 * nu (w dimension): the azimuth angle of the light source. */
TEXTURE3D(_SingleScatteringTableAir);
TEXTURE3D(_SingleScatteringTableAerosol);
/* Table dimensions. Must match those in ExpanseSkyRenderer.cs. TODO: would
 * be great to be able to set these as constants. I don't want to make them
 * accessible because I haven't set up any way to reallocate these on the
 * fly. */
#define SINGLE_SCATTERING_TABLE_SIZE_H 32
#define SINGLE_SCATTERING_TABLE_SIZE_PHI 128
#define SINGLE_SCATTERING_TABLE_SIZE_PHI_L 32
#define SINGLE_SCATTERING_TABLE_SIZE_NU 64

CBUFFER_END

/* Sampler for tables. */
#ifndef UNITY_SHADER_VARIABLES_INCLUDED
    SAMPLER(s_linear_clamp_sampler);
    SAMPLER(s_trilinear_clamp_sampler);
#endif

/* Some mathematical constants that are good to have pre-computed. */
#define PI 3.1415926535
#define SQRT_PI 1.77245385091
#define E 2.7182818285
#define GOLDEN_RATIO 1.6180339887498948482
#define GOLDEN_ANGLE 2.39996322972865332

#define FLT_EPSILON 0.00001 /* TODO: good value for this? Use Unity's FLT_EPS? */

/******************************************************************************/
/*************************** END GLOBAL VARIABLES *****************************/
/******************************************************************************/

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
  float planetRadius, float d, bool groundHit) {
  float rho = sqrt(max(0.0, r * r - planetRadius * planetRadius));
  float H = sqrt(max(0.0, atmosphereRadius * atmosphereRadius - planetRadius * planetRadius));
  float delta = r * r * mu * mu - rho * rho;

  float u_mu = 0.0;
  float discriminant = r * r * mu * mu - r * r + planetRadius * planetRadius;
  if (groundHit) {
    float d_min = r - planetRadius;
    float d_max = rho;
    /* Use lower half of [0, 1] range. */
    u_mu = 0.49 - 0.49 * (d_max == d_min ? 0.0 : (d - d_min) / (d_max - d_min));
  } else {
    float d_min = atmosphereRadius - r;
    float d_max = rho + H;
    /* Use upper half of [0, 1] range. */
    u_mu = 0.51 + 0.49 * (d_max == d_min ? 0.0 : (d - d_min) / (d_max - d_min));
  }

  float u_r = rho / H;

  // naive mapping of mu. TODO: make toggable?:
  // float u_mu = (1.0 + mu) * 0.5;

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
    float d = d_min + (d_max - d_min) * (1.0 - (1.0 / 0.49) * u_mu);
    mu = (d == 0.0) ? -1.0 : clampCosine(-(rho * rho + d * d) / (2 * r * d));
  } else {
    float d_min = atmosphereRadius - r;
    float d_max = rho + H;
    float d = d_min + (d_max - d_min) * (2.0 * u_mu - 1.02);
    mu = (d == 0.0) ? 1.0 : clampCosine((H * H - rho * rho - d * d) / (2 * r * d));
  }

  // naive mapping of mu. TODO: make toggable?:
  // float mu = 2.0 * u_mu - 1.0;

  return float2(r, mu);
}

/* Follow the strategy in physically based sky and lerp between 2 4D
 * texture lookups to solve the issue of uv-mapping for a deep texture. */
struct TexCoord5D {
  float x, y, z, w, a;
};

/* Converts u, v in unit range to a deep texture coordinate (w0, w1, a) with
 * zTexSize rows and zTexCount columns.
 *
 * Returns (w0, w1, a), where w0 and w1 are the locations to sample the
 * texture at and a is the blend amount to use when interpolating between
 * them. Mathematically:
 *
 *         sample(u, v) = a * sample(w0) + (1 - a) * sample(w1)
 *
 */
float3 uvToDeepTexCoord(float u, float v, int zTexSize, int zTexCount) {
  float w = (0.5 + u * (zTexSize - 1)) * (1.0/zTexSize);
  float k = v * (zTexCount - 1);
  float w0 = (floor(k) + w) * (1.0/zTexCount);
  float w1 = (ceil(k) + w) * (1.0/zTexCount);
  float a = frac(k);
  return float3(w0, w1, a);
}

/* Converts deep texture index in range zTexSize * zTexCount to the
 * uv coordinate in unit range that represents the 2D table index for a
 * table with zTexSize rows and zTexCount columns.
 *
 * Returns (u, v).
 */
float2 deepTexIndexToUV(int deepTexCoord, int zTexSize, int zTexCount) {
  uint texId = deepTexCoord / zTexSize;
  uint texCoord = deepTexCoord & (zTexSize - 1);
  float u = saturate(texCoord / (float(zTexSize) - 1.0));
  float v = saturate(texId / (float(zTexCount) - 1.0));
  return float2(u, v);
}


/* Returns u_r, u_mu, u_mu_l/u_nu bundled into z. */
TexCoord5D mapSingleScatteringCoordinates(float r, float mu, float mu_l, float nu,
  float atmosphereRadius, float planetRadius, float d, bool groundHit) {

  float2 u_r_mu = mapTransmittanceCoordinates(r, mu, atmosphereRadius,
    planetRadius, d, groundHit);

  /* TODO: this is an ad hoc mapping. Could be improved! */
  float u_mu_l = saturate((1.0 - exp(-3 * mu_l - 0.6)) / (1 - exp(-3.6)));

  float u_nu = saturate((1.0 + nu) / 2.0);

  // naive mapping of u_mu and u_mu_l:
  /* TODO: make togglable? could slow things down a lot! but we already have
   * a branch up there ^ anyway... */
  // u_mu = (1.0 + mu) * 0.5;
  // u_mu_l = (1.0 + mu_l) * 0.5;

  float3 deepTexCoord = uvToDeepTexCoord(u_mu_l, u_nu,
    SINGLE_SCATTERING_TABLE_SIZE_PHI_L, SINGLE_SCATTERING_TABLE_SIZE_NU);

  TexCoord5D toRet = {u_r_mu.x, u_r_mu.y, deepTexCoord.x,
    deepTexCoord.y, deepTexCoord.z};
  return toRet;
}

/* Returns r, mu, mu_l, and nu. */
float4 unmapSingleScatteringCoordinates(float u_r, float u_mu, float u_mu_l,
  float u_nu, float atmosphereRadius, float planetRadius) {

  float2 r_mu = unmapTransmittanceCoordinates(u_r, u_mu, atmosphereRadius,
    planetRadius);

  /* TODO: this is an ad hoc mapping. Could be improved! */
  float mu_l = clampCosine((log(1.0 - (u_mu_l * (1 - exp(-3.6)))) + 0.6) / -3.0);

  float nu = clampCosine((u_nu * 2.0) - 1.0);

  // naive mapping of u_mu and u_mu_l:
  /* TODO: make togglable? could slow things down a lot! but we already have
   * a branch up there ^ anyway... */
  // mu_l = 2.0 * u_mu_l - 1.0;
  // mu = 2.0 * u_mu - 1.0;

  return float4(r_mu.x, r_mu.y, mu_l, nu);
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

/* Generates linear location from a sample index.
 * Returns (sample, ds). */
float2 generateLinearSampleFromIndex(int i, int numberOfSamples) {
  return float2((float(i) + 0.5) / float(numberOfSamples),
    1.0 / ((float) numberOfSamples));
}

/* Generates cubed "importance sample" location from a sample index.
 * Returns (sample, ds). TODO: this is a hack. Should really use an
 * actual importance sampler. */
float2 generateCubicSampleFromIndex(int i, int numberOfSamples) {
  float t_left = float(i) / float(_numberOfSamples);
  float t_middle = (float(i) + 0.5) / float(numberOfSamples);
  float t_right = (float(i) + 1.0) / float(numberOfSamples);
  t_left *= t_left * t_left;
  t_middle *= t_middle * t_middle;
  t_right *= t_right * t_right;
  return float2(t_middle, t_right - t_left);
}

/* Generates importance sample location for an exponential distribution from a
 * sample index and the distribution parameters.
 *
 * h_min and h_max are the minimum and maximum heights of the sample range.
 *
 * Returns (sample, ds). */
float2 generateExponentialSampleFromIndex(int i, int numberOfSamples, float3
  startPoint, float3 endPoint, float planetR, float scaleHeight) {
  float3 p = startPoint;
  float3 d = endPoint - startPoint;
  float R = planetR;
  float H = scaleHeight;
  float pp = dot(p, p);
  float pd = dot(p, d);
  float dd = dot(d, d);

  float u_min = exp(-(length(startPoint) - R)/H);
  float u_max = exp(-(length(endPoint) - R)/H);
  float u_diff = u_max - u_min;

  float u_left = u_min
    + (float(i) / float(numberOfSamples)) * u_diff;
  float u_middle = u_min
    + ((float(i) + 0.5) / float(numberOfSamples)) * u_diff;
  float u_right = u_min
    + ((float(i) + 1.0) / float(numberOfSamples)) * u_diff;

  float A = dd;
  float B = 2 * pd;
  float3 C = float3(pp - ((R - H * log(u_left)) * (R - H * log(u_left))),
                    pp - ((R - H * log(u_middle)) * (R - H * log(u_middle))),
                    pp - ((R - H * log(u_right)) * (R - H * log(u_right))));

  float3 discriminant = B * B - 4 * A * C;
  /* Pretty sure this can never happen. */
  if (discriminant.x < 0.0 || discriminant.y < 0.0
    || discriminant.z < 0.0) {
    return generateLinearSampleFromIndex(i, numberOfSamples);
  }

  float t_local_minimum = -pd / dd;
  float u_local_minimum = exp(-(length(p + t_local_minimum * d) - R)/H);

  float3 t_hi = (-B + sqrt(discriminant)) / (2.0 * A);
  float3 t_lo = (-B - sqrt(discriminant)) / (2.0 * A);

  float3 t = float3(0, 0, 0);

  if (floatLT(t_local_minimum, 0.0)) {
    t = t_hi;
  } else if (floatGT(t_local_minimum, 1.0)) {
    t = t_lo;
  } else {
    // return generateCubicSampleFromIndex(i, numberOfSamples);
    if (u_left > u_local_minimum) {
      t = t_lo.x;
    } else {
      t = t_hi.x;
    }
    if (u_middle > u_local_minimum) {
      t.y = t_lo.y;
    } else {
      t.y = t_hi.y;
    }
    if (u_right > u_local_minimum) {
      t.z = t_lo.z;
    } else {
      t.z = t_hi.z;
    }
  }

  return saturate(float2(t.y, t.z - t.x));
}

/* TODO: set up importance sampling for these optical depth functions? */
/* Computes the optical depth for an exponentially distributed layer. */
float computeOpticalDepthExponential(float3 originPoint, float3 samplePoint,
  float planetR, float scaleHeight, float density) {
  // Evaluate integral over curved planet with a midpoint integrator.
  float3 d = samplePoint - originPoint;
  float length_d = length(d);
  float acc = 0.0;
  float h_min = length(originPoint) - planetR;
  float h_max = length(samplePoint) - planetR;
  for (int i = 0; i < _numberOfSamples; i++) {
    /* Compute where along the ray we're going to sample. */
    float2 t_ds = float2(0, 0);
    if (_useImportanceSampling) {
      if (_useCubicApproximation) {
        t_ds = generateCubicSampleFromIndex(i, _numberOfSamples);
      } else {
        t_ds = generateExponentialSampleFromIndex(i, _numberOfSamples,
          originPoint, samplePoint, planetR, scaleHeight);
      }
    } else {
      t_ds = generateLinearSampleFromIndex(i, _numberOfSamples);
    }

    /* Compute the point we're going to sample at. */
    float3 pt = originPoint + (d * t_ds.x);

    /* Accumulate the density at that point. */
    acc += computeDensityExponential(pt, planetR, scaleHeight, density)
      * t_ds.y * length_d;
  }
  return acc;
}

/* TODO: set up importance sampling for these optical depth functions? */
/* Computes the optical depth for a layer distributed as a tent
 * function at specified height with specified thickness. */
float computeOpticalDepthTent(float3 originPoint, float3 samplePoint,
  float planetR, float height, float thickness, float density) {
  // Evaluate integral over curved planet with a midpoint integrator.
  float3 d = samplePoint - originPoint;
  float length_d = length(d);
  float acc = 0.0;
  for (int i = 0; i < _numberOfSamples; i++) {
    /* Compute where along the ray we're going to sample. */
    float2 t_ds = float2(0, 0);
    if (_useImportanceSampling) {
      t_ds = generateCubicSampleFromIndex(i, _numberOfSamples);
    } else {
      t_ds = generateLinearSampleFromIndex(i, _numberOfSamples);
    }

    /* Compute the point we're going to sample at. */
    float3 pt = originPoint + (d * t_ds.x);

    /* Accumulate the density at that point. */
    acc += computeDensityTent(pt, planetR, height, thickness, density)
      * t_ds.y * length_d;
  }
  return acc;
}

#endif // EXPANSE_SKY_COMMON_INCLUDED
