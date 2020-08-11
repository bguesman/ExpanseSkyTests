using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

[VolumeComponentMenu("Sky/Expanse Sky")]
// SkyUniqueID does not need to be part of built-in HDRP SkyType enumeration.
// This is only provided to track IDs used by HDRP natively.
// You can use any integer value.
[SkyUniqueID(EXPANSE_SKY_UNIQUE_ID)]
public class ExpanseSky : SkySettings
{
  const int EXPANSE_SKY_UNIQUE_ID = 100003;

  /********************************************************************************/
  /********************************* Parameters ***********************************/
  /********************************************************************************/

  /* TODO: how do we group these nicely under titles in the UI? */

  /* Planet Parameters. */

  [Tooltip("Specify the total thickness of the atmosphere.")]
  public ClampedFloatParameter atmosphereThickness = new ClampedFloatParameter(40000, 0, 200000);

  [Tooltip("Specify the radius of the planet.")]
  public ClampedFloatParameter planetRadius = new ClampedFloatParameter(6360000, 0, 20000000);

  [Tooltip("Specify the ground albedo.")]
  public CubemapParameter groundColorTexture = new CubemapParameter(null);

  [Tooltip("Specify a color to tint the ground texture. If there is no ground texture specified, this is just the color of the ground.")]
  public ColorParameter groundTint = new ColorParameter(Color.white, hdr: false, showAlpha: false, showEyeDropper: true);

  [Tooltip("Specify emissive parts of the ground.")]
  public CubemapParameter groundEmissiveTexture = new CubemapParameter(null);

  [Tooltip("Specify a multiplier on the ground emissive texture.")]
  public ClampedFloatParameter groundEmissiveMultiplier = new ClampedFloatParameter(1.0f, 0.0f, 100000.0f);

  /* Night Sky Parameters. */
  [Tooltip("Specify the cubemap HDRP uses to render the sky.")]
  public CubemapParameter nightSkyHDRI = new CubemapParameter(null);

  [Tooltip("Specify a color to tint the night sky HDRI.")]
  public ColorParameter nightTint = new ColorParameter(Color.white, hdr: false, showAlpha: false, showEyeDropper: true);

  [Tooltip("Specify the intensity of the night sky.")]
  public ClampedFloatParameter nightIntensity = new ClampedFloatParameter(10.0f, 0.0f, 100000.0f);

  /* Aerosol Parameters. */
  [Tooltip("Specify the scattering coefficient for Mie scattering due to aerosols.")]
  public FloatParameter aerosolCoefficient = new FloatParameter(0.000021f);

  [Tooltip("Specify the scale height for aerosols. This parameterizes the density falloff of the aerosol layer as it extends toward space")]
  public ClampedFloatParameter scaleHeightAerosols = new ClampedFloatParameter(1200, 0, 100000);

  [Tooltip("Specify the anisotropy factor for Mie scattering.")]
  public ClampedFloatParameter aerosolAnisotropy = new ClampedFloatParameter(0.76f, -1.0f, 1.0f);

  [Tooltip("Controls the density of aerosols in the atmosphere. 1.0 is the density you would find on earth.")]
  public ClampedFloatParameter aerosolDensity = new ClampedFloatParameter(1.0f, 0.0f, 10.0f);

  /* Air Parameters. */
  [Tooltip("Specify the scattering coefficients for Rayleigh scattering due to air, which is wavelength dependent.")]
  public Vector3Parameter airCoefficients = new Vector3Parameter(new Vector3(0.0000058f, 0.0000135f, 0.0000331f));

  [Tooltip("Specify the scale height for air. This parameterizes the density falloff of the air layer as it extends toward space.")]
  public ClampedFloatParameter scaleHeightAir = new ClampedFloatParameter(8000, 0, 100000);

  [Tooltip("Controls the density of the air layer. 1.0 is the density you would find on earth.")]
  public ClampedFloatParameter airDensity = new ClampedFloatParameter(1.0f, 0.0f, 10.0f);

  /* Ozone Parameters. */
  [Tooltip("Specify the scattering coefficients for Rayleigh scattering due to the ozone, which is wavelength dependent.")]
  public Vector3Parameter ozoneCoefficients = new Vector3Parameter(new Vector3(0.00000206f, 0.00000498f, 0.000000214f));

  [Tooltip("Specify the thickness of the ozone layer.")]
  public ClampedFloatParameter ozoneThickness = new ClampedFloatParameter(30000, 0, 100000);

  [Tooltip("Specify the height of the ozone layer.")]
  public ClampedFloatParameter ozoneHeight = new ClampedFloatParameter(25000, 0, 100000);

  [Tooltip("Controls the density of the ozone layer. 1.0 is the density you would find on earth.")]
  public ClampedFloatParameter ozoneDensity = new ClampedFloatParameter(1.0f, 0.0f, 10.0f);

  /* Artistic overrides. */
  [Tooltip("Specify a tint to the overall sky color.")]
  public ColorParameter skyTint = new ColorParameter(Color.white, hdr: false, showAlpha: false, showEyeDropper: true);

  [Tooltip("Makes the scattering of starlight due to aerosols artificially stronger or weaker.")]
  public ClampedFloatParameter starAerosolScatterMultiplier = new ClampedFloatParameter(1.0f, 0.0f, 5.0f);

  [Tooltip("Darkens the edges of celestial bodies. A value of 1.0 is physically accurate. Higher values will darken more, lower values will darken less. A value of 0.0 will turn off the effect entirely.")]
  public ClampedFloatParameter limbDarkening = new ClampedFloatParameter(1.0f, 0.0f, 30.0f);

  /* Sampling. */
  [Tooltip("Specify the number of samples to use when raymarching.")]
  public ClampedIntParameter numberOfSamples = new ClampedIntParameter(5, 1, 30);

  [Tooltip("Specify whether or not to use importance sampling.")]
  public BoolParameter useImportanceSampling = new BoolParameter(true);

  [Tooltip("If using importance sampling, specify if a cubic approximation should be used (checked) or the proper importance sampler (unchecked). The cubic approximation actually seems to give a better result.")]
  public BoolParameter useCubicApproximation = new BoolParameter(true);

  [Tooltip("Controls amount of dithering used to reduce color banding. If this is too high, noise will be visible.")]
  public ClampedFloatParameter ditherAmount = new ClampedFloatParameter(0.01f, 0.0f, 1.0f);

  /********************************************************************************/
  /******************************* End Parameters ***********************************/
  /********************************************************************************/

  ExpanseSky()
  {
    displayName = "Expanse Sky";
  }

  public override Type GetSkyRendererType()
  {
    return typeof(ExpanseSkyRenderer);
  }

  public override int GetHashCode()
  {
    int hash = base.GetHashCode();
    unchecked
    {
      hash = hash * 23 + atmosphereThickness.value.GetHashCode();
      hash = hash * 23 + planetRadius.value.GetHashCode();
      hash = groundColorTexture.value != null ? hash * 23 + groundColorTexture.GetHashCode() : hash;
      hash = hash * 23 + groundTint.value.GetHashCode();
      hash = groundEmissiveTexture.value != null ? hash * 23 + groundEmissiveTexture.GetHashCode() : hash;
      hash = hash * 23 + groundEmissiveMultiplier.value.GetHashCode();
      hash = nightSkyHDRI.value != null ? hash * 23 + nightSkyHDRI.GetHashCode() : hash;
      hash = hash * 23 + nightTint.value.GetHashCode();
      hash = hash * 23 + nightIntensity.value.GetHashCode();
      hash = hash * 23 + aerosolCoefficient.value.GetHashCode();
      hash = hash * 23 + scaleHeightAerosols.value.GetHashCode();
      hash = hash * 23 + aerosolAnisotropy.value.GetHashCode();
      hash = hash * 23 + aerosolDensity.value.GetHashCode();
      hash = hash * 23 + airCoefficients.value.GetHashCode();
      hash = hash * 23 + scaleHeightAir.value.GetHashCode();
      hash = hash * 23 + airDensity.value.GetHashCode();
      hash = hash * 23 + ozoneCoefficients.value.GetHashCode();
      hash = hash * 23 + ozoneThickness.value.GetHashCode();
      hash = hash * 23 + ozoneHeight.value.GetHashCode();
      hash = hash * 23 + ozoneDensity.value.GetHashCode();
      hash = hash * 23 + skyTint.value.GetHashCode();
      hash = hash * 23 + starAerosolScatterMultiplier.value.GetHashCode();
      hash = hash * 23 + limbDarkening.value.GetHashCode();
      hash = hash * 23 + numberOfSamples.value.GetHashCode();
      hash = hash * 23 + useImportanceSampling.value.GetHashCode();
      hash = hash * 23 + useCubicApproximation.value.GetHashCode();
      hash = hash * 23 + ditherAmount.value.GetHashCode();
    }
    return hash;
  }

  public int GetPrecomputationHashCode()
  {
    int hash = base.GetHashCode();
    unchecked
    {
      hash = hash * 23 + atmosphereThickness.value.GetHashCode();
      hash = hash * 23 + planetRadius.value.GetHashCode();
      hash = hash * 23 + aerosolCoefficient.value.GetHashCode();
      hash = hash * 23 + scaleHeightAerosols.value.GetHashCode();
      hash = hash * 23 + aerosolDensity.value.GetHashCode();
      hash = hash * 23 + airCoefficients.value.GetHashCode();
      hash = hash * 23 + scaleHeightAir.value.GetHashCode();
      hash = hash * 23 + airDensity.value.GetHashCode();
      hash = hash * 23 + ozoneCoefficients.value.GetHashCode();
      hash = hash * 23 + ozoneThickness.value.GetHashCode();
      hash = hash * 23 + ozoneHeight.value.GetHashCode();
      hash = hash * 23 + ozoneDensity.value.GetHashCode();
      hash = hash * 23 + numberOfSamples.value.GetHashCode();
      hash = hash * 23 + useImportanceSampling.value.GetHashCode();
      hash = hash * 23 + useCubicApproximation.value.GetHashCode();
    }
    return hash;
  }
}
