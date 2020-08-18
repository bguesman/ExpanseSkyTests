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
  /* TODO: when we finalize these, write nice descriptions. */

  /* Planet Parameters. */

  [Tooltip("Specify the total thickness of the atmosphere.")]
  public ClampedFloatParameter atmosphereThickness = new ClampedFloatParameter(40000, 0, 200000);

  [Tooltip("Specify the radius of the planet.")]
  public ClampedFloatParameter planetRadius = new ClampedFloatParameter(6360000, 0, 20000000);

  [Tooltip("Specify the ground albedo.")]
  public CubemapParameter groundAlbedoTexture = new CubemapParameter(null);

  [Tooltip("Specify a color to tint the ground texture. If there is no ground texture specified, this is just the color of the ground.")]
  public ColorParameter groundTint = new ColorParameter(Color.grey, hdr: false, showAlpha: false, showEyeDropper: true);

  [Tooltip("Specify emissive parts of the ground.")]
  public CubemapParameter groundEmissionTexture = new CubemapParameter(null);

  [Tooltip("Specify a multiplier on the ground emission texture.")]
  public ClampedFloatParameter groundEmissionMultiplier = new ClampedFloatParameter(1.0f, 0.0f, 100000.0f);

  [Tooltip("Specify a color to tint to the light pollution.")]
  public ColorParameter lightPollutionTint = new ColorParameter(Color.white, hdr: false, showAlpha: false, showEyeDropper: true);

  [Tooltip("Specify a multiplier on the ground emission texture.")]
  public ClampedFloatParameter lightPollutionIntensity = new ClampedFloatParameter(0.0f, 0.0f, 10000.0f);

  [Tooltip("Specify the rotation of the planet textures as euler angles. This won't do anything to light directions, star rotations, etc. It is purely for rotating the planet's albedo and emissive textures.")]
  public Vector3Parameter planetRotation = new Vector3Parameter(new Vector3(0.0f, 0.0f, 0.0f));

  /* Night Sky Parameters. */
  [Tooltip("Specify the cubemap HDRP uses to render the sky.")]
  public CubemapParameter nightSkyTexture = new CubemapParameter(null);

  [Tooltip("Specify the rotation of the night sky as euler angles.")]
  public Vector3Parameter nightSkyRotation = new Vector3Parameter(new Vector3(0.0f, 0.0f, 0.0f));

  [Tooltip("Specify a color to tint the night sky HDRI.")]
  public ColorParameter nightTint = new ColorParameter(Color.grey, hdr: false, showAlpha: false, showEyeDropper: true);

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
  public ColorParameter skyTint = new ColorParameter(Color.grey, hdr: false, showAlpha: false, showEyeDropper: true);

  [Tooltip("Multiplier on multiple scattering")]
  public ClampedFloatParameter multipleScatteringMultiplier = new ClampedFloatParameter(1.0f, 0.0f, 30.0f);

  /* Celestial Bodies. TODO: as of now, we support up to 4 celestial bodies.
   * Settings such as angular diameter, angular position, distance,
   * color, intensity, and surface tint are specified in the directional light
   * object. But other settings, like limb darkening, cubemap textures,
   * and whether the body is a sun or a moon have no parameter in the
   * directional light itself, and so must be specified here if we aren't
   * going to hack the Unity base code. */
  [Tooltip("Darkens the edges of emissive celestial bodies. A value of 1.0 is "
  + "physically accurate for emissive bodies. A value of 0.0 will turn off the effect "
  + "entirely. Higher values will darken more, lower values will darken less.")]
  public ClampedFloatParameter body1LimbDarkening = new ClampedFloatParameter(1.0f, 0.0f, 30.0f);
  [Tooltip("Specifies whether the body receives light from other celestial bodies.")]
  public BoolParameter body1ReceivesLight = new BoolParameter(false);
  [Tooltip("Specifies texture for surface albedo of celestial body #1.")]
  public CubemapParameter body1AlbedoTexture = new CubemapParameter(null);
  [Tooltip("Specifies whether the body is emissive.")]
  public BoolParameter body1Emissive = new BoolParameter(true);
  [Tooltip("Specifies texture for surface albedo of celestial body #1.")]
  public CubemapParameter body1EmissionTexture = new CubemapParameter(null);
  [Tooltip("Specifies the rotation of the albedo and emission textures for celestial body #1.")]
  public Vector3Parameter body1Rotation = new Vector3Parameter(new Vector3(0.0f, 0.0f, 0.0f));

  [Tooltip("Darkens the edges of emissive celestial bodies. A value of 1.0 is "
  + "physically accurate for emissive bodies. A value of 0.0 will turn off the effect "
  + "entirely. Higher values will darken more, lower values will darken less.")]
  public ClampedFloatParameter body2LimbDarkening = new ClampedFloatParameter(1.0f, 0.0f, 30.0f);
  [Tooltip("Specifies whether the body receives light from other celestial bodies.")]
  public BoolParameter body2ReceivesLight = new BoolParameter(false);
  [Tooltip("Specifies texture for surface albedo of celestial body #2.")]
  public CubemapParameter body2AlbedoTexture = new CubemapParameter(null);
  [Tooltip("Specifies whether the body is emissive.")]
  public BoolParameter body2Emissive = new BoolParameter(true);
  [Tooltip("Specifies texture for surface albedo of celestial body #2.")]
  public CubemapParameter body2EmissionTexture = new CubemapParameter(null);
  [Tooltip("Specifies the rotation of the albedo and emission textures for celestial body #2.")]
  public Vector3Parameter body2Rotation = new Vector3Parameter(new Vector3(0.0f, 0.0f, 0.0f));

  [Tooltip("Darkens the edges of emissive celestial bodies. A value of 1.0 is "
  + "physically accurate for emissive bodies. A value of 0.0 will turn off the effect "
  + "entirely. Higher values will darken more, lower values will darken less.")]
  public ClampedFloatParameter body3LimbDarkening = new ClampedFloatParameter(1.0f, 0.0f, 30.0f);
  [Tooltip("Specifies whether the body receives light from other celestial bodies.")]
  public BoolParameter body3ReceivesLight = new BoolParameter(false);
  [Tooltip("Specifies texture for surface albedo of celestial body #3.")]
  public CubemapParameter body3AlbedoTexture = new CubemapParameter(null);
  [Tooltip("Specifies whether the body is emissive.")]
  public BoolParameter body3Emissive = new BoolParameter(true);
  [Tooltip("Specifies texture for surface albedo of celestial body #3.")]
  public CubemapParameter body3EmissionTexture = new CubemapParameter(null);
  [Tooltip("Specifies the rotation of the albedo and emission textures for celestial body #2.")]
  public Vector3Parameter body3Rotation = new Vector3Parameter(new Vector3(0.0f, 0.0f, 0.0f));

  [Tooltip("Darkens the edges of emissive celestial bodies. A value of 1.0 is "
  + "physically accurate for emissive bodies. A value of 0.0 will turn off the effect "
  + "entirely. Higher values will darken more, lower values will darken less.")]
  public ClampedFloatParameter body4LimbDarkening = new ClampedFloatParameter(1.0f, 0.0f, 30.0f);
  [Tooltip("Specifies whether the body receives light from other celestial bodies.")]
  public BoolParameter body4ReceivesLight = new BoolParameter(false);
  [Tooltip("Specifies texture for surface albedo of celestial body #4.")]
  public CubemapParameter body4AlbedoTexture = new CubemapParameter(null);
  [Tooltip("Specifies whether the body is emissive.")]
  public BoolParameter body4Emissive = new BoolParameter(true);
  [Tooltip("Specifies texture for surface albedo of celestial body #4.")]
  public CubemapParameter body4EmissionTexture = new CubemapParameter(null);
  [Tooltip("Specifies the rotation of the albedo and emission textures for celestial body #2.")]
  public Vector3Parameter body4Rotation = new Vector3Parameter(new Vector3(0.0f, 0.0f, 0.0f));

  /* Sampling. */
  [Tooltip("Specify the number of samples to use when sampling along the primary ray.")]
  public ClampedIntParameter numberOfTransmittanceSamples = new ClampedIntParameter(10, 1, 256);

  [Tooltip("Specify the number of samples to use when sampling along the secondary ray to compute light pollution.")]
  public ClampedIntParameter numberOfLightPollutionSamples = new ClampedIntParameter(10, 1, 256);

  [Tooltip("Specify the number of samples to use when sampling along the secondary ray in the single scattering computation.")]
  public ClampedIntParameter numberOfScatteringSamples = new ClampedIntParameter(10, 1, 256);

  [Tooltip("Specify the number of samples to use when sampling the ground irradiance.")]
  public ClampedIntParameter numberOfGroundIrradianceSamples = new ClampedIntParameter(10, 1, 256);

  [Tooltip("Specify the number of samples to use when computing the initial isotropic estimate of multiple scattering.")]
  public ClampedIntParameter numberOfMultipleScatteringSamples = new ClampedIntParameter(10, 1, 256);

  [Tooltip("Specify the number of samples to use when computing the actual accumulated estimate of multiple scattering from the isotropic estimate.")]
  public ClampedIntParameter numberOfMultipleScatteringAccumulationSamples = new ClampedIntParameter(10, 1, 256);

  [Tooltip("Specify whether or not to use importance sampling.")]
  public BoolParameter useImportanceSampling = new BoolParameter(true);

  [Tooltip("Specify whether or not to use anti-aliasing. MSAA 8x.")]
  public BoolParameter useAntiAliasing = new BoolParameter(true);

  [Tooltip("Controls amount of dithering used to reduce color banding. If this is too high, noise will be visible.")]
  public ClampedFloatParameter ditherAmount = new ClampedFloatParameter(0.05f, 0.0f, 1.0f);

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
      /* Planet. */
      hash = hash * 23 + atmosphereThickness.value.GetHashCode();
      hash = hash * 23 + planetRadius.value.GetHashCode();
      hash = groundAlbedoTexture.value != null ? hash * 23 + groundAlbedoTexture.GetHashCode() : hash;
      hash = hash * 23 + groundTint.value.GetHashCode();
      hash = groundEmissionTexture.value != null ? hash * 23 + groundEmissionTexture.GetHashCode() : hash;
      hash = hash * 23 + groundEmissionMultiplier.value.GetHashCode();
      hash = hash * 23 + lightPollutionTint.value.GetHashCode();
      hash = hash * 23 + lightPollutionIntensity.value.GetHashCode();
      hash = hash * 23 + planetRotation.value.GetHashCode();

      /* Night sky. */
      hash = nightSkyTexture.value != null ? hash * 23 + nightSkyTexture.GetHashCode() : hash;
      hash = hash * 23 + nightSkyRotation.value.GetHashCode();
      hash = hash * 23 + nightTint.value.GetHashCode();
      hash = hash * 23 + nightIntensity.value.GetHashCode();

      /* Aerosols. */
      hash = hash * 23 + aerosolCoefficient.value.GetHashCode();
      hash = hash * 23 + scaleHeightAerosols.value.GetHashCode();
      hash = hash * 23 + aerosolAnisotropy.value.GetHashCode();
      hash = hash * 23 + aerosolDensity.value.GetHashCode();

      /* Air. */
      hash = hash * 23 + airCoefficients.value.GetHashCode();
      hash = hash * 23 + scaleHeightAir.value.GetHashCode();
      hash = hash * 23 + airDensity.value.GetHashCode();

      /* Ozone. */
      hash = hash * 23 + ozoneCoefficients.value.GetHashCode();
      hash = hash * 23 + ozoneThickness.value.GetHashCode();
      hash = hash * 23 + ozoneHeight.value.GetHashCode();
      hash = hash * 23 + ozoneDensity.value.GetHashCode();

      /* Artistic overrides. */
      hash = hash * 23 + skyTint.value.GetHashCode();
      hash = hash * 23 + multipleScatteringMultiplier.value.GetHashCode();

      /* Celestial bodies. */
      hash = hash * 23 + body1LimbDarkening.value.GetHashCode();
      hash = hash * 23 + body1ReceivesLight.value.GetHashCode();
      hash = body1AlbedoTexture.value != null ? hash * 23 + body1AlbedoTexture.GetHashCode() : hash;
      hash = hash * 23 + body1Emissive.value.GetHashCode();
      hash = body1EmissionTexture.value != null ? hash * 23 + body1EmissionTexture.GetHashCode() : hash;
      hash = hash * 23 + body1Rotation.value.GetHashCode();

      hash = hash * 23 + body2LimbDarkening.value.GetHashCode();
      hash = hash * 23 + body2ReceivesLight.value.GetHashCode();
      hash = body2AlbedoTexture.value != null ? hash * 23 + body2AlbedoTexture.GetHashCode() : hash;
      hash = hash * 23 + body2Emissive.value.GetHashCode();
      hash = body2EmissionTexture.value != null ? hash * 23 + body2EmissionTexture.GetHashCode() : hash;
      hash = hash * 23 + body2Rotation.value.GetHashCode();

      hash = hash * 23 + body3LimbDarkening.value.GetHashCode();
      hash = hash * 23 + body3ReceivesLight.value.GetHashCode();
      hash = body3AlbedoTexture.value != null ? hash * 23 + body3AlbedoTexture.GetHashCode() : hash;
      hash = hash * 23 + body3Emissive.value.GetHashCode();
      hash = body3EmissionTexture.value != null ? hash * 23 + body3EmissionTexture.GetHashCode() : hash;
      hash = hash * 23 + body3Rotation.value.GetHashCode();

      hash = hash * 23 + body4LimbDarkening.value.GetHashCode();
      hash = hash * 23 + body4ReceivesLight.value.GetHashCode();
      hash = body4AlbedoTexture.value != null ? hash * 23 + body4AlbedoTexture.GetHashCode() : hash;
      hash = hash * 23 + body4Emissive.value.GetHashCode();
      hash = body4EmissionTexture.value != null ? hash * 23 + body4EmissionTexture.GetHashCode() : hash;
      hash = hash * 23 + body4Rotation.value.GetHashCode();

      /* Sampling. */
      hash = hash * 23 + numberOfTransmittanceSamples.value.GetHashCode();
      hash = hash * 23 + numberOfLightPollutionSamples.value.GetHashCode();
      hash = hash * 23 + numberOfScatteringSamples.value.GetHashCode();
      hash = hash * 23 + numberOfGroundIrradianceSamples.value.GetHashCode();
      hash = hash * 23 + numberOfMultipleScatteringSamples.value.GetHashCode();
      hash = hash * 23 + numberOfMultipleScatteringAccumulationSamples.value.GetHashCode();
      hash = hash * 23 + useImportanceSampling.value.GetHashCode();
      hash = hash * 23 + useAntiAliasing.value.GetHashCode();
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
      hash = groundAlbedoTexture.value != null ? hash * 23 + groundAlbedoTexture.GetHashCode() : hash;
      hash = hash * 23 + groundTint.value.GetHashCode();
      hash = hash * 23 + lightPollutionTint.value.GetHashCode();
      hash = hash * 23 + lightPollutionIntensity.value.GetHashCode();
      hash = hash * 23 + planetRotation.value.GetHashCode();
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
      hash = hash * 23 + numberOfTransmittanceSamples.value.GetHashCode();
      hash = hash * 23 + numberOfLightPollutionSamples.value.GetHashCode();
      hash = hash * 23 + numberOfScatteringSamples.value.GetHashCode();
      hash = hash * 23 + numberOfGroundIrradianceSamples.value.GetHashCode();
      hash = hash * 23 + numberOfMultipleScatteringSamples.value.GetHashCode();
      hash = hash * 23 + numberOfMultipleScatteringAccumulationSamples.value.GetHashCode();
      hash = hash * 23 + useImportanceSampling.value.GetHashCode();
    }
    return hash;
  }
}
