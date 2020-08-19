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

  /* Planet Parameters. */
  [Tooltip("The total thickness of the atmosphere, in meters.")]
  public ClampedFloatParameter atmosphereThickness = new ClampedFloatParameter(40000, 0, 200000);

  [Tooltip("The radius of the planet, in meters.")]
  public ClampedFloatParameter planetRadius = new ClampedFloatParameter(6360000, 10, 20000000);

  [Tooltip("The ground albedo as a cubemap texture. The ground is modeled as a Lambertian (completely diffuse) reflector. If no texture is specified, the color of the ground will just be the ground tint.")]
  public CubemapParameter groundAlbedoTexture = new CubemapParameter(null);

  [Tooltip("A color tint to the ground texture. Perfect grey, (128, 128, 128), specifies no tint. If there is no ground texture specified, this is just the color of the ground.")]
  public ColorParameter groundTint = new ColorParameter(Color.grey, hdr: false, showAlpha: false, showEyeDropper: true);

  [Tooltip("The ground emission as a cubemap texture. Useful for modeling things like city lights. Has no effect on the sky. See \"Light Pollution\" for a way of modeling an emissive ground's effect on the atmosphere.")]
  public CubemapParameter groundEmissionTexture = new CubemapParameter(null);

  [Tooltip("An intensity multiplier on the ground emission texture.")]
  public ClampedFloatParameter groundEmissionMultiplier = new ClampedFloatParameter(1.0f, 0.0f, 100000.0f);

  [Tooltip("The rotation of the planet textures as euler angles. This won't do anything to light directions, star rotations, etc. It is purely for rotating the planet's albedo and emissive textures.")]
  public Vector3Parameter planetRotation = new Vector3Parameter(new Vector3(0.0f, 0.0f, 0.0f));

  /* Night Sky Parameters. */
  [Tooltip("The cubemap texture used to render stars and nebulae.")]
  public CubemapParameter nightSkyTexture = new CubemapParameter(null);

  [Tooltip("The rotation of the night sky texture as euler angles.")]
  public Vector3Parameter nightSkyRotation = new Vector3Parameter(new Vector3(0.0f, 0.0f, 0.0f));

  [Tooltip("A color tint to the night sky texture. Perfect grey, (128, 128, 128), specifies no tint. If there is no night sky texture specified, this is just the color of the night sky.")]
  public ColorParameter nightTint = new ColorParameter(Color.grey, hdr: false, showAlpha: false, showEyeDropper: true);

  [Tooltip("An intensity multiplier on the night sky texture. Physical luminance values of stars on Earth are very close to zero. However, this rarely plays well with auto-exposure settings.")]
  public ClampedFloatParameter nightIntensity = new ClampedFloatParameter(10.0f, 0.0f, 100000.0f);

  [Tooltip("The color of light pollution from emissive elements on the ground, i.e. city lights, cars, buildings")]
  public ColorParameter lightPollutionTint = new ColorParameter(Color.white, hdr: false, showAlpha: false, showEyeDropper: true);

  [Tooltip("The intensity of light pollution from emissive elements on the ground, i.e. city lights, cars, buildings.")]
  public ClampedFloatParameter lightPollutionIntensity = new ClampedFloatParameter(0.0f, 0.0f, 10000.0f);

  /* Aerosol Parameters. */
  [Tooltip("The scattering coefficient for Mie scattering due to aerosols at sea level. The value on Earth is 0.000021.")]
  public FloatParameter aerosolCoefficient = new FloatParameter(0.000021f);

  [Tooltip("The scale height for aerosols. This parameterizes the density falloff of the aerosol layer as it extends toward space. Adjusting this and the aerosol scale height can give the sky a dusty or foggy look.")]
  public ClampedFloatParameter scaleHeightAerosols = new ClampedFloatParameter(1200, 0, 100000);

  [Tooltip("The anisotropy factor for Mie scattering. 0.76 is a reasonable value for Earth. 1.0 specifies fully directional toward the light source, and -1.0 specifies fully directional away from the light source.")]
  public ClampedFloatParameter aerosolAnisotropy = new ClampedFloatParameter(0.76f, -1.0f, 1.0f);

  [Tooltip("The density of aerosols in the atmosphere. 1.0 is the density you would find on Earth. Adjusting this and the aerosol scale height can give the sky a dusty or foggy look.")]
  public ClampedFloatParameter aerosolDensity = new ClampedFloatParameter(1.0f, 0.0f, 10.0f);

  /* Air Parameters. */
  [Tooltip("The scattering coefficients for wavelength dependent Rayleigh scattering due to air at sea level. Adjusting this can subtly can model changes in the gas composition of the air on the Earth. Adjusting it dramatically will take you into the territory of alien skies.")]
  public Vector3Parameter airCoefficients = new Vector3Parameter(new Vector3(0.0000058f, 0.0000135f, 0.0000331f));

  [Tooltip("The scale height for air. This parameterizes the density falloff of the air layer as it extends toward space.")]
  public ClampedFloatParameter scaleHeightAir = new ClampedFloatParameter(8000, 0, 100000);

  [Tooltip("The density of the air. 1.0 is the density you would find on Earth.")]
  public ClampedFloatParameter airDensity = new ClampedFloatParameter(1.0f, 0.0f, 10.0f);

  /* Ozone Parameters. */
  [Tooltip("The scattering coefficients for wavelength dependent Rayleigh scattering due to the ozone at sea level.")]
  public Vector3Parameter ozoneCoefficients = new Vector3Parameter(new Vector3(0.00000206f, 0.00000498f, 0.000000214f));

  [Tooltip("The thickness of the ozone layer.")]
  public ClampedFloatParameter ozoneThickness = new ClampedFloatParameter(30000, 0, 100000);

  [Tooltip("The height of the ozone layer.")]
  public ClampedFloatParameter ozoneHeight = new ClampedFloatParameter(25000, 0, 100000);

  [Tooltip("Controls the density of the ozone layer. Anywhere between 0.0 and 1.0 is reasonable for a density you would find on Earth. Pushing this higher will deepen the blue of the daytime sky, and further saturate the vibrant colors of sunsets and sunrises.")]
  public ClampedFloatParameter ozoneDensity = new ClampedFloatParameter(0.3f, 0.0f, 10.0f);

  /* Artistic overrides. */
  [Tooltip("A tint to the overall sky color. Perfect grey, (128, 128, 128), specifies no tint.")]
  public ColorParameter skyTint = new ColorParameter(Color.grey, hdr: false, showAlpha: false, showEyeDropper: true);

  [Tooltip("A multiplier on the multiple scattering contribution. 1.0 is physically accurate. Pushing this above 1.0 can make the daytime sky brighter and more vibrant.")]
  public ClampedFloatParameter multipleScatteringMultiplier = new ClampedFloatParameter(1.0f, 0.0f, 30.0f);

  /* Celestial Bodies. TODO: as of now, we support up to 4 celestial bodies.
   * Settings such as angular diameter, angular position, distance,
   * color, intensity, and surface tint are specified in the directional light
   * object. But other settings, like limb darkening, cubemap textures,
   * and whether the body is a sun or a moon have no parameter in the
   * directional light itself, and so must be specified here if we aren't
   * going to hack the Unity base code. */
  [Tooltip("Darkens the edges of emissive celestial bodies. A value of 1.0 is "
  + "physically accurate for emissive celestial bodies. A value of 0.0 will turn off the effect "
  + "entirely. Higher values will darken more, lower values will darken less.")]
  public ClampedFloatParameter body1LimbDarkening = new ClampedFloatParameter(1.0f, 0.0f, 30.0f);
  [Tooltip("Whether the celestial body receives light from other celestial bodies.")]
  public BoolParameter body1ReceivesLight = new BoolParameter(false);
  [Tooltip("The texture for the surface albedo of the celestial body. If no texture is specified, the surface tint (specified in the corresponding directional light) will be used.")]
  public CubemapParameter body1AlbedoTexture = new CubemapParameter(null);
  [Tooltip("Whether the celestial body is emissive.")]
  public BoolParameter body1Emissive = new BoolParameter(true);
  [Tooltip("The texture for emission of the celestial body. If no texture is specified, the surface tint (specified in the corresponding directional light) will be used.")]
  public CubemapParameter body1EmissionTexture = new CubemapParameter(null);
  [Tooltip("The rotation of the albedo and emission textures for the celestial body.")]
  public Vector3Parameter body1Rotation = new Vector3Parameter(new Vector3(0.0f, 0.0f, 0.0f));

  [Tooltip("Darkens the edges of emissive celestial bodies. A value of 1.0 is "
  + "physically accurate for emissive celestial bodies. A value of 0.0 will turn off the effect "
  + "entirely. Higher values will darken more, lower values will darken less.")]
  public ClampedFloatParameter body2LimbDarkening = new ClampedFloatParameter(1.0f, 0.0f, 30.0f);
  [Tooltip("Whether the celestial body receives light from other celestial bodies.")]
  public BoolParameter body2ReceivesLight = new BoolParameter(false);
  [Tooltip("The texture for the surface albedo of the celestial body. If no texture is specified, the surface tint (specified in the corresponding directional light) will be used.")]
  public CubemapParameter body2AlbedoTexture = new CubemapParameter(null);
  [Tooltip("Whether the celestial body is emissive.")]
  public BoolParameter body2Emissive = new BoolParameter(true);
  [Tooltip("The texture for emission of the celestial body. If no texture is specified, the surface tint (specified in the corresponding directional light) will be used.")]
  public CubemapParameter body2EmissionTexture = new CubemapParameter(null);
  [Tooltip("The rotation of the albedo and emission textures for the celestial body.")]
  public Vector3Parameter body2Rotation = new Vector3Parameter(new Vector3(0.0f, 0.0f, 0.0f));

  [Tooltip("Darkens the edges of emissive celestial bodies. A value of 1.0 is "
  + "physically accurate for emissive celestial bodies. A value of 0.0 will turn off the effect "
  + "entirely. Higher values will darken more, lower values will darken less.")]
  public ClampedFloatParameter body3LimbDarkening = new ClampedFloatParameter(1.0f, 0.0f, 30.0f);
  [Tooltip("Whether the celestial body receives light from other celestial bodies.")]
  public BoolParameter body3ReceivesLight = new BoolParameter(false);
  [Tooltip("The texture for the surface albedo of the celestial body. If no texture is specified, the surface tint (specified in the corresponding directional light) will be used.")]
  public CubemapParameter body3AlbedoTexture = new CubemapParameter(null);
  [Tooltip("Whether the celestial body is emissive.")]
  public BoolParameter body3Emissive = new BoolParameter(true);
  [Tooltip("The texture for emission of the celestial body. If no texture is specified, the surface tint (specified in the corresponding directional light) will be used.")]
  public CubemapParameter body3EmissionTexture = new CubemapParameter(null);
  [Tooltip("The rotation of the albedo and emission textures for the celestial body.")]
  public Vector3Parameter body3Rotation = new Vector3Parameter(new Vector3(0.0f, 0.0f, 0.0f));

  [Tooltip("Darkens the edges of emissive celestial bodies. A value of 1.0 is "
  + "physically accurate for emissive celestial bodies. A value of 0.0 will turn off the effect "
  + "entirely. Higher values will darken more, lower values will darken less.")]
  public ClampedFloatParameter body4LimbDarkening = new ClampedFloatParameter(1.0f, 0.0f, 30.0f);
  [Tooltip("Whether the celestial body receives light from other celestial bodies.")]
  public BoolParameter body4ReceivesLight = new BoolParameter(false);
  [Tooltip("The texture for the surface albedo of the celestial body. If no texture is specified, the surface tint (specified in the corresponding directional light) will be used.")]
  public CubemapParameter body4AlbedoTexture = new CubemapParameter(null);
  [Tooltip("Whether the celestial body is emissive.")]
  public BoolParameter body4Emissive = new BoolParameter(true);
  [Tooltip("The texture for emission of the celestial body. If no texture is specified, the surface tint (specified in the corresponding directional light) will be used.")]
  public CubemapParameter body4EmissionTexture = new CubemapParameter(null);
  [Tooltip("The rotation of the albedo and emission textures for the celestial body.")]
  public Vector3Parameter body4Rotation = new Vector3Parameter(new Vector3(0.0f, 0.0f, 0.0f));

  /* Sampling. */
  [Tooltip("The number of samples used when computing transmittance lookup tables. With importance sampling turned on, a value of as low as 10 gives near-perfect results on the ground. A value as low as 4 is ok if some visible inaccuracy is tolerable. Without importantance sampling, a value of 32 or higher is recommended.")]
  public ClampedIntParameter numberOfTransmittanceSamples = new ClampedIntParameter(10, 1, 256);

  [Tooltip("The number of samples used when computing light pollution. With importance sampling turned on, a value of as low as 10 gives near-perfect results on the ground. A value as low as 8 is ok if some visible inaccuracy is tolerable. Without importantance sampling, a value of 64 or higher is recommended.")]
  public ClampedIntParameter numberOfLightPollutionSamples = new ClampedIntParameter(10, 1, 256);

  [Tooltip("The number of samples used when computing single scattering. With importance sampling turned on, a value of as low as 10 gives near-perfect results on the ground. A value as low as 5 is ok if some visible inaccuracy is tolerable. Without importantance sampling, a value of 32 or higher is recommended.")]
  public ClampedIntParameter numberOfScatteringSamples = new ClampedIntParameter(10, 1, 256);

  [Tooltip("The number of samples used when sampling the ground irradiance. Importance sampling does not apply here. To get a near-perfect result, around 10 samples is necessary. But it is a fairly subtle effect, so as low as 6 samples gives a decent result.")]
  public ClampedIntParameter numberOfGroundIrradianceSamples = new ClampedIntParameter(10, 1, 256);

  [Tooltip("The number of samples to use when computing the initial isotropic estimate of multiple scattering. Importance sampling does not apply here. To get a near-perfect result, around 15 samples is necessary. But it is a fairly subtle effect, so as low as 6 samples gives a decent result.")]
  public ClampedIntParameter numberOfMultipleScatteringSamples = new ClampedIntParameter(10, 1, 256);

  [Tooltip("The number of samples to use when computing the actual accumulated estimate of multiple scattering from the isotropic estimate. The number of samples to use when computing the initial isotropic estimate of multiple scattering. With importance sample, 8 samples gives a near-perfect result. However, multiple scattering is a fairly subtle effect, so as low as 3 samples gives a decent result. Without importance sampling, a value of 32 or higher is necessary for near perfect results, but a value of 4 is sufficient for most needs.")]
  public ClampedIntParameter numberOfMultipleScatteringAccumulationSamples = new ClampedIntParameter(10, 1, 256);

  [Tooltip("Whether or not to use importance sampling. Importance sampling is a sample distribution strategy that increases fidelity given a limited budget of samples. It is recommended to turn it on, as it doesn't decrease fidelity, but does allow for fewer samples to be taken, boosting performance. However, for outer-space perspectives, it can sometimes introduce inaccuracies, so it can be useful to increase sample counts and turn off importance sampling in those cases.")]
  public BoolParameter useImportanceSampling = new BoolParameter(true);

  [Tooltip("Whether or not to use MSAA 8x anti-aliasing. This does negatively affect performance.")]
  public BoolParameter useAntiAliasing = new BoolParameter(true);

  [Tooltip("Amount of dithering used to reduce color banding. If this is too high, noise will be visible.")]
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
