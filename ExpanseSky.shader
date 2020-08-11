/* Much of this is adapted from https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/s2016-pbs-frostbite-sky-clouds-new.pdf. */
Shader "HDRP/Sky/ExpanseSky"
{
  HLSLINCLUDE

  #pragma vertex Vert

  #pragma editor_sync_compilation
  #pragma target 4.5
  #pragma only_renderers d3d11 ps4 xboxone vulkan metal switch


  #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
  #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/CommonLighting.hlsl"
  #include "Packages/com.unity.render-pipelines.high-definition/Runtime/Sky/SkyUtils.hlsl"
  #include "Assets/CustomSky/ExpanseSky/ExpanseSkyCommon.hlsl"


  struct Attributes
  {
    uint vertexID : SV_VertexID;
    UNITY_VERTEX_INPUT_INSTANCE_ID
  };

  struct Varyings
  {
    float4 positionCS : SV_POSITION;

    UNITY_VERTEX_OUTPUT_STEREO
  };

  /* Vertex shader just sets vertex position. All the heavy lifting is
   * done in the fragment shader. */
  Varyings Vert(Attributes input)
  {
    Varyings output;
    UNITY_SETUP_INSTANCE_ID(input);
    UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
    output.positionCS = GetFullScreenTriangleVertexPosition(input.vertexID, UNITY_RAW_FAR_CLIP_VALUE);

    return output;
  }

  /* Compute the luminance of a Celestial given the illuminance and the cosine
   * of half the angular extent. */
  float3 computeCelestialBodyLuminance(float3 zenithIlluminance, float cosTheta) {
    /* Compute solid angle. */
    float solidAngle = 2.0 * PI * (1.0 - cosTheta);
    return zenithIlluminance / solidAngle;
  }

  float limbDarkening(float LdotV, float cosInner) {
    float centerToEdge = 1.0 - abs((LdotV - cosInner) / (1.0 - cosInner));
    float mu = sqrt(1.0 - centerToEdge * centerToEdge);
    float mu2 = mu * mu;
    float mu3 = mu2 * mu;
    float mu4 = mu2 * mu2;
    float mu5 = mu3 * mu2;
    float3 a0 = 1 * float3 (0.34685, 0.26073, 0.15248);
    float3 a1 = 1 * float3 (1.37539, 1.27428, 1.38517);
    float3 a2 = 1 * float3 (-2.04425, -1.30352, -1.49615);
    float3 a3 = 1 * float3 (2.70493, 1.47085, 1.99886);
    float3 a4 = 1 * float3 (-1.94290, -0.96618, -1.48155);
    float3 a5 = 1 * float3 (0.55999, 0.26384, 0.44119);
    return a0 + a1 * mu + a2 * mu2 + a3 * mu3 + a4 * mu4 + a5 * mu5;
  }

  float3 RenderSky(Varyings i, float exposure)
  {
    /* Get the origin point and sample direction. */
    float3 O = _WorldSpaceCameraPos1 - float3(0, -_planetRadius, 0);
    float3 d = normalize(-GetSkyViewDirWS(i.positionCS.xy));

    /* For efficiency, precompute the atmosphere radius. TODO: move this
    * computation to C# side and pass a uniform variable. */
    float atmosphereRadius = _planetRadius + _atmosphereThickness;

    /* See if we're looking at the ground or the sky. */
    float3 t_ground = intersectSphere(O, d, _planetRadius);
    float3 t_atmo = intersectSphere(O, d, atmosphereRadius);
    bool groundHit = t_ground.z >= 0.0 && (t_ground.x >= 0.0 || t_ground.y >= 0.0);
    bool atmoHit = t_atmo.z >= 0.0 && (t_atmo.x >= 0.0 || t_atmo.y >= 0.0);

    /* TODO: may want to get rid of this branch if possible. */
    if (!groundHit && !atmoHit) {
      /* We've hit space. Return black. */
      return float4(0.0, 0.0, 0.0, 1.0);
    }

    /* Figure out the point we're raymarching to. */
    float t_hit = 0.0;
    if (groundHit) {
      /* We've hit the ground. The point we want to raymarch to is the
      * closest positive ground hit. */
      t_hit = (t_ground.x < 0.0) ? t_ground.y :
      ((t_ground.y < 0.0) ? t_ground.x : min(t_ground.x, t_ground.y));
    } else {
      /* We've hit only the atmosphere. The point we want to raymarch to is
      * the furthest positive atmo hit, since we want to march through the
      * whole volume. */
      t_hit = max(t_atmo.x, t_atmo.y);
    }

    float3 hitPoint = O + d * t_hit;

    /* Loop through lights and accumulate direct illumination. */
    float3 L0 = float3(0, 0, 0);
    /* Put the loop inside the conditional so we only have to evaluate once. */
    if (groundHit) {
      for (int i = 0; i < _DirectionalLightCount; i++) {
        DirectionalLightData light = _DirectionalLightDatas[i];
        float3 L = -normalize(light.forward.xyz);
        float3 lightColor = light.color;
        /* Ground is just a diffuse BRDF. */
        L0 += _groundTintF3 * lightColor * (1.0 / PI) * saturate(dot(normalize(hitPoint), L));
      }
    } else {
      for (int i = 0; i < _DirectionalLightCount; i++) {
        DirectionalLightData light = _DirectionalLightDatas[i];
        float3 L = -normalize(light.forward.xyz);
        float LdotV    = dot(L, d);
        float radInner = 0.5 * light.angularDiameter;
        float cosInner = cos(radInner);
        //float cosOuter = cos(radInner + light.flareSize);
        float3 lightColor = light.color;
        float3 luminance = computeCelestialBodyLuminance(lightColor, cosInner);
        if (LdotV >= cosInner) {
          L0 += luminance * limbDarkening(LdotV, cosInner)
            * light.surfaceTint;
        }
      }
    }

    /* Compute some things for the lookup tables. */
    float r = length(O);
    float mu = clampCosine(dot(normalize(O), d));

    /* Perform the transmittance table lookup attenuating direct lighting. */
    float2 transmittanceUV = mapTransmittanceCoordinates(r,
      mu, atmosphereRadius, _planetRadius, groundHit);
    float3 T = SAMPLE_TEXTURE2D(_TransmittanceTable, s_linear_clamp_sampler,
      transmittanceUV);


    // /* Perform single scattering table lookup for the sky color.
    //  * TODO: loop over all lights and accumulate. */
    float3 skyColor = float3(0, 0, 0);
     // if (!groundHit) {
     for (int k = 0; k < 2; k++) {
       DirectionalLightData light = _DirectionalLightDatas[k];
       float3 L = -normalize(light.forward.xyz);
       float3 lightColor = light.color;

       // /* Need to compute mu_l and nu. */
       float mu_l = clampCosine(dot(normalize(O), L));

       /* Nu is the azimuth angle. */
       float3 proj_L = normalize(L - normalize(O) * mu_l);
       float3 proj_d = normalize(d - normalize(O) * dot(normalize(O), d));
       float nu  = clampCosine(dot(proj_L, proj_d));

       nu = clamp(nu, mu * mu_l - sqrt((1.0 - mu * mu) * (1.0 - mu_l * mu_l)),
           mu * mu_l + sqrt((1.0 - mu * mu) * (1.0 - mu_l * mu_l)));

       TexCoord5D ssCoord =
        mapSingleScatteringCoordinates(r, mu, mu_l, nu,
         atmosphereRadius, _planetRadius, t_hit, groundHit);

       float3 uvw0 = float3(ssCoord.x, ssCoord.y, ssCoord.z);
       float3 uvw1 = float3(ssCoord.x, ssCoord.y, ssCoord.w);

       float3 ssContrib0Air = SAMPLE_TEXTURE3D(_SingleScatteringTableAir,
         s_linear_clamp_sampler, uvw0).rgb;
       float3 ssContrib1Air = SAMPLE_TEXTURE3D(_SingleScatteringTableAir,
         s_linear_clamp_sampler, uvw1).rgb;

       float3 singleScatteringContributionAir = lerp(ssContrib0Air, ssContrib1Air, ssCoord.a);

       float3 ssContrib0Aerosol = SAMPLE_TEXTURE3D(_SingleScatteringTableAerosol,
         s_linear_clamp_sampler, uvw0).rgb;
       float3 ssContrib1Aerosol = SAMPLE_TEXTURE3D(_SingleScatteringTableAerosol,
         s_linear_clamp_sampler, uvw1).rgb;

       float3 singleScatteringContributionAerosol = lerp(ssContrib0Aerosol, ssContrib1Aerosol, ssCoord.a);

       float dot_L_d = dot(L, d);
       float rayleighPhase = 3.f / (16.f * PI) * (1 + dot_L_d * dot_L_d);
       float miePhase = 3.f / (8.0 * PI) * ((1.f - g * g) * (1.f + dot_L_d * dot_L_d))
         / ((2.f + g * g) * pow(1.f + g * g - 2.f * g * dot_L_d, 1.5f));

       float3 finalSingleScattering = (2.0 * _skyTintF3 * _airCoefficientsF3
         * singleScatteringContributionAir * rayleighPhase
         + _aerosolCoefficient * singleScatteringContributionAerosol * miePhase);

       skyColor += finalSingleScattering * lightColor;
     }
   // }

    /* Now, we perform the actual scattering accumulation. */
    // float3 skyColor = float3(0.0, 0.0, 0.0);
    // for (int k = 0; k < _DirectionalLightCount; k++) {
    //   DirectionalLightData light = _DirectionalLightDatas[k];
    //   float3 inScatteringAir = float3(0.0, 0.0, 0.0);
    //   float3 inScatteringAerosol = float3(0.0, 0.0, 0.0);
    //   float3 L = -normalize(light.forward.xyz);
    //   for (int i = 0; i < _numberOfSamples; i++) {
    //     /* TODO: make this non-linear. */
    //     float sampleT = (((float) i) + 0.5) / ((float) _numberOfSamples);
    //     float ds = t_hit / (float) (_numberOfSamples);
    //     float3 samplePoint = O + d * sampleT * t_hit;
    //
    //     /* Trace a ray from the sample point to the light to check visibility. */
    //     float3 t_light = intersectSphere(samplePoint, L, atmosphereRadius);
    //     float3 t_light_ground = intersectSphere(samplePoint, L, _planetRadius);
    //     bool lightHit = floatGT(t_light.z, 0.0)
    //       && (floatGT(t_light.x, 0.0) || floatGT(t_light.y, 0.0));
    //     bool lightGroundHit = floatGT(t_light_ground.z, 0.0)
    //      && (floatGT(t_light_ground.x, 0.0) || floatGT(t_light_ground.y, 0.0));
    //     if (lightHit && !lightGroundHit) {
    //       /* Compute the light hit point. */
    //       float3 lightHitPoint = samplePoint + L * max(t_light.x, t_light.y);
    //
    //       /* Compute the scaled densities of air and aerosol layers at the
    //        * sample point. */
    //       float scaledDensityAir = computeDensityExponential(samplePoint,
    //         _planetRadius, _scaleHeightAir, _airDensity) * ds;
    //       float scaledDensityAerosol = computeDensityExponential(samplePoint,
    //         _planetRadius, _scaleHeightAerosols, _aerosolDensity) * ds;
    //
    //       /* Compute transmittance from O to sample point, and then from sample
    //        * point through to the light hit. */
    //       float2 oToSample = mapTransmittanceCoordinates(length(O),
    //         clampCosine(dot(normalize(O), d)), atmosphereRadius, _planetRadius, groundHit);
    //       float2 sampleToL = mapTransmittanceCoordinates(length(samplePoint),
    //        clampCosine(dot(normalize(samplePoint), L)), atmosphereRadius, _planetRadius,
    //        lightGroundHit);
    //
    //       float3 T_oToSample = SAMPLE_TEXTURE2D_LOD(_TransmittanceTable,
    //         s_linear_clamp_sampler, oToSample, 0).rgb;
    //       float3 T_sampleToL = SAMPLE_TEXTURE2D_LOD(_TransmittanceTable,
    //         s_linear_clamp_sampler, sampleToL, 0).rgb;
    //
    //       /* If we didn't hit the ground, then our transmittance value for
    //        * O to the sample point is too large---we need to divide out
    //        * the transmittance from the sample point to the atmosphere. */
    //       if (!groundHit) {
    //         float2 sampleOut = mapTransmittanceCoordinates(length(samplePoint),
    //           clampCosine(dot(normalize(samplePoint), d)), atmosphereRadius, _planetRadius,
    //           false);
    //         float3 T_sampleOut = SAMPLE_TEXTURE2D_LOD(_TransmittanceTable,
    //           s_linear_clamp_sampler, sampleOut, 0).rgb;
    //         /* Clamp sample out for division. */
    //         T_sampleOut = max(T_sampleOut, FLT_EPSILON);
    //         T_oToSample /= T_sampleOut;
    //       }
    //
    //       float3 T = T_oToSample * T_sampleToL;
    //
    //       inScatteringAir += scaledDensityAir * T;
    //       inScatteringAerosol += scaledDensityAerosol * T;
    //     }
    //   }
    //
    //   /* ssCoord.w is nu, which is usually called mu in these equations. It's
    //    * dot(L, d). */
    //   float m_l = clampCosine(dot(L, d));
    //   float rayleighPhase = 3.f / (16.f * PI) * (1 + m_l * m_l);
    //   float miePhase = 3.f / (8.0 * PI) * ((1.f - g * g) * (1.f + m_l * m_l))
    //     / ((2.f + g * g) * pow(1.f + g * g - 2.f * g * m_l, 1.5f));
    //
    //   skyColor += light.color * (2.0 * _skyTintF3 * _airCoefficientsF3
    //     * inScatteringAir * rayleighPhase
    //     + _aerosolCoefficient * inScatteringAerosol * miePhase);
    // }

    float3 finalDirectLighting = (L0 * T);

    /* TODO: HACK: sky color goes negative (!!) here, clamping just as reminder
     * This must be some kind of issue with clamping some values to safe
     * values in the table generation and mapping/unmapping? May want to make
     * some helpers to clamp certain things like cosines/dot products, etc. */
    float dither = 1.0 + _ditherAmount * (0.5 - random(d.xy));
    return dither * exposure * (finalDirectLighting + skyColor + (groundHit ? 0.0 : _nightTint * _nightIntensity));

    // /* We will assume the planet center is at (0, -planetRadius, 0) in
    //  * world space. This calculation then transforms us to the planet's
    //  * frame of reference, so from here on out, we'll be working in the frame
    //  * where the planet is at the origin. "Planet space".
    //  * TODO: make this something you can specify. Not sure how to make it
    //  * easy to specify without being annoying. */
    //  float3 originPoint = _WorldSpaceCameraPos1 - float3(0, -_planetRadius, 0);
    //  float3 sampleDirection = normalize(-GetSkyViewDirWS(i.positionCS.xy));
    //
    //  /* The light direction we'll get from the first directional light. */
    //  DirectionalLightData light = _DirectionalLightDatas[0];
    //  float3 L = -normalize(light.forward.xyz);
    //  float LdotV    = dot(L, sampleDirection);
    //  float radInner = 0.5 * light.angularDiameter;
    //  float cosInner = cos(radInner);
    //  float cosOuter = cos(radInner + light.flareSize);
    //  float3 sunColor = light.color;
    //
    //  /* For convenience and efficiency, pre-compute the atmosphere radius. */
    //  float atmosphereRadius = _planetRadius + _atmosphereThickness;
    //
    //  /* We need to see if the ray hits the ground or goes off into the sky.
    //  * If it goes into the sky, we need to know where it hits the edge of
    //  * the atmosphere. */
    //  float3 t_ground = intersectSphere(originPoint, sampleDirection, _planetRadius);
    //  float3 t_atmo = intersectSphere(originPoint, sampleDirection, atmosphereRadius);
    //  bool groundHit = t_ground.z >= 0.0 && (t_ground.x >= 0.0 || t_ground.y >= 0.0);
    //  bool atmoHit = t_atmo.z >= 0.0 && (t_atmo.x >= 0.0 || t_atmo.y >= 0.0);
    //
    //  if (!groundHit && !atmoHit) {
    //    /* We've hit space. Return black. */
    //    return float4(0.0, 0.0, 0.0, 1.0);
    //  }
    //
    //  /* Initialize point we will raymarch to with a default value. */
    //  float t_hit = 0.0;
    //
    //  /* Set hit point depending on whether we hit ground or sky. */
    //  if (groundHit) {
    //    /* We've hit the ground. The point we want to raymarch to is the
    //     * closest positive ground hit. */
    //     t_hit = (t_ground.x < 0.0) ? t_ground.y :
    //       ((t_ground.y < 0.0) ? t_ground.x : min(t_ground.x, t_ground.y));
    //  } else {
    //    /* We've hit only the atmosphere. The point we want to raymarch to is
    //     * the furthest positive atmo hit, since we want to march through the
    //     * whole volume. */
    //     t_hit = max(t_atmo.x, t_atmo.y);
    //  }
    //
    //  float3 hitPoint = originPoint + sampleDirection * t_hit;
    //
    //  /* If we hit the ground, we have direct light to compute. */
    //  float3 L0 = float3(0, 0, 0);
    //  float3 sunLuminance = computeSunLuminance(sunColor, cosInner);
    //  if (groundHit) {
    //    /* We'll just use lambert's cosine law to color the ground. */
    //    L0 += saturate(dot(normalize(hitPoint), L)) * (1.0 / PI) * _groundTintF3 * sunColor; // float4(light.color, 1);
    //  } else {
    //    /* If we hit the sun, we also have to include that contribution. */
    //    if (LdotV >= cosInner) {
    //    /* The flux of the sun's light dies off around the edges of the
    //     * disc, since the sun is a sphere. This effect is called "limb
    //     * darkening". We model this here. TODO: make togglable and
    //     * come up with a strength control. Maybe a single number that
    //     * decreases a1/3 and increases a2/4. TODO: make function. */
    //     float centerToEdge = 1.0 - abs((LdotV - cosInner) / (1.0 - cosInner));
    //     float mu = sqrt(1.0 - centerToEdge * centerToEdge);
    //     float mu2 = mu * mu;
    //     float mu3 = mu2 * mu;
    //     float mu4 = mu2 * mu2;
    //     float mu5 = mu3 * mu2;
    //     float3 a0 = 1 * float3 (0.34685, 0.26073, 0.15248);
    //     float3 a1 = 1 * float3 (1.37539, 1.27428, 1.38517);
    //     float3 a2 = 1 * float3 (-2.04425, -1.30352, -1.49615);
    //     float3 a3 = 1 * float3 (2.70493, 1.47085, 1.99886);
    //     float3 a4 = 1 * float3 (-1.94290, -0.96618, -1.48155);
    //     float3 a5 = 1 * float3 (0.55999, 0.26384, 0.44119);
    //     float3 limbDarkening = a0 + a1 * mu + a2 * mu2 + a3 * mu3 + a4 * mu4 + a5 * mu5;
    //     L0 += sunLuminance * limbDarkening * light.surfaceTint;
    //    } else {
    //      /* Finally, we also have to sample the night sky texture if we don't
    //       * hit the ground. TODO: null check? What do we do if there's no
    //       * night sky texture? */
    //       L0 += _nightIntensity * (_nightTintF3 * SAMPLE_TEXTURECUBE_LOD(_nightSkyHDRI, sampler_Cubemap, sampleDirection, 0));
    //    }
    //  }
    //
    //  /* Main scattering loop. */
    //  /* Optical depth from origin to current sample point. This is a
    //   * running average. */
    //  float opticalDepthAir = 0.0;
    //  float opticalDepthOzone = 0.0;
    //  float opticalDepthAerosol = 0.0;
    //  /* In scattering contribution from air and aerosol layers of the
    //   * atmosphere. Note that we accumulate this without the phase function
    //   * or sunlight factored in since we can multiply the result by those
    //   * once after the loop. */
    //  float3 inScatteringAir = float3(0.0, 0.0, 0.0);
    //  float3 inScatteringAerosol = float3(0.0, 0.0, 0.0);
    //
    //  float3 inScatteringAirNight = float3(0.0, 0.0, 0.0);
    //  float3 inScatteringAerosolNight = float3(0.0, 0.0, 0.0);
    //  for (int i = 0; i < _numberOfSamples; i++) {
    //    /* Get the sample point along the ray. TODO: make into functions. */
    //    float3 samplePoint = float3(0, 0, 0);
    //    float ds = 0;
    //    if (_useImportanceSampling) {
    //        if (_useCubicApproximation) {
    //            /* This is a hack to approximate importance sampling with
    //             * fewer computations. In practice, it actually results in
    //             * too much scattering, and can actually be a worse approximator
    //             * than the vanilla linear one. */
    //            float left = (((float) i)) / ((float) _numberOfSamples);
    //            float middle = (((float) i) + 0.5) / ((float) _numberOfSamples);
    //            float right = (((float) i) + 1.0) / ((float) _numberOfSamples);
    //            left *= left * left;
    //            middle *= middle * middle;
    //            right *= right * right;
    //            ds = t_hit * (right - left);
    //            samplePoint = originPoint + sampleDirection * middle * t_hit;
    //        } else {
    //            /* We can derive a proper importance sampler for the atmosphere distribution
    //             * using the inversion method.
    //             * u = e^-(t/H)
    //             * ln(u) = -t/H
    //             * t = -ln(u)/H
    //             * range of t needs to be 0-1.
    //             * TODO: corrected calculation but this last line still doesn't
    //             * account for H. Need separate samples for every atmo layer.
    //             * so range of u needs to be 1-(1/e). */
    //            float u_left = (((float) i)) / ((float) _numberOfSamples);
    //            float u_middle = (((float) i) + 0.5) / ((float) _numberOfSamples);
    //            float u_right = (((float) i) + 1.0) / ((float) _numberOfSamples);
    //            /* Transform to range [1, (1/e)]. */
    //            u_left = 1.f - (1.f - 1.f/E) * u_left;
    //            u_middle = 1.f - (1.f - 1.f/E) * u_middle;
    //            u_right = 1.f - (1.f - 1.f/E) * u_right;
    //            /* Run through inverse of probability function---inversion method. */
    //            float left = -log(u_left);
    //            float middle = -log(u_middle);
    //            float right = -log(u_right);
    //            /* Use to sample. */
    //            ds = t_hit * (right - left);
    //            samplePoint = originPoint + sampleDirection * middle * t_hit;
    //        }
    //    } else {
    //        float sampleT = (((float) i) + 0.5) / ((float) _numberOfSamples);
    //        ds = t_hit / (float) (_numberOfSamples);
    //        samplePoint = originPoint + sampleDirection * sampleT * t_hit;
    //    }
    //
    //    /* Compute the densities of each atmosphere component scaled
    //    * by the sample interval ds. */
    //    float scaledDensityAir = computeDensityExponential(samplePoint,
    //      _planetRadius, _scaleHeightAir, _airDensity) * ds;
    //    float scaledDensityOzone = computeDensityTent(samplePoint,
    //      _planetRadius, _ozoneHeight, _ozoneThickness, _ozoneDensity) * ds;
    //    float scaledDensityAerosol = computeDensityExponential(samplePoint,
    //      _planetRadius, _scaleHeightAerosols, _aerosolDensity) * ds;
    //
    //    /* Accumulate the optical depth estimate that will ultimately be
    //    * used to compute the transmission factor that we'll use to model
    //    * extinction. */
    //    opticalDepthAir += scaledDensityAir;
    //    opticalDepthOzone += scaledDensityOzone;
    //    opticalDepthAerosol += scaledDensityAerosol;
    //
    //    /* Now we have to accumulate in-scattering contributions. */
    //
    //    /* First, trace a ray from the sample point to the light. TODO:
    //     * factor in visibility function here when terrain is incorporated. */
    //    float3 t_light = intersectSphere(samplePoint, L, atmosphereRadius);
    //    if (t_light.z >= 0) {
    //      float3 lightHit = samplePoint + L * max(t_light.x, t_light.y);
    //
    //      /* Compute the optical depth from the sample point to the light hit. */
    //      float opticalDepthLightAir = computeOpticalDepthExponential(samplePoint,
    //        lightHit, _planetRadius, _scaleHeightAir, _airDensity);
    //      float opticalDepthLightOzone = computeOpticalDepthTent(samplePoint,
    //        lightHit, _planetRadius, _ozoneHeight, _ozoneThickness, _ozoneDensity);
    //      float opticalDepthLightAerosol =
    //        computeOpticalDepthExponential(samplePoint, lightHit, _planetRadius,
    //        _scaleHeightAerosols, _aerosolDensity);
    //
    //      /* Compute the transmittance from the light to the sample point and
    //       * through to the origin point. */
    //      float3 T = exp(-_airCoefficientsF3 * (opticalDepthLightAir + opticalDepthAir)
    //        - _ozoneCoefficientsF3 * (opticalDepthLightOzone + opticalDepthOzone)
    //        - 1.1 * _aerosolCoefficient * (opticalDepthLightAerosol + opticalDepthAerosol));
    //
    //      /* Accumulate in scattering contributions using the scaled density
    //       * values for air and ozone at the sample point's height, which, lucky
    //       * for us, we've already computed. */
    //       inScatteringAir += scaledDensityAir * T;
    //       inScatteringAerosol += scaledDensityAerosol * T;
    //    }
    //
    //    /* Now, we factor in the night sky. For this, we ignore scattering
    //     * due to aerosols, since we are just using an approximation driven
    //     * by an average sky color. */
    //    int nightRayleighSamples = 5;
    //    for (int j = 0; j < nightRayleighSamples; j++) {
    //      /* TODO: can easily precompute these. */
    //      float lat = asin(-0.0 + 1.0 * float(j) / (nightRayleighSamples+1.0));
    //      float lon = GOLDEN_ANGLE * j;
    //      float3 nightDir = float3(cos(lon)*cos(lat), sin(lat), sin(lon)*cos(lat));
    //
    //      float mu_night = saturate(dot(sampleDirection, nightDir));
    //
    //      float rayleighPhaseNight = 3.f / (16.f * PI) * (1 + mu_night * mu_night);
    //
    //      float3 t_night = intersectSphere(samplePoint, nightDir, atmosphereRadius);
    //
    //      if (t_night.z >= 0) {
    //        float3 nightHit = samplePoint + nightDir * max(t_night.x, t_night.y);
    //        /* Compute the optical depth from the sample point to the light hit. */
    //        float opticalDepthNightAir = computeOpticalDepthExponential(samplePoint,
    //          nightHit, _planetRadius, _scaleHeightAir, _airDensity);
    //        float opticalDepthNightOzone = computeOpticalDepthTent(samplePoint,
    //          nightHit, _planetRadius, _ozoneHeight, _ozoneThickness, _ozoneDensity);
    //        float3 T_night = exp(-_airCoefficientsF3 * (opticalDepthNightAir + opticalDepthAir)
    //          - _ozoneCoefficientsF3 * (opticalDepthNightOzone + opticalDepthOzone));
    //        inScatteringAirNight += scaledDensityAir * rayleighPhaseNight * T_night / (float (nightRayleighSamples));
    //      }
    //    }
    //
    //
    //    if (!groundHit) {
    //      int nightMieSamples = 40;
    //      float opticalDepthNightAerosol =
    //        computeOpticalDepthExponential(samplePoint, hitPoint, _planetRadius,
    //          _scaleHeightAerosols, _aerosolDensity);
    //      float3 T_night = exp(-_aerosolCoefficient * opticalDepthNightAerosol);
    //      float3 r = normalize(cross(sampleDirection, float3(0, 0, 1)));
    //      float3 u = normalize(cross(r, sampleDirection));
    //      for (int j = 0; j < nightMieSamples; j++) {
    //        float rot = (((float)j) / ((float) nightMieSamples)) * (2.f * PI);
    //        float sampleRadius = 0.004 * random(float2(j, j+1));
    //        float3 nightMieDir = normalize(sampleDirection + sampleRadius * r * cos(rot) + sampleRadius * u * sin(rot));
    //
    //        float mu_nightMie = saturate(dot(sampleDirection, nightMieDir));
    //
    //        float miePhaseNight = 3.f / (8.0 * PI) * ((1.f - g * g) * (1.f + mu_nightMie * mu_nightMie))
    //           / ((2.f + g * g) * pow(1.f + g * g - 2.f * g * mu_nightMie, 1.5f));
    //
    //        /* Compute the optical depth from the sample point to the light hit. */
    //          inScatteringAerosolNight +=
    //          SAMPLE_TEXTURECUBE_LOD(_nightSkyHDRI, sampler_Cubemap, nightMieDir, 0)
    //             * scaledDensityAerosol * miePhaseNight * T_night / (float (nightMieSamples));
    //        }
    //      }
    //
    //  }
    //
    //  /* Now, we compute the phase functions for air and aerosol scattering:
    //   * correspondingly, rayleigh and mie phase functions. */
    //  float mu = saturate(dot(sampleDirection, L));
    //  float rayleighPhase = 3.f / (16.f * PI) * (1 + mu * mu);
    //  float miePhase = 3.f / (8.0 * PI) * ((1.f - g * g) * (1.f + mu * mu))
    //      / ((2.f + g * g) * pow(1.f + g * g - 2.f * g * mu, 1.5f));
    //
    //  /* Finally, we can compute the sky color using the in scattering
    //   * contribution we computed, the air and aerosol scattering coefficients,
    //   * the phase functions, the sun luminance, and the sky tint artistic
    //   * override. TODO: should we use the light color here or something? */
    //  float3 skyColor = sunColor
    //    * (2.0 * _skyTintF3 * _airCoefficientsF3 * inScatteringAir * rayleighPhase
    //    + _aerosolCoefficient * inScatteringAerosol * miePhase);
    //
    //  float3 nightColor = _nightIntensity * _nightTintF3
    //    * (_airCoefficientsF3 * inScatteringAirNight);
    //  float3 nightStarScatter = _nightIntensity * _nightTintF3
    //   * _aerosolCoefficient * inScatteringAerosolNight * _starAerosolScatterMultiplier;
    //
    //  /* Compute the final transmittance to apply to the direct light
    //   * sources. */
    //  float3 T = exp(-_airCoefficientsF3 * opticalDepthAir
    //    - _ozoneCoefficientsF3 * opticalDepthOzone
    //    - 1.1 * _aerosolCoefficient * opticalDepthAerosol);
    //
    //  float3 toRet = L0 * T + skyColor + nightColor + nightStarScatter;
    //
    //  /* TODO: make dithering draw from a noise texture for efficiency. */
    //  float dither = 1.0 + _ditherAmount * (0.5 - random(sampleDirection.xy));
    //  return (toRet * dither) * exposure;
  }

  float4 FragBaking(Varyings input) : SV_Target
  {
    return float4(RenderSky(input, 1.0), 1.0);
  }

  float4 FragRender(Varyings input) : SV_Target
  {
    //UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
    return float4(RenderSky(input, GetCurrentExposureMultiplier()), 1.0);
  }

  ENDHLSL

  SubShader
  {
    // Regular New Sky
    // For cubemap
    Pass
    {
      ZWrite Off
      ZTest Always
      Blend Off
      Cull Off

      HLSLPROGRAM
      #pragma fragment FragBaking
      ENDHLSL
    }

    // For fullscreen Sky
    Pass
    {
      ZWrite Off
      ZTest LEqual
      Blend Off
      Cull Off

      HLSLPROGRAM
      #pragma fragment FragRender
      ENDHLSL
    }
  }
  Fallback Off
}
