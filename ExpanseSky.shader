/* Much of this is adapted from https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/s2016-pbs-frostbite-sky-clouds-new.pdf. */
Shader "HDRP/Sky/ExpanseSky"
{
  HLSLINCLUDE

  #pragma vertex Vert

  #pragma editor_sync_compilation
  #pragma target 4.5
  #pragma only_renderers d3d11 ps4 xboxone vulkan metal switch


  /* Unity. */
  #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
  #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/CommonLighting.hlsl"
  #include "Packages/com.unity.render-pipelines.high-definition/Runtime/Lighting/LightDefinition.cs.hlsl"
  #include "Packages/com.unity.render-pipelines.high-definition/Runtime/Sky/SkyUtils.hlsl"
  #include "Packages/com.unity.render-pipelines.high-definition/Runtime/Lighting/LightLoop/CookieSampling.hlsl"

  /* Common functions and global variables. */
  #include "Assets/CustomSky/ExpanseSky/code/ExpanseSkyCommon.hlsl"

/* TODO: had to remove a check that physically based sky was active in
 * the main light loop C# script to get additional light data like light
 * distance. Is there a way to avoid hacking the source? EDIT: also looks
 * like it resets it every time you open the editor. */

/********************************************************************************/
/****************************** UNIFORM VARIABLES *******************************/
/********************************************************************************/

  TEXTURECUBE(_nightSkyTexture);
  bool _hasNightSkyTexture;
  float4 _nightTint;
  float4x4 _nightSkyRotation;
  float _nightIntensity;
  float4 _skyTint;
  float _starAerosolScatterMultiplier;
  float _multipleScatteringMultiplier;
  bool _useAntiAliasing;
  float _ditherAmount;

  /* Celestial bodies. */
  /* Body 1. */
  float _body1LimbDarkening;
  bool _body1ReceivesLight;
  TEXTURECUBE(_body1AlbedoTexture);
  bool _body1HasAlbedoTexture;
  bool _body1Emissive;
  TEXTURECUBE(_body1EmissionTexture);
  bool _body1HasEmissionTexture;
  float4x4 _body1Rotation;
  /* Body 2. */
  float _body2LimbDarkening;
  bool _body2ReceivesLight;
  TEXTURECUBE(_body2AlbedoTexture);
  bool _body2HasAlbedoTexture;
  bool _body2Emissive;
  TEXTURECUBE(_body2EmissionTexture);
  bool _body2HasEmissionTexture;
  float4x4 _body2Rotation;
  /* Body 3. */
  float _body3LimbDarkening;
  bool _body3ReceivesLight;
  TEXTURECUBE(_body3AlbedoTexture);
  bool _body3HasAlbedoTexture;
  bool _body3Emissive;
  TEXTURECUBE(_body3EmissionTexture);
  bool _body3HasEmissionTexture;
  float4x4 _body3Rotation;
  /* Body 4. */
  float _body4LimbDarkening;
  bool _body4ReceivesLight;
  TEXTURECUBE(_body4AlbedoTexture);
  bool _body4HasAlbedoTexture;
  bool _body4Emissive;
  TEXTURECUBE(_body4EmissionTexture);
  bool _body4HasEmissionTexture;
  float4x4 _body4Rotation;

  /* HACK: We only allow 4 celestial bodies now. */
  #define MAX_DIRECTIONAL_LIGHTS 4

  float3   _WorldSpaceCameraPos1;
  float4x4 _ViewMatrix1;
  #undef UNITY_MATRIX_V
  #define UNITY_MATRIX_V _ViewMatrix1


  /* Redefine colors to float3's for efficiency, since Unity can only set
   * float4's. */
  #define _nightTintF3 _nightTint.xyz
  #define _skyTintF3 _skyTint.xyz

/********************************************************************************/
/**************************** END UNIFORM VARIABLES *****************************/
/********************************************************************************/

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

  float3 limbDarkening(float LdotV, float cosInner, float amount) {
    /* amount = max(FLT_EPS, amount); */
    float centerToEdge = 1.0 - abs((LdotV - cosInner) / (1.0 - cosInner));
    float mu = safeSqrt(1.0 - centerToEdge * centerToEdge);
    float mu2 = mu * mu;
    float mu3 = mu2 * mu;
    float mu4 = mu2 * mu2;
    float mu5 = mu3 * mu2;
    float3 a0 = float3 (0.34685, 0.26073, 0.15248);
    float3 a1 = float3 (1.37539, 1.27428, 1.38517);
    float3 a2 = float3 (-2.04425, -1.30352, -1.49615);
    float3 a3 = float3 (2.70493, 1.47085, 1.99886);
    float3 a4 = float3 (-1.94290, -0.96618, -1.48155);
    float3 a5 = float3 (0.55999, 0.26384, 0.44119);
    return max(0.0, pow(a0 + a1 * mu + a2 * mu2 + a3 * mu3 + a4 * mu4 + a5 * mu5, amount));
  }

  /* Accumulate direct lighting on the ground. */
  float3 shadeGround(float3 groundPoint) {
    int i = 0; /* Number of lights we've seen. */
    int k = 0; /* Number of lights that apply to the sky. */
    float3 result = float3(0, 0, 0);
    while (i < min(_DirectionalLightCount, 2 * MAX_DIRECTIONAL_LIGHTS)
      && k < MAX_DIRECTIONAL_LIGHTS) {
      DirectionalLightData light = _DirectionalLightDatas[i];
      /* This lets us know if the light affects the physical sky. */
      if (asint(light.distanceFromCamera) >= 0) {
        /* Get the light direction and color. */
        float3 L = -normalize(light.forward.xyz);
        float3 lightColor = light.color;

        /* Get the ground emission and add it to the direct light.  */
        float3 groundEmission = float3(0, 0, 0);
        if (_hasGroundEmissionTexture) {
          groundEmission = _groundEmissionMultiplier
            * SAMPLE_TEXTURECUBE_LOD(_groundEmissionTexture,
              sampler_Cubemap, mul(normalize(groundPoint), (float3x3)_planetRotation), 0).rgb;
        }
        result += groundEmission;

        /* Get the ground albedo. */
        float3 groundAlbedo = _groundTintF3;
        if (_hasGroundAlbedoTexture) {
          groundAlbedo *= 2.0 * SAMPLE_TEXTURECUBE_LOD(_groundAlbedoTexture,
            sampler_Cubemap, mul(normalize(groundPoint), (float3x3)_planetRotation), 0).rgb;
        }
        groundAlbedo /= PI;

        /* Compute the ground reflectance from the light. */
        float cos_hit_l = dot(normalize(groundPoint), L);
        result += groundAlbedo * lightColor * saturate(cos_hit_l);

        /* Compute the ground reflectance from the sky. */
        float2 groundIrradianceUV = mapGroundIrradianceCoordinates(cos_hit_l);
        float3 groundIrradianceAir =
          SAMPLE_TEXTURE2D(_GroundIrradianceTableAir,
          s_linear_clamp_sampler, groundIrradianceUV).rgb;
        float3 groundIrradianceAerosol =
          SAMPLE_TEXTURE2D(_GroundIrradianceTableAerosol,
          s_linear_clamp_sampler, groundIrradianceUV).rgb;
        result += groundAlbedo * lightColor
          * (_skyTintF3 * 2.0 * _airCoefficientsF3 * groundIrradianceAir
            + _aerosolCoefficient * groundIrradianceAerosol);
        k++;
      }
      i++;
    }
    return result;
  }

  /* Shade the celestial body that is closest. Return a negative vector if we
   * didn't hit anything. TODO: this doesn't work for eclipses. */
  float3 shadeClosestCelestialBody(float3 d) {
    /* Define arrays of celestial body variables to make things easier to
     * write. Syntactic sugar, if you will. */
    float celestialBodyLimbDarkening[MAX_DIRECTIONAL_LIGHTS] =
      {_body1LimbDarkening, _body2LimbDarkening, _body3LimbDarkening, _body4LimbDarkening};
    bool celestialBodyReceivesLight[MAX_DIRECTIONAL_LIGHTS] =
      {_body1ReceivesLight, _body2ReceivesLight, _body3ReceivesLight, _body4ReceivesLight};
    TextureCube celestialBodyAlbedoTexture[MAX_DIRECTIONAL_LIGHTS] =
      {_body1AlbedoTexture, _body2AlbedoTexture, _body3AlbedoTexture, _body4AlbedoTexture};
    bool celestialBodyHasAlbedoTexture[MAX_DIRECTIONAL_LIGHTS] =
      {_body1HasAlbedoTexture, _body2HasAlbedoTexture, _body3HasAlbedoTexture, _body4HasAlbedoTexture};
    bool celestialBodyEmissive[MAX_DIRECTIONAL_LIGHTS] =
      {_body1Emissive, _body2Emissive, _body3Emissive, _body4Emissive};
    TextureCube celestialBodyEmissionTexture[MAX_DIRECTIONAL_LIGHTS] =
      {_body1EmissionTexture, _body2EmissionTexture, _body3EmissionTexture, _body4EmissionTexture};
    bool celestialBodyHasEmissionTexture[MAX_DIRECTIONAL_LIGHTS] =
      {_body1HasEmissionTexture, _body2HasEmissionTexture, _body3HasEmissionTexture, _body4HasEmissionTexture};
    float4x4 celestialBodyRotations[MAX_DIRECTIONAL_LIGHTS] =
      {_body1Rotation, _body2Rotation, _body3Rotation, _body4Rotation};

    /* Keep track of the closest body we've seen. */
    float minDist = -1.0;
    int i = 0; /* Number of lights we've seen. */
    int k = 0; /* Number of lights that apply to the sky. */
    float3 celestialBodyLight = float3(0.0, 0.0, 0.0);
    while (i < min(_DirectionalLightCount, 2 * MAX_DIRECTIONAL_LIGHTS)
      && k < MAX_DIRECTIONAL_LIGHTS) {
      DirectionalLightData light = _DirectionalLightDatas[i];
      /* This lets us know if the light affects the physical sky. */
      if (asint(light.distanceFromCamera) >= 0) {
        float3 L = -normalize(light.forward.xyz);
        float LdotV    = dot(L, d);
        float radInner = 0.5 * light.angularDiameter;
        float cosInner = cos(radInner);
        float cosOuter = cos(radInner + light.flareSize); /* TODO: use for flares in the future. */
        if (LdotV > cosInner + FLT_EPSILON && (minDist < 0 || light.distanceFromCamera < minDist)) {
          /* We can see the celestial body and it's closer than the others
           * we've seen so far. Reset minDist and the celestial body
           * light accumulator. */
          minDist = light.distanceFromCamera;
          celestialBodyLight = float3(0.0, 0.0, 0.0);

          /* We take the approach of allowing a celestial body both to be
           * emissive and to receive light. This is useful for portraying
           * something like city lights on a moon. */

          /* Add the emissive contribution if the celestial body is
           * emissive. */
          if (celestialBodyEmissive[i]) {
            float3 emission = computeCelestialBodyLuminance(light.color, cosInner);
            /* Apply limb darkening. */
            emission *= limbDarkening(LdotV, cosInner, celestialBodyLimbDarkening[i]);
            /* Apply surface tint. */
            emission *= light.surfaceTint.rgb;
            /* Apply emission texture if we have one. */
            if (celestialBodyHasEmissionTexture[i]) {
              /* We have to do some work to compute the intersection. */
              float bodyDist = light.distanceFromCamera;
              float bodyRadius = safeSqrt(1.0 - cosInner * cosInner) * bodyDist;
              float3 planetOriginInBodyFrame = -(L * bodyDist);
              /* Intersect the body at the point we're looking at. */
              float3 bodyIntersection =
              intersectSphere(planetOriginInBodyFrame, d, bodyRadius);
              float3 bodyIntersectionPoint = planetOriginInBodyFrame
                + minNonNegative(bodyIntersection.x, bodyIntersection.y) * d;
              float3 surfaceNormal = normalize(bodyIntersectionPoint);

              emission *=
                SAMPLE_TEXTURECUBE_LOD(celestialBodyEmissionTexture[i],
                sampler_Cubemap, mul(surfaceNormal, (float3x3) celestialBodyRotations[i]), 0).rgb;
            }
            celestialBodyLight += emission;
          }

          /* Add the diffuse contribution if the celestial body receives
           * light. */
          if (celestialBodyReceivesLight[i]) {
            /* We have to do some work to compute the surface normal. */
            float bodyDist = light.distanceFromCamera;
            float bodyRadius = safeSqrt(1.0 - cosInner * cosInner) * bodyDist;
            float3 planetOriginInBodyFrame = -(L * bodyDist);
            float3 bodyIntersection = intersectSphere(planetOriginInBodyFrame, d, bodyRadius);
            float3 bodyIntersectionPoint = planetOriginInBodyFrame + minNonNegative(bodyIntersection.x, bodyIntersection.y) * d;
            float3 bodySurfaceNormal = normalize(bodyIntersectionPoint);

            float3 bodyAlbedo = 2.0 * light.surfaceTint.rgb;
            if (celestialBodyHasAlbedoTexture[i]) {
              bodyAlbedo *= SAMPLE_TEXTURECUBE_LOD(celestialBodyAlbedoTexture[i],
                sampler_Cubemap, mul(bodySurfaceNormal, (float3x3) celestialBodyRotations[i]), 0).rgb;
            }
            bodyAlbedo *= 1.0/PI;

            int j = 0;
            int jk = 0;
            while (j < min(_DirectionalLightCount, 2 * MAX_DIRECTIONAL_LIGHTS)
              && jk < MAX_DIRECTIONAL_LIGHTS) {
              DirectionalLightData emissiveLight = _DirectionalLightDatas[j];
              /* Body can't light itself. */
              if (j != i && asint(emissiveLight.distanceFromCamera) >= 0) {
                /* Since both bodies may be pretty far away, we can't just
                 * use the emissive body's direction. We have to take
                 * the difference in body positions. */
                float3 emissivePosition = emissiveLight.distanceFromCamera
                  * -normalize(emissiveLight.forward.xyz);
                float3 bodyPosition = L * bodyDist;
                float3 emissiveDir = normalize(emissivePosition - bodyPosition);
                celestialBodyLight += saturate(dot(emissiveDir, bodySurfaceNormal))
                  * emissiveLight.color * bodyAlbedo;
                jk++;
              }
              j++;
            }
          }
        }
        k++;
      }
      i++;
    }
    /* Return a negative vector if we didn't hit anything. */
    if (minDist < 0) {
      celestialBodyLight = float3(-1.0, -1.0, -1.0);
    }
    return celestialBodyLight;
  }

  /* Accumulate sky color. */
  float3 shadeSky(float r, float mu, float3 startPoint, float t_hit,
    float3 d, bool groundHit) {

    /* Precompute some things outside the loop for efficiency. */
    float3 normalizedStartPoint = normalize(startPoint);

    int i = 0; /* Number of lights we've seen. */
    int k = 0; /* Number of lights that apply to the sky. */

    float3 skyColor = float3(0, 0, 0);

    while (i < min(_DirectionalLightCount, 2 * MAX_DIRECTIONAL_LIGHTS)
      && k < MAX_DIRECTIONAL_LIGHTS) {

      DirectionalLightData light = _DirectionalLightDatas[i];
      /* This lets us know if the light affects the physical sky. */
      if (asint(light.distanceFromCamera) >= 0) {
        float3 L = -normalize(light.forward.xyz);
        float3 lightColor = light.color;

        /* Mu is the zenith angle of the light. */
        float mu_l = clampCosine(dot(normalize(startPoint), L));

        /* Nu is the azimuth angle of the light, relative to the projection of
         * d onto the plane tangent to the surface of the planet at point O. */
        /* Project both L and d onto that plane by removing their "O"
         * component. */
        float3 proj_L = normalize(L - normalizedStartPoint * mu_l);
        float3 proj_d = normalize(d - normalizedStartPoint * dot(normalizedStartPoint, d));
        /* Take their dot product to get the cosine of the angle between them. */
        float nu = clampCosine(dot(proj_L, proj_d));

        TexCoord4D ssCoord = mapSingleScatteringCoordinates(r, mu, mu_l, nu,
          _atmosphereRadius, _planetRadius, t_hit, groundHit);

        float3 singleScatteringContributionAir =
          sampleTexture4D(_SingleScatteringTableAir, ssCoord);

        float3 singleScatteringContributionAerosol =
          sampleTexture4D(_SingleScatteringTableAerosol, ssCoord);

        float dot_L_d = dot(L, d);
        float rayleighPhase = computeAirPhase(dot_L_d);
        float miePhase = computeAerosolPhase(dot_L_d, g);

        float3 finalSingleScattering = (2.0 * _skyTintF3 * _airCoefficientsF3
          * singleScatteringContributionAir * rayleighPhase
          + _aerosolCoefficient * singleScatteringContributionAerosol * miePhase);

        /* Sample multiple scattering. */
        TexCoord4D msCoord = mapGlobalMultipleScatteringCoordinates(r, mu,
          mu_l, nu, _atmosphereRadius, _planetRadius, t_hit,
          groundHit);

        float3 msAir =
          sampleTexture4D(_GlobalMultipleScatteringTableAir, msCoord);

        float3 msAerosol =
          sampleTexture4D(_GlobalMultipleScatteringTableAerosol, msCoord);

        float3 finalMultipleScattering = (2.0 * _skyTintF3
          * _airCoefficientsF3 * msAir
          + _aerosolCoefficient * msAerosol)
          * _multipleScatteringMultiplier;

        skyColor +=
          (finalSingleScattering + finalMultipleScattering) * lightColor;

        k++;

      }

      i++;

    }
    return skyColor;
  }

  /* Accumulate light pollution. */
  float3 shadeLightPollution(float2 uv) {
    float3 lightPollutionAir = SAMPLE_TEXTURE2D(_LightPollutionTableAir,
      s_linear_clamp_sampler, uv);
    float3 lightPollutionAerosol = SAMPLE_TEXTURE2D(_LightPollutionTableAerosol,
      s_linear_clamp_sampler, uv);
    /* Consider light pollution to be isotropically scattered, so don't
     * apply phase functions. */
    return (2.0 * _skyTintF3 * _airCoefficientsF3
      * lightPollutionAir + _aerosolCoefficient * lightPollutionAerosol)
      * _lightPollutionTint * _lightPollutionIntensity;
  }

  /* Accumulate scattering due to the night sky. */
  float3 shadeNightScattering(float r, float mu, float t_hit, bool groundHit) {
    /* HACK: to get some sort of approximation of rayleigh scattering
     * for the ambient night color of the sky. */
    TexCoord4D ssCoord_night = mapSingleScatteringCoordinates(r, mu, mu, 1.0,
      _atmosphereRadius, _planetRadius, t_hit, groundHit);
     float3 singleScatteringContributionAirNight =
       sampleTexture4D(_SingleScatteringTableAir, ssCoord_night);
     float rayleighPhase_night = computeAirPhase(1.0);

     float3 finalSingleScatteringNight = (2.0 * _skyTintF3 * _airCoefficientsF3
       * singleScatteringContributionAirNight * rayleighPhase_night);

     /* Sample multiple scattering. */
     TexCoord4D msCoord_night = mapGlobalMultipleScatteringCoordinates(r, mu,
       mu, 1, _atmosphereRadius, _planetRadius, t_hit, groundHit);

     float3 msAirNight =
       sampleTexture4D(_GlobalMultipleScatteringTableAir, msCoord_night);

     float3 finalMultipleScatteringNight = (2.0 * _skyTintF3
       * _airCoefficientsF3 * msAirNight)
       * _multipleScatteringMultiplier;

    /* HACK: 1.0/8.0 here is a magic number representing something like the
     * star density. */
    return (finalSingleScatteringNight
      + finalMultipleScatteringNight) * _nightTintF3 * _nightIntensity/8.0;
  }

  float3 RenderSky(Varyings input, float exposure, float2 jitter)
  {

    /* Get the origin point and sample direction. */
    float3 O = _WorldSpaceCameraPos1 - float3(0, -_planetRadius, 0);
    float3 d = normalize(-GetSkyViewDirWS(input.positionCS.xy + jitter));

    /* Trace a ray to see what we hit. */
    IntersectionData intersection = traceRay(O, d, _planetRadius,
      _atmosphereRadius);

    /* You might think the start point is just O, but we may be looking
     * at the planet from space, in which case the start point is the point
     * that we hit the atmosphere. */
    float3 startPoint = O + d * intersection.startT;
    float3 endPoint = O + d * intersection.endT;
    float t_hit = intersection.endT - intersection.startT;

    /* Accumulate direct illumination. */
    float3 L0 = float3(0, 0, 0);
    if (intersection.groundHit) {
      L0 += shadeGround(endPoint);
    } else {
      float3 closestCelestialBodyShading = shadeClosestCelestialBody(d);
      if (closestCelestialBodyShading.x < 0) {
        /* We didn't hit anything. Add the stars. */
        float3 starTexture = SAMPLE_TEXTURECUBE_LOD(_nightSkyTexture,
          sampler_Cubemap, mul(d, (float3x3)_nightSkyRotation), 0).rgb;
        L0 += starTexture * _nightTintF3 * _nightIntensity;
      } else {
        /* We did hit a celestial body. Add its shading. */
        L0 += shadeClosestCelestialBody(d);
      }
    }

    /* Compute r and mu for the lookup tables. */
    float r = length(startPoint);
    float mu = clampCosine(dot(normalize(startPoint), d));

    /* Perform the transmittance table lookup to attenuate direct lighting. */
    float2 transmittanceUV = mapTransmittanceCoordinates(r,
      mu, _atmosphereRadius, _planetRadius, t_hit, intersection.groundHit);
    float3 T = SAMPLE_TEXTURE2D(_TransmittanceTable, s_linear_clamp_sampler,
      transmittanceUV);
    float3 finalDirectLighting = (L0 * T);

    /* If we hit the sky or the ground, compute the sky color, air scattering
     * due to the night sky, and the light pollution. */
    float3 skyColor = float3(0, 0, 0);
    float3 nightAirScattering = float3(0, 0, 0);
    float3 lightPollution = float3(0, 0, 0);
    if (intersection.groundHit || intersection.atmoHit) {
      skyColor = shadeSky(r, mu, startPoint, t_hit, d, intersection.groundHit);
      lightPollution = shadeLightPollution(transmittanceUV);
      nightAirScattering = shadeNightScattering(r, mu, t_hit,
        intersection.groundHit);
    }

    float dither = 1.0 + _ditherAmount * (0.5 - random(d.xy));
    return dither * exposure * (finalDirectLighting
      + skyColor + nightAirScattering + lightPollution);
  }

  float4 FragBaking(Varyings input) : SV_Target
  {
    return float4(RenderSky(input, 1.0, float3(0, 0, 0)), 1.0);
  }

  float4 FragRender(Varyings input) : SV_Target
  {
    float MSAA_8X_OFFSETS_X[8] = {1.0/16.0, -1.0/16.0, 5.0/16.0, -3.0/16.0, -5.0/16.0, -7.0/16.0, 3.0/16.0, 7.0/16.0};
    float MSAA_8X_OFFSETS_Y[8] =  {-3.0/16.0, 3.0/16.0, 1.0/16.0, -5.0/16.0, 5.0/16.0, -1.0/16.0, 7.0/16.0, -7.0/16.0};
    UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
    float exposure = GetCurrentExposureMultiplier();
    float4 skyResult = float4(0.0, 0.0, 0.0, 1.0);
    if (_useAntiAliasing) {
      for (int i = 0; i < 8; i++) {
        skyResult += float4(RenderSky(input, exposure,
          float2(MSAA_8X_OFFSETS_X[i], MSAA_8X_OFFSETS_Y[i])), 1.0);
      }
      skyResult /= 8.0;
      skyResult.w = 1.0;
    } else {
      skyResult = float4(RenderSky(input, exposure, float2(0.0, 0.0)), 1.0);
    }
    return skyResult;
  }

  ENDHLSL

  SubShader
  {
    /* For cubemap. */
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

    /* For fullscreen sky. */
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
