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
  #include "Packages/com.unity.render-pipelines.high-definition/Runtime/Sky/SkyUtils.hlsl"

  /* Common functions and global variables. */
  #include "Assets/CustomSky/ExpanseSky/code/ExpanseSkyCommon.hlsl"

/********************************************************************************/
/****************************** UNIFORM VARIABLES *******************************/
/********************************************************************************/

  TEXTURECUBE(_groundColorTexture);
  float4 _groundTint;
  TEXTURECUBE(_groundEmissiveTexture);
  float _groundEmissiveMultiplier;
  TEXTURECUBE(_nightSkyHDRI);
  float4 _nightTint;
  float _nightIntensity;
  float4 _skyTint;
  float _starAerosolScatterMultiplier;
  float _limbDarkening;  /* TODO: make this per celestial body. */
  float _ditherAmount;

  float3   _WorldSpaceCameraPos1;
  float4x4 _ViewMatrix1;
  #undef UNITY_MATRIX_V
  #define UNITY_MATRIX_V _ViewMatrix1


  /* Redefine colors to float3's for efficiency, since Unity can only set
   * float4's. */
  #define _groundTintF3 _groundTint.xyz
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

  float limbDarkening(float LdotV, float cosInner, float amount) {
    /* amount = max(FLT_EPS, amount); */
    float centerToEdge = 1.0 - abs((LdotV - cosInner) / (1.0 - cosInner));
    float mu = sqrt(1.0 - centerToEdge * centerToEdge);
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

  float3 RenderSky(Varyings i, float exposure, float3 jitter)
  {
    /* Get the origin point and sample direction. */
    float3 O = _WorldSpaceCameraPos1 - float3(0, -_planetRadius, 0);
    float3 d = normalize(-GetSkyViewDirWS(i.positionCS.xy) + jitter);

    /* See if we're looking at the ground or the sky. */
    float3 t_ground = intersectSphere(O, d, _planetRadius);
    float3 t_atmo = intersectSphere(O, d, _atmosphereRadius);
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
      for (int i = 0; i < min(4, _DirectionalLightCount); i++) {
        DirectionalLightData light = _DirectionalLightDatas[i];
        float3 L = -normalize(light.forward.xyz);
        float3 lightColor = light.color;
        float cos_hit_l = dot(normalize(hitPoint), L);
        float mapped_cos_hit_l = (cos_hit_l + 1.0) * 0.5;
        float2 groundIrradianceUV = float2(mapped_cos_hit_l, 0.0);
        /* Direct lighting. */
        L0 += _groundTintF3 * lightColor * (1.0 / PI) * saturate(cos_hit_l);
        /* Ground irradiance lighting. */
        float3 groundIrradianceAir =
          SAMPLE_TEXTURE2D(_GroundIrradianceTableAir,
          s_linear_clamp_sampler, groundIrradianceUV);
        float3 groundIrradianceAerosol =
          SAMPLE_TEXTURE2D(_GroundIrradianceTableAerosol,
          s_linear_clamp_sampler, groundIrradianceUV);
        L0 += _groundTintF3 * lightColor
          * (_skyTintF3 * 2.0 * groundIrradianceAir
            + groundIrradianceAerosol);
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
          L0 += luminance * limbDarkening(LdotV, cosInner, _limbDarkening)
            * light.surfaceTint;
        }
      }
    }

    /* Compute r and mu for the lookup tables. */
    float r = length(O);
    float mu = clampCosine(dot(normalize(O), d));

    /* Perform the transmittance table lookup attenuating direct lighting. */
    float2 transmittanceUV = mapTransmittanceCoordinates(r,
      mu, _atmosphereRadius, _planetRadius, t_hit, groundHit);
    float3 T = SAMPLE_TEXTURE2D(_TransmittanceTable, s_linear_clamp_sampler,
      transmittanceUV);

    /* Perform single scattering table lookup for the sky color.
     * HACK: clamping directional light count because of weird bug
     * where it's >100 for a sec. */
    float3 skyColor = float3(0, 0, 0);
    int lightCount = _DirectionalLightCount;
    for (int k = 0; k < min(4, _DirectionalLightCount); k++) {
      DirectionalLightData light = _DirectionalLightDatas[k];
      float3 L = -normalize(light.forward.xyz);
      float3 lightColor = light.color;

      /* Mu is the zenith angle of the light. */
      float mu_l = clampCosine(dot(normalize(O), L));

      /* Nu is the azimuth angle of the light, relative to the projection of
       * d onto the plane tangent to the surface of the planet at point O. */
      /* Project both L and d onto that plane by removing their "O"
       * component. */
      float3 proj_L = normalize(L - normalize(O) * mu_l);
      float3 proj_d = normalize(d - normalize(O) * dot(normalize(O), d));
      /* Take their dot product to get the cosine of the angle between them. */
      float nu  = clampCosine(dot(proj_L, proj_d));

      TexCoord5D ssCoord = mapSingleScatteringCoordinates(r, mu, mu_l, nu,
        _atmosphereRadius, _planetRadius, t_hit, groundHit);

      float3 uvw0 = float3(ssCoord.x, ssCoord.y, ssCoord.z);
      float3 uvw1 = float3(ssCoord.x, ssCoord.y, ssCoord.w);

      float3 ssContrib0Air = SAMPLE_TEXTURE3D(_SingleScatteringTableAir,
        s_trilinear_clamp_sampler, uvw0).rgb;
      float3 ssContrib1Air = SAMPLE_TEXTURE3D(_SingleScatteringTableAir,
        s_trilinear_clamp_sampler, uvw1).rgb;

      float3 singleScatteringContributionAir = lerp(ssContrib0Air, ssContrib1Air, ssCoord.a);

      float3 ssContrib0Aerosol = SAMPLE_TEXTURE3D(_SingleScatteringTableAerosol,
        s_trilinear_clamp_sampler, uvw0).rgb;
      float3 ssContrib1Aerosol = SAMPLE_TEXTURE3D(_SingleScatteringTableAerosol,
        s_trilinear_clamp_sampler, uvw1).rgb;

      float3 singleScatteringContributionAerosol = lerp(ssContrib0Aerosol, ssContrib1Aerosol, ssCoord.a);

      float dot_L_d = dot(L, d);
      float rayleighPhase = computeAirPhase(dot_L_d);
      float miePhase = computeAerosolPhase(dot_L_d, g);

      float3 finalSingleScattering = (2.0 * _skyTintF3 * _airCoefficientsF3
        * singleScatteringContributionAir * rayleighPhase
        + _aerosolCoefficient * singleScatteringContributionAerosol * miePhase);

      skyColor += finalSingleScattering * lightColor;
    }

    float3 finalDirectLighting = (L0 * T);

    float dither = 1.0 + _ditherAmount * (0.5 - random(d.xy));
    return dither * exposure * (finalDirectLighting
      + skyColor
      + (groundHit ? 0.0 : _nightTint * _nightIntensity));
  }

  float4 FragBaking(Varyings input) : SV_Target
  {
    return float4(RenderSky(input, 1.0, float3(0, 0, 0)), 1.0);
  }

  float4 FragRender(Varyings input) : SV_Target
  {
    //UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
    float exposure = GetCurrentExposureMultiplier();
    return float4(RenderSky(input, exposure, float3(0, 0, 0)), 1.0);
    /* HACK: Hacky 8x MSAA. */
    /* return float4((RenderSky(input, exposure, float3(0, 0, 0))
      + RenderSky(input, exposure, float3(0.0005, 0.0005, 0.0005))
      + RenderSky(input, exposure, float3(-0.0005, 0.0005, -0.0005))
      + RenderSky(input, exposure, float3(0.0005, -0.0005, -0.0005))
      + RenderSky(input, exposure, float3(-0.0003, -0.0001, 0.0005))
      + RenderSky(input, exposure, float3(0.0001, 0.0002, 0.0005))
      + RenderSky(input, exposure, float3(0.0005, -0.0005, 0.0005))
      + RenderSky(input, exposure, float3(-0.0003, -0.0005, 0.0002))) / 8.0, 1.0); */
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
