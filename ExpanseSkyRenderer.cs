using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

class ExpanseSkyRenderer : SkyRenderer
{

    /********************************************************************************/
    /**************************** Shader Variable ID's ******************************/
    /********************************************************************************/

    public static readonly int _atmosphereThicknessID = Shader.PropertyToID("_atmosphereThickness");
    public static readonly int _atmosphereRadiusID = Shader.PropertyToID("_atmosphereRadius");
    public static readonly int _planetRadiusID = Shader.PropertyToID("_planetRadius");
    public static readonly int _groundColorTextureID = Shader.PropertyToID("_groundColorTexture");
    public static readonly int _groundTintID = Shader.PropertyToID("_groundTint");
    public static readonly int _groundEmissiveTextureID = Shader.PropertyToID("_groundEmissiveTexture");
    public static readonly int _groundEmissiveMultiplierID = Shader.PropertyToID("_groundEmissiveMultiplier");
    public static readonly int _nightSkyHDRIID = Shader.PropertyToID("_nightSkyHDRI");
    public static readonly int _nightTintID = Shader.PropertyToID("_nightTint");
    public static readonly int _nightIntensityID = Shader.PropertyToID("_nightIntensity");
    public static readonly int _aerosolCoefficientID = Shader.PropertyToID("_aerosolCoefficient");
    public static readonly int _scaleHeightAerosolsID = Shader.PropertyToID("_scaleHeightAerosols");
    public static readonly int _aerosolAnisotropyID = Shader.PropertyToID("_aerosolAnisotropy");
    public static readonly int _aerosolDensityID = Shader.PropertyToID("_aerosolDensity");
    public static readonly int _airCoefficientsID = Shader.PropertyToID("_airCoefficients");
    public static readonly int _scaleHeightAirID = Shader.PropertyToID("_scaleHeightAir");
    public static readonly int _airDensityID = Shader.PropertyToID("_airDensity");
    public static readonly int _ozoneCoefficientsID = Shader.PropertyToID("_ozoneCoefficients");
    public static readonly int _ozoneThicknessID = Shader.PropertyToID("_ozoneThickness");
    public static readonly int _ozoneHeightID = Shader.PropertyToID("_ozoneHeight");
    public static readonly int _ozoneDensityID = Shader.PropertyToID("_ozoneDensity");
    public static readonly int _skyTintID = Shader.PropertyToID("_skyTint");
    public static readonly int _starAerosolScatterMultiplierID = Shader.PropertyToID("_starAerosolScatterMultiplier");
    public static readonly int _limbDarkeningID = Shader.PropertyToID("_limbDarkening");
    public static readonly int _numberOfSamplesID = Shader.PropertyToID("_numberOfSamples");
    public static readonly int _useImportanceSamplingID = Shader.PropertyToID("_useImportanceSampling");
    public static readonly int _useCubicApproximationID = Shader.PropertyToID("_useCubicApproximation");
    public static readonly int _ditherAmountID = Shader.PropertyToID("_ditherAmount");

    public static readonly int _WorldSpaceCameraPos1ID = Shader.PropertyToID("_WorldSpaceCameraPos1");
    public static readonly int _ViewMatrix1ID = Shader.PropertyToID("_ViewMatrix1");
    public static readonly int _PixelCoordToViewDirWS = Shader.PropertyToID("_PixelCoordToViewDirWS");

    /* Tables. */
    public static readonly int _transmittanceTableID = Shader.PropertyToID("_TransmittanceTable");
    public static readonly int _singleScatteringTableAirID = Shader.PropertyToID("_SingleScatteringTableAir");
    public static readonly int _singleScatteringTableAerosolID = Shader.PropertyToID("_SingleScatteringTableAerosol");

    /********************************************************************************/
    /************************** End Shader Variable ID's ****************************/
    /********************************************************************************/

    /* Dimensions of precomputed tables. */
    const int TransmittanceTableSizeH = 32;
    const int TransmittanceTableSizePhi = 128;

    const int SingleScatteringTableSizeH = 32;
    const int SingleScatteringTableSizePhi = 128;
    const int SingleScatteringTableSizePhiL = 32;
    const int SingleScatteringTableSizeNu = 64;

    /* Transmittance table. Leverages spherical symmetry of the atmosphere,
     * parameterized by:
     * h (x dimension): the height of the camera.
     * phi (y dimension): the zenith angle of the viewing direction. */
    RTHandle[]                   m_TransmittanceTables;

    /* Single scattering table. Leverages spherical symmetry of the atmosphere,
     * parameterized by:
     * h (x dimension): the height of the camera.
     * phi (y dimension): the zenith angle of the viewing direction.
     * phi_l (z dimension): the zenith angle of the light source.
     * nu (w dimension): the azimuth angle of the light source. */
    RTHandle[]                   m_SingleScatteringTables;

    static ComputeShader         s_PrecomputeCS;

    static GraphicsFormat s_ColorFormat = GraphicsFormat.R16G16B16A16_SFloat;

    /* Use the same strategy as Physically Based Sky to determine when we
     * need to update our tables---compute a hash of the relevant parameters
     * every frame and check for differences. */
    int m_LastPrecomputationParamHash;

    RTHandle AllocateTransmittanceTable(int index)
    {
        var table = RTHandles.Alloc(TransmittanceTableSizeH,
                                    TransmittanceTableSizePhi,
                                    dimension: TextureDimension.Tex2D,
                                    colorFormat: s_ColorFormat,
                                    enableRandomWrite: true,
                                    name: string.Format("TransmittanceTable{0}", index));

        Debug.Assert(table != null);

        return table;
    }

    RTHandle AllocateSingleScatteringTable(int index)
    {
        var table = RTHandles.Alloc(SingleScatteringTableSizeH,
                                    SingleScatteringTableSizePhi,
                                    SingleScatteringTableSizePhiL *
                                    SingleScatteringTableSizeNu,
                                    dimension: TextureDimension.Tex3D,
                                    colorFormat: s_ColorFormat,
                                    enableRandomWrite: true,
                                    name: string.Format("SingleScatteringTable{0}", index));

        Debug.Assert(table != null);

        return table;
    }

    // Renders a cubemap into a render texture (can be cube or 2D)
    Material m_ExpanseSkyMaterial;
    MaterialPropertyBlock m_PropertyBlock = new MaterialPropertyBlock();
    private static int m_RenderCubemapID = 0; // FragBaking
    private static int m_RenderFullscreenSkyID = 1; // FragRender

    public override void Build()
    {
      /* Shaders. */
      m_ExpanseSkyMaterial = CoreUtils.CreateEngineMaterial(GetExpanseSkyShader());
      s_PrecomputeCS = GetExpanseSkyPrecomputeShader();

      Debug.Assert(m_ExpanseSkyMaterial != null);
      Debug.Assert(s_PrecomputeCS != null);

      m_TransmittanceTables = new RTHandle[2];
      m_TransmittanceTables[0] = AllocateTransmittanceTable(0);
      m_TransmittanceTables[1] = AllocateTransmittanceTable(1);

      m_SingleScatteringTables = new RTHandle[2];
      m_SingleScatteringTables[0] = AllocateSingleScatteringTable(0);
      m_SingleScatteringTables[1] = AllocateSingleScatteringTable(1);
    }

    // Project dependent way to retrieve a shader.
    Shader GetExpanseSkyShader()
    {
        return Shader.Find("HDRP/Sky/ExpanseSky");
    }

    ComputeShader GetExpanseSkyPrecomputeShader()
    {
        return Resources.Load<ComputeShader>("ExpanseSkyPrecompute");
    }

    public override void Cleanup()
    {
        CoreUtils.Destroy(m_ExpanseSkyMaterial);
        RTHandles.Release(m_TransmittanceTables[0]);
        m_TransmittanceTables[0] = null;
        RTHandles.Release(m_TransmittanceTables[1]);
        m_TransmittanceTables[1] = null;

        RTHandles.Release(m_SingleScatteringTables[0]);
        m_SingleScatteringTables[0] = null;
        RTHandles.Release(m_SingleScatteringTables[1]);
        m_SingleScatteringTables[1] = null;
    }

    void SetGlobalConstants(CommandBuffer cmd, BuiltinSkyParameters builtinParams) {

        var expanseSky = builtinParams.skySettings as ExpanseSky;

        /* Set the parameters that need to be used for sky table
         * precomputation, aka, everything in ExpanseSkyCommon.hlsl. */
        cmd.SetGlobalFloat(_atmosphereThicknessID, expanseSky.atmosphereThickness.value);
        cmd.SetGlobalFloat(_atmosphereRadiusID, expanseSky.atmosphereThickness.value + expanseSky.planetRadius.value);
        cmd.SetGlobalFloat(_planetRadiusID, expanseSky.planetRadius.value);
        cmd.SetGlobalFloat(_aerosolCoefficientID, expanseSky.aerosolCoefficient.value);
        cmd.SetGlobalFloat(_scaleHeightAerosolsID, expanseSky.scaleHeightAerosols.value);
        cmd.SetGlobalFloat(_aerosolDensityID, expanseSky.aerosolDensity.value);
        cmd.SetGlobalVector(_airCoefficientsID, expanseSky.airCoefficients.value);
        cmd.SetGlobalFloat(_scaleHeightAirID, expanseSky.scaleHeightAir.value);
        cmd.SetGlobalFloat(_airDensityID, expanseSky.airDensity.value);
        cmd.SetGlobalVector(_ozoneCoefficientsID, expanseSky.ozoneCoefficients.value);
        cmd.SetGlobalFloat(_ozoneThicknessID, expanseSky.ozoneThickness.value);
        cmd.SetGlobalFloat(_ozoneHeightID, expanseSky.ozoneHeight.value);
        cmd.SetGlobalFloat(_ozoneDensityID, expanseSky.ozoneDensity.value);
        cmd.SetGlobalInt(_numberOfSamplesID, expanseSky.numberOfSamples.value);
        cmd.SetGlobalFloat(_useImportanceSamplingID, expanseSky.useImportanceSampling.value ? 1f : 0f);
        cmd.SetGlobalFloat(_useCubicApproximationID, expanseSky.useCubicApproximation.value ? 1f : 0f);

        /* Set the texture for the actual sky shader. */
        cmd.SetGlobalTexture(_transmittanceTableID, m_TransmittanceTables[0]);
        cmd.SetGlobalTexture(_singleScatteringTableAirID, m_SingleScatteringTables[0]);
        cmd.SetGlobalTexture(_singleScatteringTableAerosolID, m_SingleScatteringTables[1]);
    }

    void SetPrecomputeTextures() {
      int transmittanceKernelHandle =
        s_PrecomputeCS.FindKernel("COMPUTE_TRANSMITTANCE");
      /* Set the texture for the compute shader. */
      s_PrecomputeCS.SetTexture(transmittanceKernelHandle, "_TransmittanceTableRW",
        m_TransmittanceTables[0]);

      int singleScatteringKernelHandle =
        s_PrecomputeCS.FindKernel("COMPUTE_SINGLE_SCATTERING");
      /* Set the textures for the compute shader. */
      s_PrecomputeCS.SetTexture(singleScatteringKernelHandle, "_SingleScatteringTableAirRW",
        m_SingleScatteringTables[0]);
      s_PrecomputeCS.SetTexture(singleScatteringKernelHandle, "_SingleScatteringTableAerosolRW",
        m_SingleScatteringTables[1]);
    }

    void PrecomputeTables(CommandBuffer cmd)
    {
      using (new ProfilingSample(cmd, "Precompute Expanse Sky Tables"))
      {
        int transmittanceKernelHandle =
          s_PrecomputeCS.FindKernel("COMPUTE_TRANSMITTANCE");
        cmd.DispatchCompute(s_PrecomputeCS, transmittanceKernelHandle,
          TransmittanceTableSizeH / 4, TransmittanceTableSizePhi / 4, 1);

        int singleScatteringKernelHandle =
          s_PrecomputeCS.FindKernel("COMPUTE_SINGLE_SCATTERING");
        cmd.DispatchCompute(s_PrecomputeCS, singleScatteringKernelHandle,
          SingleScatteringTableSizeH / 4, SingleScatteringTableSizePhi / 4,
          (SingleScatteringTableSizePhiL * SingleScatteringTableSizeNu) / 4);
      }
    }

    protected override bool Update(BuiltinSkyParameters builtinParams)
    {
      var expanseSky = builtinParams.skySettings as ExpanseSky;
      int currentPrecomputationHash = expanseSky.GetPrecomputationHashCode();
      if (currentPrecomputationHash != m_LastPrecomputationParamHash) {
        SetGlobalConstants(builtinParams.commandBuffer, builtinParams);

        SetPrecomputeTextures();

        PrecomputeTables(builtinParams.commandBuffer);

        m_LastPrecomputationParamHash = currentPrecomputationHash;
      }

      return builtinParams.skySettings.updateMode != EnvironmentUpdateMode.Realtime;
    }

    /* As in the unity physical sky, renderSunDisk is unsupported, because we
     * instead use the celestial body system. */
    public override void RenderSky(BuiltinSkyParameters builtinParams, bool renderForCubemap, bool renderSunDisk)
    {
        /* TODO: no idea why this using thing is here. It must be a debug
         * type thing. Unity says it's obselete, but finding a way to
         * profile would actually be very cool. */
        using (new ProfilingSample(builtinParams.commandBuffer, "Draw Expanse Sky"))
        {
            var expanseSky = builtinParams.skySettings as ExpanseSky;

            int passID = renderForCubemap ? m_RenderCubemapID : m_RenderFullscreenSkyID;

            /* Set the shader uniform variables. */
            m_PropertyBlock.SetTexture(_groundColorTextureID, expanseSky.groundColorTexture.value);
            m_PropertyBlock.SetVector(_groundTintID, expanseSky.groundTint.value);
            m_PropertyBlock.SetTexture(_groundEmissiveTextureID, expanseSky.groundEmissiveTexture.value);
            m_PropertyBlock.SetFloat(_groundEmissiveMultiplierID, expanseSky.groundEmissiveMultiplier.value);
            m_PropertyBlock.SetTexture(_nightSkyHDRIID, expanseSky.nightSkyHDRI.value);
            m_PropertyBlock.SetVector(_nightTintID, expanseSky.nightTint.value);
            m_PropertyBlock.SetFloat(_nightIntensityID, expanseSky.nightIntensity.value);
            m_PropertyBlock.SetFloat(_aerosolAnisotropyID, expanseSky.aerosolAnisotropy.value);
            m_PropertyBlock.SetVector(_skyTintID, expanseSky.skyTint.value);
            m_PropertyBlock.SetFloat(_starAerosolScatterMultiplierID, expanseSky.starAerosolScatterMultiplier.value);
            m_PropertyBlock.SetFloat(_limbDarkeningID, expanseSky.limbDarkening.value);
            m_PropertyBlock.SetFloat(_ditherAmountID, expanseSky.ditherAmount.value);

            m_PropertyBlock.SetVector(_WorldSpaceCameraPos1ID, builtinParams.worldSpaceCameraPos);
            m_PropertyBlock.SetMatrix(_ViewMatrix1ID, builtinParams.viewMatrix);
            m_PropertyBlock.SetMatrix(_PixelCoordToViewDirWS, builtinParams.pixelCoordToViewDirMatrix);

            CoreUtils.DrawFullScreen(builtinParams.commandBuffer, m_ExpanseSkyMaterial, m_PropertyBlock, passID);
        }
    }
}
