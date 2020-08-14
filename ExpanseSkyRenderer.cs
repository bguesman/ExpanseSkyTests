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
    public static readonly int _multipleScatteringMultiplierID = Shader.PropertyToID("_multipleScatteringMultiplier");
    public static readonly int _limbDarkeningID = Shader.PropertyToID("_limbDarkening");
    public static readonly int _numberOfTransmittanceSamplesID = Shader.PropertyToID("_numberOfTransmittanceSamples");
    public static readonly int _numberOfScatteringSamplesID = Shader.PropertyToID("_numberOfScatteringSamples");
    public static readonly int _numberOfGroundIrradianceSamplesID = Shader.PropertyToID("_numberOfGroundIrradianceSamples");
    public static readonly int _useImportanceSamplingID = Shader.PropertyToID("_useImportanceSampling");
    public static readonly int _ditherAmountID = Shader.PropertyToID("_ditherAmount");

    public static readonly int _WorldSpaceCameraPos1ID = Shader.PropertyToID("_WorldSpaceCameraPos1");
    public static readonly int _ViewMatrix1ID = Shader.PropertyToID("_ViewMatrix1");
    public static readonly int _PixelCoordToViewDirWS = Shader.PropertyToID("_PixelCoordToViewDirWS");

    /* Tables. */
    public static readonly int _transmittanceTableID = Shader.PropertyToID("_TransmittanceTable");
    public static readonly int _groundIrradianceTableAirID = Shader.PropertyToID("_GroundIrradianceTableAir");
    public static readonly int _groundIrradianceTableAerosolID = Shader.PropertyToID("_GroundIrradianceTableAerosol");
    public static readonly int _singleScatteringTableAirID = Shader.PropertyToID("_SingleScatteringTableAir");
    public static readonly int _singleScatteringTableAerosolID = Shader.PropertyToID("_SingleScatteringTableAerosol");
    public static readonly int _singleScatteringTableAirNoShadowsID = Shader.PropertyToID("_SingleScatteringTableAirNoShadows");
    public static readonly int _singleScatteringTableAerosolNoShadowsID = Shader.PropertyToID("_SingleScatteringTableAerosolNoShadows");
    public static readonly int _localMultipleScatteringTableID = Shader.PropertyToID("_LocalMultipleScatteringTable");
    public static readonly int _globalMultipleScatteringTableAirID = Shader.PropertyToID("_GlobalMultipleScatteringTableAir");
    public static readonly int _globalMultipleScatteringTableAerosolID = Shader.PropertyToID("_GlobalMultipleScatteringTableAerosol");
    /********************************************************************************/
    /************************** End Shader Variable ID's ****************************/
    /********************************************************************************/

    /* Dimensions of precomputed tables. */
    const int TransmittanceTableSizeH = 32;
    const int TransmittanceTableSizePhi = 128;

    const int SingleScatteringTableSizeH = 32;
    const int SingleScatteringTableSizePhi = 128;
    const int SingleScatteringTableSizePhiL = 32;
    const int SingleScatteringTableSizeNu = 32;

    const int GroundIrradianceTableSize = 256;

    const int LocalMultipleScatteringTableSizeH = 32;
    const int LocalMultipleScatteringTableSizePhiL = 32;

    const int GlobalMultipleScatteringTableSizeH = 32;
    const int GlobalMultipleScatteringTableSizePhi = 128;
    const int GlobalMultipleScatteringTableSizePhiL = 32;
    const int GlobalMultipleScatteringTableSizeNu = 32;

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

    /* Ground irradiance table. Leverages spherical symmetry of the atmosphere,
     * parameterized by:
     * phi (x dimension): dot product between the surface normal and the
     * light direction. */
    RTHandle[]                   m_GroundIrradianceTables;

    /* Multiple scattering table. Leverages spherical symmetry of the atmosphere,
     * parameterized by:
     * h (x dimension): the height of the camera.
     * phi_l (y dimension): dot product between the surface normal and the
     * light direction. */
    RTHandle[]                   m_LocalMultipleScatteringTables;

    /* Multiple scattering table. Leverages spherical symmetry of the atmosphere,
     * parameterized by:
     * h (x dimension): the height of the camera.
     * phi_l (y dimension): dot product between the surface normal and the
     * light direction. */
    RTHandle[]                   m_GlobalMultipleScatteringTables;

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

    RTHandle AllocateGroundIrradianceTable(int index)
    {
        var table = RTHandles.Alloc(GroundIrradianceTableSize,
                                    1,
                                    dimension: TextureDimension.Tex2D,
                                    colorFormat: s_ColorFormat,
                                    enableRandomWrite: true,
                                    name: string.Format("GroundIrradianceTable{0}", index));

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

    RTHandle AllocateLocalMultipleScatteringTable(int index)
    {
        var table = RTHandles.Alloc(LocalMultipleScatteringTableSizeH,
                                    LocalMultipleScatteringTableSizePhiL,
                                    dimension: TextureDimension.Tex2D,
                                    colorFormat: s_ColorFormat,
                                    enableRandomWrite: true,
                                    name: string.Format("LocalMultipleScatteringTable{0}", index));

        Debug.Assert(table != null);

        return table;
    }

    RTHandle AllocateGlobalMultipleScatteringTable(int index)
    {
        var table = RTHandles.Alloc(GlobalMultipleScatteringTableSizeH,
                                    GlobalMultipleScatteringTableSizePhi,
                                    GlobalMultipleScatteringTableSizePhiL *
                                    GlobalMultipleScatteringTableSizeNu,
                                    dimension: TextureDimension.Tex3D,
                                    colorFormat: s_ColorFormat,
                                    enableRandomWrite: true,
                                    name: string.Format("GlobalMultipleScatteringTable{0}", index));

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

      m_GroundIrradianceTables = new RTHandle[2];
      m_GroundIrradianceTables[0] = AllocateGroundIrradianceTable(0);
      m_GroundIrradianceTables[1] = AllocateGroundIrradianceTable(1);

      m_SingleScatteringTables = new RTHandle[4];
      m_SingleScatteringTables[0] = AllocateSingleScatteringTable(0);
      m_SingleScatteringTables[1] = AllocateSingleScatteringTable(1);
      m_SingleScatteringTables[2] = AllocateSingleScatteringTable(2);
      m_SingleScatteringTables[3] = AllocateSingleScatteringTable(3);

      m_LocalMultipleScatteringTables = new RTHandle[2];
      m_LocalMultipleScatteringTables[0] = AllocateLocalMultipleScatteringTable(0);
      m_LocalMultipleScatteringTables[1] = AllocateLocalMultipleScatteringTable(1);

      m_GlobalMultipleScatteringTables = new RTHandle[2];
      m_GlobalMultipleScatteringTables[0] = AllocateGlobalMultipleScatteringTable(0);
      m_GlobalMultipleScatteringTables[1] = AllocateGlobalMultipleScatteringTable(1);
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
        RTHandles.Release(m_SingleScatteringTables[2]);
        m_SingleScatteringTables[2] = null;
        RTHandles.Release(m_SingleScatteringTables[3]);
        m_SingleScatteringTables[3] = null;

        RTHandles.Release(m_GroundIrradianceTables[0]);
        m_GroundIrradianceTables[0] = null;
        RTHandles.Release(m_GroundIrradianceTables[1]);
        m_GroundIrradianceTables[1] = null;

        RTHandles.Release(m_LocalMultipleScatteringTables[0]);
        m_LocalMultipleScatteringTables[0] = null;
        RTHandles.Release(m_LocalMultipleScatteringTables[1]);
        m_LocalMultipleScatteringTables[1] = null;

        RTHandles.Release(m_GlobalMultipleScatteringTables[0]);
        m_GlobalMultipleScatteringTables[0] = null;
        RTHandles.Release(m_GlobalMultipleScatteringTables[1]);
        m_GlobalMultipleScatteringTables[1] = null;
    }

    void SetGlobalConstants(CommandBuffer cmd, BuiltinSkyParameters builtinParams) {

        var expanseSky = builtinParams.skySettings as ExpanseSky;

        /* Set the parameters that need to be used for sky table
         * precomputation, aka, everything in ExpanseSkyCommon.hlsl. */
        cmd.SetGlobalFloat(_atmosphereThicknessID, expanseSky.atmosphereThickness.value);
        cmd.SetGlobalFloat(_planetRadiusID, expanseSky.planetRadius.value);
        cmd.SetGlobalFloat(_atmosphereRadiusID, expanseSky.atmosphereThickness.value + expanseSky.planetRadius.value);
        cmd.SetGlobalVector(_groundTintID, expanseSky.groundTint.value);
        cmd.SetGlobalFloat(_aerosolCoefficientID, expanseSky.aerosolCoefficient.value);
        cmd.SetGlobalFloat(_scaleHeightAerosolsID, expanseSky.scaleHeightAerosols.value);
        cmd.SetGlobalFloat(_aerosolAnisotropyID, expanseSky.aerosolAnisotropy.value);
        cmd.SetGlobalFloat(_aerosolDensityID, expanseSky.aerosolDensity.value);
        cmd.SetGlobalVector(_airCoefficientsID, expanseSky.airCoefficients.value);
        cmd.SetGlobalFloat(_scaleHeightAirID, expanseSky.scaleHeightAir.value);
        cmd.SetGlobalFloat(_airDensityID, expanseSky.airDensity.value);
        cmd.SetGlobalVector(_ozoneCoefficientsID, expanseSky.ozoneCoefficients.value);
        cmd.SetGlobalFloat(_ozoneThicknessID, expanseSky.ozoneThickness.value);
        cmd.SetGlobalFloat(_ozoneHeightID, expanseSky.ozoneHeight.value);
        cmd.SetGlobalFloat(_ozoneDensityID, expanseSky.ozoneDensity.value);
        cmd.SetGlobalInt(_numberOfTransmittanceSamplesID, expanseSky.numberOfTransmittanceSamples.value);
        cmd.SetGlobalInt(_numberOfScatteringSamplesID, expanseSky.numberOfScatteringSamples.value);
        cmd.SetGlobalInt(_numberOfGroundIrradianceSamplesID, expanseSky.numberOfGroundIrradianceSamples.value);
        cmd.SetGlobalFloat(_useImportanceSamplingID, expanseSky.useImportanceSampling.value ? 1f : 0f);

        /* Set the texture for the actual sky shader. */
        cmd.SetGlobalTexture(_transmittanceTableID, m_TransmittanceTables[0]);
        cmd.SetGlobalTexture(_groundIrradianceTableAirID, m_GroundIrradianceTables[0]);
        cmd.SetGlobalTexture(_groundIrradianceTableAerosolID, m_GroundIrradianceTables[1]);
        cmd.SetGlobalTexture(_singleScatteringTableAirID, m_SingleScatteringTables[0]);
        cmd.SetGlobalTexture(_singleScatteringTableAerosolID, m_SingleScatteringTables[1]);
        cmd.SetGlobalTexture(_singleScatteringTableAirNoShadowsID, m_SingleScatteringTables[2]);
        cmd.SetGlobalTexture(_singleScatteringTableAerosolNoShadowsID, m_SingleScatteringTables[3]);
        cmd.SetGlobalTexture(_localMultipleScatteringTableID, m_LocalMultipleScatteringTables[0]);
        cmd.SetGlobalTexture(_globalMultipleScatteringTableAirID, m_GlobalMultipleScatteringTables[0]);
        cmd.SetGlobalTexture(_globalMultipleScatteringTableAerosolID, m_GlobalMultipleScatteringTables[1]);
    }

    void SetPrecomputeTextures() {
      int transmittanceKernelHandle =
        s_PrecomputeCS.FindKernel("COMPUTE_TRANSMITTANCE");
      /* Set the texture for the compute shader. */
      s_PrecomputeCS.SetTexture(transmittanceKernelHandle, "_TransmittanceTableRW",
        m_TransmittanceTables[0]);

      int groundIrradianceKernelHandle =
        s_PrecomputeCS.FindKernel("COMPUTE_GROUND_IRRADIANCE");
      /* Set the textures for the compute shader. */
      s_PrecomputeCS.SetTexture(groundIrradianceKernelHandle, "_GroundIrradianceTableAirRW",
        m_GroundIrradianceTables[0]);
      s_PrecomputeCS.SetTexture(groundIrradianceKernelHandle, "_GroundIrradianceTableAerosolRW",
        m_GroundIrradianceTables[1]);

      int singleScatteringKernelHandle =
        s_PrecomputeCS.FindKernel("COMPUTE_SINGLE_SCATTERING");
      /* Set the textures for the compute shader. */
      s_PrecomputeCS.SetTexture(singleScatteringKernelHandle, "_SingleScatteringTableAirRW",
        m_SingleScatteringTables[0]);
      s_PrecomputeCS.SetTexture(singleScatteringKernelHandle, "_SingleScatteringTableAerosolRW",
        m_SingleScatteringTables[1]);
      s_PrecomputeCS.SetTexture(singleScatteringKernelHandle, "_SingleScatteringTableAirNoShadowsRW",
        m_SingleScatteringTables[2]);
      s_PrecomputeCS.SetTexture(singleScatteringKernelHandle, "_SingleScatteringTableAerosolNoShadowsRW",
        m_SingleScatteringTables[3]);

      int localMultipleScatteringKernelHandle =
        s_PrecomputeCS.FindKernel("COMPUTE_LOCAL_MULTIPLE_SCATTERING");
      /* Set the textures for the compute shader. */
      s_PrecomputeCS.SetTexture(localMultipleScatteringKernelHandle, "_LocalMultipleScatteringTableRW",
        m_LocalMultipleScatteringTables[0]);

      int globalMultipleScatteringKernelHandle =
        s_PrecomputeCS.FindKernel("COMPUTE_GLOBAL_MULTIPLE_SCATTERING");
      /* Set the textures for the compute shader. */
      s_PrecomputeCS.SetTexture(globalMultipleScatteringKernelHandle, "_GlobalMultipleScatteringTableAirRW",
        m_GlobalMultipleScatteringTables[0]);
      s_PrecomputeCS.SetTexture(globalMultipleScatteringKernelHandle, "_GlobalMultipleScatteringTableAerosolRW",
        m_GlobalMultipleScatteringTables[1]);
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

        int localMultipleScatteringKernelHandle =
          s_PrecomputeCS.FindKernel("COMPUTE_LOCAL_MULTIPLE_SCATTERING");
        cmd.DispatchCompute(s_PrecomputeCS, localMultipleScatteringKernelHandle,
          LocalMultipleScatteringTableSizeH / 4,
          LocalMultipleScatteringTableSizeH / 4, 1);

        int globalMultipleScatteringKernelHandle =
          s_PrecomputeCS.FindKernel("COMPUTE_GLOBAL_MULTIPLE_SCATTERING");
        cmd.DispatchCompute(s_PrecomputeCS, globalMultipleScatteringKernelHandle,
          GlobalMultipleScatteringTableSizeH / 4,
          GlobalMultipleScatteringTableSizePhi / 4,
          (GlobalMultipleScatteringTableSizePhiL * GlobalMultipleScatteringTableSizeNu) / 4);

        int groundIrradianceKernelHandle =
          s_PrecomputeCS.FindKernel("COMPUTE_GROUND_IRRADIANCE");
        cmd.DispatchCompute(s_PrecomputeCS, groundIrradianceKernelHandle,
          GroundIrradianceTableSize / 4, 1, 1);
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
            m_PropertyBlock.SetTexture(_groundEmissiveTextureID, expanseSky.groundEmissiveTexture.value);
            m_PropertyBlock.SetFloat(_groundEmissiveMultiplierID, expanseSky.groundEmissiveMultiplier.value);
            m_PropertyBlock.SetTexture(_nightSkyHDRIID, expanseSky.nightSkyHDRI.value);
            m_PropertyBlock.SetVector(_nightTintID, expanseSky.nightTint.value);
            m_PropertyBlock.SetFloat(_nightIntensityID, expanseSky.nightIntensity.value);
            m_PropertyBlock.SetVector(_skyTintID, expanseSky.skyTint.value);
            m_PropertyBlock.SetFloat(_starAerosolScatterMultiplierID, expanseSky.starAerosolScatterMultiplier.value);
            m_PropertyBlock.SetFloat(_multipleScatteringMultiplierID, expanseSky.multipleScatteringMultiplier.value);
            m_PropertyBlock.SetFloat(_limbDarkeningID, expanseSky.limbDarkening.value);
            m_PropertyBlock.SetFloat(_ditherAmountID, expanseSky.ditherAmount.value);

            m_PropertyBlock.SetVector(_WorldSpaceCameraPos1ID, builtinParams.worldSpaceCameraPos);
            m_PropertyBlock.SetMatrix(_ViewMatrix1ID, builtinParams.viewMatrix);
            m_PropertyBlock.SetMatrix(_PixelCoordToViewDirWS, builtinParams.pixelCoordToViewDirMatrix);

            CoreUtils.DrawFullScreen(builtinParams.commandBuffer, m_ExpanseSkyMaterial, m_PropertyBlock, passID);
        }
    }
}
