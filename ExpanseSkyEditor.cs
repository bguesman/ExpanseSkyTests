using UnityEditor.Rendering;
using UnityEngine.Rendering.HighDefinition;
using UnityEditor.Rendering.HighDefinition;

// [CanEditMultipleObjects]
[VolumeComponentEditor(typeof(ExpanseSky))]
class ExpanseSkyEditor : SkySettingsEditor
{
    /* Planet. */
    SerializedDataParameter atmosphereThickness;
    SerializedDataParameter planetRadius;
    SerializedDataParameter groundAlbedoTexture;
    SerializedDataParameter groundTint;
    SerializedDataParameter groundEmissionTexture;
    SerializedDataParameter groundEmissionMultiplier;
    SerializedDataParameter planetRotation;

    /* Night sky. */
    SerializedDataParameter nightSkyTexture;
    SerializedDataParameter nightSkyRotation;
    SerializedDataParameter nightTint;
    SerializedDataParameter nightIntensity;
    SerializedDataParameter lightPollutionTint;
    SerializedDataParameter lightPollutionIntensity;

    /* Aerosols. */
    SerializedDataParameter aerosolCoefficient;
    SerializedDataParameter scaleHeightAerosols;
    SerializedDataParameter aerosolAnisotropy;
    SerializedDataParameter aerosolDensity;

    /* Air. */
    SerializedDataParameter airCoefficients;
    SerializedDataParameter scaleHeightAir;
    SerializedDataParameter airDensity;

    /* Ozone. */
    SerializedDataParameter ozoneCoefficients;
    SerializedDataParameter ozoneThickness;
    SerializedDataParameter ozoneHeight;
    SerializedDataParameter ozoneDensity;

    /* Artistic Overrides. */
    SerializedDataParameter skyTint;
    SerializedDataParameter multipleScatteringMultiplier;

    /* Body 1. */
    SerializedDataParameter body1LimbDarkening;
    SerializedDataParameter body1ReceivesLight;
    SerializedDataParameter body1AlbedoTexture;
    SerializedDataParameter body1Emissive;
    SerializedDataParameter body1EmissionTexture;
    SerializedDataParameter body1Rotation;
    /* Body 2. */
    SerializedDataParameter body2LimbDarkening;
    SerializedDataParameter body2ReceivesLight;
    SerializedDataParameter body2AlbedoTexture;
    SerializedDataParameter body2Emissive;
    SerializedDataParameter body2EmissionTexture;
    SerializedDataParameter body2Rotation;
    /* Body 3. */
    SerializedDataParameter body3LimbDarkening;
    SerializedDataParameter body3ReceivesLight;
    SerializedDataParameter body3AlbedoTexture;
    SerializedDataParameter body3Emissive;
    SerializedDataParameter body3EmissionTexture;
    SerializedDataParameter body3Rotation;
    /* Body 4. */
    SerializedDataParameter body4LimbDarkening;
    SerializedDataParameter body4ReceivesLight;
    SerializedDataParameter body4AlbedoTexture;
    SerializedDataParameter body4Emissive;
    SerializedDataParameter body4EmissionTexture;
    SerializedDataParameter body4Rotation;

    /* Sampling and Rendering. */
    SerializedDataParameter numberOfTransmittanceSamples;
    SerializedDataParameter numberOfLightPollutionSamples;
    SerializedDataParameter numberOfScatteringSamples;
    SerializedDataParameter numberOfGroundIrradianceSamples;
    SerializedDataParameter numberOfMultipleScatteringSamples;
    SerializedDataParameter numberOfMultipleScatteringAccumulationSamples;
    SerializedDataParameter useImportanceSampling;
    SerializedDataParameter useAntiAliasing;
    SerializedDataParameter ditherAmount;

    public override void OnEnable()
    {
        base.OnEnable();

        m_CommonUIElementsMask = (uint)SkySettingsUIElement.UpdateMode;

        var o = new PropertyFetcher<ExpanseSky>(serializedObject);

        atmosphereThickness = Unpack(o.Find(x => x.atmosphereThickness));
        planetRadius = Unpack(o.Find(x => x.planetRadius));
        groundAlbedoTexture = Unpack(o.Find(x => x.groundAlbedoTexture));
        groundTint = Unpack(o.Find(x => x.groundTint));
        groundEmissionTexture = Unpack(o.Find(x => x.groundEmissionTexture));
        groundEmissionMultiplier = Unpack(o.Find(x => x.groundEmissionMultiplier));
        planetRotation = Unpack(o.Find(x => x.planetRotation));

        /* Night sky. */
        nightSkyTexture = Unpack(o.Find(x => x.nightSkyTexture));
        nightSkyRotation = Unpack(o.Find(x => x.nightSkyRotation));
        nightTint = Unpack(o.Find(x => x.nightTint));
        nightIntensity = Unpack(o.Find(x => x.nightIntensity));
        lightPollutionTint = Unpack(o.Find(x => x.lightPollutionTint));
        lightPollutionIntensity = Unpack(o.Find(x => x.lightPollutionIntensity));

        /* Aerosols. */
        aerosolCoefficient = Unpack(o.Find(x => x.aerosolCoefficient));
        scaleHeightAerosols = Unpack(o.Find(x => x.scaleHeightAerosols));
        aerosolAnisotropy = Unpack(o.Find(x => x.aerosolAnisotropy));
        aerosolDensity = Unpack(o.Find(x => x.aerosolDensity));

        /* Air. */
        airCoefficients = Unpack(o.Find(x => x.airCoefficients));
        scaleHeightAir = Unpack(o.Find(x => x.scaleHeightAir));
        airDensity = Unpack(o.Find(x => x.airDensity));

        /* Ozone. */
        ozoneCoefficients = Unpack(o.Find(x => x.ozoneCoefficients));
        ozoneThickness = Unpack(o.Find(x => x.ozoneThickness));
        ozoneHeight = Unpack(o.Find(x => x.ozoneHeight));
        ozoneDensity = Unpack(o.Find(x => x.ozoneDensity));

        /* Artistic Overrides. */
        skyTint = Unpack(o.Find(x => x.ozoneHeight));
        multipleScatteringMultiplier = Unpack(o.Find(x => x.multipleScatteringMultiplier));

        /* Body 1. */
        body1LimbDarkening = Unpack(o.Find(x => x.body1LimbDarkening));
        body1ReceivesLight = Unpack(o.Find(x => x.body1ReceivesLight));
        body1AlbedoTexture = Unpack(o.Find(x => x.body1AlbedoTexture));
        body1Emissive = Unpack(o.Find(x => x.body1Emissive));
        body1EmissionTexture = Unpack(o.Find(x => x.body1EmissionTexture));
        body1Rotation = Unpack(o.Find(x => x.body1Rotation));
        /* Body 2. */
        body2LimbDarkening = Unpack(o.Find(x => x.body2LimbDarkening));
        body2ReceivesLight = Unpack(o.Find(x => x.body2ReceivesLight));
        body2AlbedoTexture = Unpack(o.Find(x => x.body2AlbedoTexture));
        body2Emissive = Unpack(o.Find(x => x.body2Emissive));
        body2EmissionTexture = Unpack(o.Find(x => x.body2EmissionTexture));
        body2Rotation = Unpack(o.Find(x => x.body2Rotation));
        /* Body 3. */
        body3LimbDarkening = Unpack(o.Find(x => x.body3LimbDarkening));
        body3ReceivesLight = Unpack(o.Find(x => x.body3ReceivesLight));
        body3AlbedoTexture = Unpack(o.Find(x => x.body3AlbedoTexture));
        body3Emissive = Unpack(o.Find(x => x.body3Emissive));
        body3EmissionTexture = Unpack(o.Find(x => x.body3EmissionTexture));
        body3Rotation = Unpack(o.Find(x => x.body3Rotation));
        /* Body 4. */
        body4LimbDarkening = Unpack(o.Find(x => x.body4LimbDarkening));
        body4ReceivesLight = Unpack(o.Find(x => x.body4ReceivesLight));
        body4AlbedoTexture = Unpack(o.Find(x => x.body4AlbedoTexture));
        body4Emissive = Unpack(o.Find(x => x.body4Emissive));
        body4EmissionTexture = Unpack(o.Find(x => x.body4EmissionTexture));
        body4Rotation = Unpack(o.Find(x => x.body4Rotation));

        /* Sampling and Rendering. */
        numberOfTransmittanceSamples = Unpack(o.Find(x => x.numberOfTransmittanceSamples));
        numberOfLightPollutionSamples = Unpack(o.Find(x => x.numberOfLightPollutionSamples));
        numberOfScatteringSamples = Unpack(o.Find(x => x.numberOfScatteringSamples));
        numberOfGroundIrradianceSamples = Unpack(o.Find(x => x.numberOfGroundIrradianceSamples));
        numberOfMultipleScatteringSamples = Unpack(o.Find(x => x.numberOfMultipleScatteringSamples));
        numberOfMultipleScatteringAccumulationSamples = Unpack(o.Find(x => x.numberOfMultipleScatteringAccumulationSamples));
        useImportanceSampling = Unpack(o.Find(x => x.useImportanceSampling));
        useAntiAliasing = Unpack(o.Find(x => x.useAntiAliasing));
        ditherAmount = Unpack(o.Find(x => x.ditherAmount));
    }

    public override void OnInspectorGUI()
    {
      // var titleStyle : GUIStyle;
      UnityEngine.GUIStyle titleStyle = new UnityEngine.GUIStyle();
      titleStyle.fontSize = 16;
      UnityEngine.GUIStyle subtitleStyle = new UnityEngine.GUIStyle();
      subtitleStyle.fontSize = 12;

      /* Planet. */
      UnityEditor.EditorGUILayout.LabelField("Planet", titleStyle);
      PropertyField(atmosphereThickness);
      PropertyField(planetRadius);
      PropertyField(groundAlbedoTexture);
      PropertyField(groundTint);
      PropertyField(groundEmissionTexture);
      PropertyField(groundEmissionMultiplier);
      PropertyField(planetRotation);

      /* Night sky. */
      UnityEditor.EditorGUILayout.LabelField("", titleStyle);
      UnityEditor.EditorGUILayout.LabelField("Night Sky", titleStyle);
      PropertyField(nightSkyTexture);
      PropertyField(nightSkyRotation);
      PropertyField(nightTint);
      PropertyField(nightIntensity);
      PropertyField(lightPollutionTint);
      PropertyField(lightPollutionIntensity);

      /* Aerosols. */
      UnityEditor.EditorGUILayout.LabelField("", titleStyle);
      UnityEditor.EditorGUILayout.LabelField("Aerosol Layer", titleStyle);
      PropertyField(aerosolCoefficient);
      PropertyField(scaleHeightAerosols);
      PropertyField(aerosolAnisotropy);
      PropertyField(aerosolDensity);

      /* Air. */
      UnityEditor.EditorGUILayout.LabelField("", titleStyle);
      UnityEditor.EditorGUILayout.LabelField("Air Layer", titleStyle);
      PropertyField(airCoefficients);
      PropertyField(scaleHeightAir);
      PropertyField(airDensity);

      /* Ozone. */
      UnityEditor.EditorGUILayout.LabelField("", titleStyle);
      UnityEditor.EditorGUILayout.LabelField("Ozone Layer", titleStyle);
      PropertyField(ozoneCoefficients);
      PropertyField(ozoneThickness);
      PropertyField(ozoneHeight);
      PropertyField(ozoneDensity);

      /* Artistic Overrides. */
      UnityEditor.EditorGUILayout.LabelField("", titleStyle);
      UnityEditor.EditorGUILayout.LabelField("Artistic Overrides", titleStyle);
      PropertyField(skyTint);
      PropertyField(multipleScatteringMultiplier);

      /* Body 1. */
      UnityEditor.EditorGUILayout.LabelField("", titleStyle);
      UnityEditor.EditorGUILayout.LabelField("Celestial Bodies", titleStyle);
      UnityEditor.EditorGUILayout.LabelField("", subtitleStyle);
      UnityEditor.EditorGUILayout.LabelField("Celestial Body 1", subtitleStyle);
      PropertyField(body1LimbDarkening);
      PropertyField(body1ReceivesLight);
      PropertyField(body1AlbedoTexture);
      PropertyField(body1Emissive);
      PropertyField(body1EmissionTexture);
      PropertyField(body1Rotation);
      /* Body 2. */
      UnityEditor.EditorGUILayout.LabelField("", subtitleStyle);
      UnityEditor.EditorGUILayout.LabelField("Celestial Body 2", subtitleStyle);
      PropertyField(body2LimbDarkening);
      PropertyField(body2ReceivesLight);
      PropertyField(body2AlbedoTexture);
      PropertyField(body2Emissive);
      PropertyField(body2EmissionTexture);
      PropertyField(body2Rotation);
      /* Body 3. */
      UnityEditor.EditorGUILayout.LabelField("", subtitleStyle);
      UnityEditor.EditorGUILayout.LabelField("Celestial Body 3", subtitleStyle);
      PropertyField(body3LimbDarkening);
      PropertyField(body3ReceivesLight);
      PropertyField(body3AlbedoTexture);
      PropertyField(body3Emissive);
      PropertyField(body3EmissionTexture);
      PropertyField(body3Rotation);
      /* Body 4. */
      UnityEditor.EditorGUILayout.LabelField("", subtitleStyle);
      UnityEditor.EditorGUILayout.LabelField("Celestial Body 4", subtitleStyle);
      PropertyField(body4LimbDarkening);
      PropertyField(body4ReceivesLight);
      PropertyField(body4AlbedoTexture);
      PropertyField(body4Emissive);
      PropertyField(body4EmissionTexture);
      PropertyField(body4Rotation);

      /* Sampling and Rendering. */
      UnityEditor.EditorGUILayout.LabelField("", titleStyle);
      UnityEditor.EditorGUILayout.LabelField("Sampling and Rendering", titleStyle);
      PropertyField(numberOfTransmittanceSamples);
      PropertyField(numberOfLightPollutionSamples);
      PropertyField(numberOfScatteringSamples);
      PropertyField(numberOfGroundIrradianceSamples);
      PropertyField(numberOfMultipleScatteringSamples);
      PropertyField(numberOfMultipleScatteringAccumulationSamples);
      PropertyField(useImportanceSampling);
      PropertyField(useAntiAliasing);
      PropertyField(ditherAmount);

      base.CommonSkySettingsGUI();
    }
}
