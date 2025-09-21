using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

public class TextureResolutionReducer : EditorWindow
{
    private int _targetResolution = 1024;

    [MenuItem("Tools/Texture Resolution Reducer")]
    private static void ShowWindow()
    {
        var window = GetWindow<TextureResolutionReducer>(true, "Texture Resolution Reducer");
        window.minSize = new Vector2(360f, 140f);
    }

    private void OnGUI()
    {
        EditorGUILayout.LabelField("Adjust textures to the specified maximum resolution.", EditorStyles.wordWrappedLabel);
        EditorGUILayout.Space();

        _targetResolution = EditorGUILayout.IntField("Max Resolution", _targetResolution);
        if (_targetResolution < 32)
            _targetResolution = 32;

        using (new EditorGUILayout.HorizontalScope())
        {
            GUILayout.FlexibleSpace();
            if (GUILayout.Button("256", GUILayout.Width(60f)))
                _targetResolution = 256;
            if (GUILayout.Button("512", GUILayout.Width(60f)))
                _targetResolution = 512;
            if (GUILayout.Button("1024", GUILayout.Width(60f)))
                _targetResolution = 1024;
            if (GUILayout.Button("2048", GUILayout.Width(60f)))
                _targetResolution = 2048;
            if (GUILayout.Button("4096", GUILayout.Width(60f)))
                _targetResolution = 4096;
            GUILayout.FlexibleSpace();
        }

        EditorGUILayout.Space();

        if (Selection.gameObjects.Length == 0)
        {
            EditorGUILayout.HelpBox("Select one or more GameObjects in the hierarchy to process.", MessageType.Info);
        }

        EditorGUI.BeginDisabledGroup(Selection.gameObjects.Length == 0);
        if (GUILayout.Button("Apply To Selected"))
        {
            AdjustTexturesOnSelection(Selection.gameObjects, _targetResolution);
        }
        EditorGUI.EndDisabledGroup();
    }

    private static void AdjustTexturesOnSelection(GameObject[] selectedObjects, int targetResolution)
    {
        var processedTextures = new HashSet<string>();
        int adjustedCount = 0;

        try
        {
            EditorUtility.DisplayProgressBar("Texture Resolution Reducer", "Collecting textures...", 0f);
            var renderers = CollectRenderers(selectedObjects);
            int totalRenderers = renderers.Count;

            for (int i = 0; i < totalRenderers; i++)
            {
                var renderer = renderers[i];
                EditorUtility.DisplayProgressBar("Texture Resolution Reducer", renderer.name, (float)i / totalRenderers);

                var sharedMaterials = renderer.sharedMaterials;
                for (int m = 0; m < sharedMaterials.Length; m++)
                {
                    var material = sharedMaterials[m];
                    if (material == null)
                        continue;

                    adjustedCount += ProcessMaterial(material, targetResolution, processedTextures);
                }
            }
        }
        finally
        {
            EditorUtility.ClearProgressBar();
        }

        if (adjustedCount > 0)
        {
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
        }

        EditorUtility.DisplayDialog(
            "Texture Resolution Reducer",
            $"Adjusted {adjustedCount} texture(s).",
            "OK");
    }

    private static List<Renderer> CollectRenderers(GameObject[] selectedObjects)
    {
        var renderers = new List<Renderer>();
        foreach (var root in selectedObjects)
        {
            if (root == null)
                continue;

            renderers.AddRange(root.GetComponentsInChildren<Renderer>(true));
        }
        return renderers;
    }

    private static int ProcessMaterial(Material material, int targetResolution, HashSet<string> processedTextures)
    {
        int adjusted = 0;
        var shader = material.shader;
        if (shader == null)
            return 0;

        int propertyCount = ShaderUtil.GetPropertyCount(shader);
        for (int p = 0; p < propertyCount; p++)
        {
            if (ShaderUtil.GetPropertyType(shader, p) != ShaderUtil.ShaderPropertyType.TexEnv)
                continue;

            string propertyName = ShaderUtil.GetPropertyName(shader, p);
            var texture = material.GetTexture(propertyName) as Texture2D;
            if (texture == null)
                continue;

            string path = AssetDatabase.GetAssetPath(texture);
            if (string.IsNullOrEmpty(path))
                continue;

            if (!processedTextures.Add(path))
                continue;

            var importer = AssetImporter.GetAtPath(path) as TextureImporter;
            if (importer == null)
                continue;

            if (texture.width <= targetResolution && texture.height <= targetResolution && importer.maxTextureSize <= targetResolution)
                continue;

            importer.maxTextureSize = targetResolution;
            importer.SaveAndReimport();
            adjusted++;
        }

        return adjusted;
    }
}
