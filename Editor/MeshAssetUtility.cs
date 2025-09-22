using System;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

public static class MeshAssetUtility
{
    public static bool TryCreateDerivedMeshAsset(Mesh mesh, Mesh originalMesh, string suffix, out string assetPath)
    {
        assetPath = null;
        if (mesh == null || originalMesh == null)
            return false;

        string originalPath = AssetDatabase.GetAssetPath(originalMesh);
        if (string.IsNullOrEmpty(originalPath))
            return false;

        string directory = Path.GetDirectoryName(originalPath);
        if (string.IsNullOrEmpty(directory))
            directory = "Assets";

        string fileName = BuildFileName(originalMesh, originalPath, suffix);
        string candidatePath = CombineAssetPath(directory, fileName);
        candidatePath = AssetDatabase.GenerateUniqueAssetPath(candidatePath);

        try
        {
            AssetDatabase.CreateAsset(mesh, candidatePath);
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to create derived mesh asset at '{candidatePath}': {ex.Message}");
            return false;
        }

        Undo.RegisterCreatedObjectUndo(mesh, "Create Derived Mesh Asset");
        assetPath = candidatePath;
        return true;
    }

    private static string CombineAssetPath(string directory, string fileName)
    {
        if (string.IsNullOrEmpty(directory))
            return fileName;

        if (directory.EndsWith("/", StringComparison.Ordinal))
            return directory + fileName;

        return directory + "/" + fileName;
    }

    private static string BuildFileName(Mesh originalMesh, string originalPath, string suffix)
    {
        string baseName = Path.GetFileNameWithoutExtension(originalPath);
        string extension = Path.GetExtension(originalPath);

        if (!string.IsNullOrEmpty(extension) && extension.Equals(".fbx", StringComparison.OrdinalIgnoreCase))
        {
            if (!string.IsNullOrEmpty(originalMesh.name))
                baseName = originalMesh.name;
        }

        if (string.IsNullOrEmpty(baseName))
            baseName = !string.IsNullOrEmpty(originalMesh.name) ? originalMesh.name : "Mesh";

        baseName = SanitizeFileName(baseName);

        string suffixPart = string.IsNullOrEmpty(suffix)
            ? string.Empty
            : (suffix.StartsWith("_", StringComparison.Ordinal) ? suffix : "_" + suffix);

        return baseName + suffixPart + ".asset";
    }

    private static string SanitizeFileName(string name)
    {
        if (string.IsNullOrEmpty(name))
            return "Mesh";

        var invalidChars = Path.GetInvalidFileNameChars();
        var builder = new StringBuilder(name.Length);
        foreach (char c in name)
        {
            bool invalid = false;
            for (int i = 0; i < invalidChars.Length; i++)
            {
                if (c == invalidChars[i])
                {
                    invalid = true;
                    break;
                }
            }

            builder.Append(invalid ? '_' : c);
        }

        string sanitized = builder.ToString().Trim();
        return string.IsNullOrEmpty(sanitized) ? "Mesh" : sanitized;
    }
}
