using System;
using System.Collections.Generic;
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
        string directory = null;
        if (!string.IsNullOrEmpty(originalPath))
        {
            directory = Path.GetDirectoryName(originalPath);
            if (string.IsNullOrEmpty(directory))
                directory = "Assets";
        }

        string fileName = BuildFileName(originalMesh, originalPath, suffix);

        foreach (string candidatePath in EnumerateCandidatePaths(directory, fileName))
        {
            string uniquePath = AssetDatabase.GenerateUniqueAssetPath(candidatePath);
            if (TryCreateAsset(mesh, uniquePath))
            {
                assetPath = uniquePath;
                return true;
            }
        }

        Debug.LogError("Failed to create a writable location for the reduced mesh asset. Please choose a folder under the Assets directory.");
        return false;
    }

    private static IEnumerable<string> EnumerateCandidatePaths(string initialDirectory, string fileName)
    {
        foreach (string candidateDirectory in BuildCandidateDirectories(initialDirectory))
        {
            if (!EnsureFolderExists(candidateDirectory))
                continue;

            yield return CombineAssetPath(candidateDirectory, fileName);
        }
    }

    private static bool TryCreateAsset(Mesh mesh, string assetPath)
    {
        try
        {
            AssetDatabase.CreateAsset(mesh, assetPath);
            Undo.RegisterCreatedObjectUndo(mesh, "Create Derived Mesh Asset");
            return true;
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"Failed to create derived mesh asset at '{assetPath}': {ex.Message}");
            return false;
        }
    }

    private static List<string> BuildCandidateDirectories(string initialDirectory)
    {
        const string FallbackDirectory = "Assets/ReducedMeshes";

        var directories = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);

        void AddDirectory(string path)
        {
            if (string.IsNullOrEmpty(path))
                return;
            if (!seen.Add(path))
                return;
            directories.Add(path);
        }

        if (IsAssetsRelativeDirectory(initialDirectory))
            AddDirectory(initialDirectory);

        AddDirectory(FallbackDirectory);
        AddDirectory("Assets");

        return directories;
    }

    private static bool EnsureFolderExists(string assetsDirectory)
    {
        if (string.IsNullOrEmpty(assetsDirectory))
            return false;

        if (!IsAssetsRelativeDirectory(assetsDirectory))
            return false;

        if (AssetDatabase.IsValidFolder(assetsDirectory))
            return true;

        string[] segments = assetsDirectory.Split(new[] { '/' }, StringSplitOptions.RemoveEmptyEntries);
        if (segments.Length == 0 || !segments[0].Equals("Assets", StringComparison.Ordinal))
            return false;

        string currentPath = segments[0];
        for (int i = 1; i < segments.Length; i++)
        {
            string segment = segments[i];
            string parent = currentPath;
            currentPath = CombineAssetPath(currentPath, segment);
            if (AssetDatabase.IsValidFolder(currentPath))
                continue;

            AssetDatabase.CreateFolder(parent, segment);
        }

        return AssetDatabase.IsValidFolder(assetsDirectory);
    }

    private static bool IsAssetsRelativeDirectory(string directory)
    {
        if (string.IsNullOrEmpty(directory))
            return false;

        return directory.Equals("Assets", StringComparison.Ordinal) ||
               directory.StartsWith("Assets/", StringComparison.Ordinal);
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
        string baseName = null;
        string extension = null;

        if (!string.IsNullOrEmpty(originalPath))
        {
            baseName = Path.GetFileNameWithoutExtension(originalPath);
            extension = Path.GetExtension(originalPath);
        }

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
