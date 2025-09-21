using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

public static class MeshHiddenRemover
{
    private const float DefaultBoundsPadding = 0.005f;
    private static readonly Vector3[] RayDirections =
    {
        Vector3.right,
        Vector3.up,
        Vector3.forward,
        new Vector3(1f, 1f, 1f).normalized,
        new Vector3(-1f, 0.5f, 0.25f).normalized
    };

    [MenuItem("Tools/Mesh/Remove Fully Hidden Meshes", priority = 200)]
    public static void RemoveHiddenMeshesMenu()
    {
        if (Selection.activeGameObject == null)
        {
            Debug.LogWarning("No GameObject selected.");
            return;
        }

        RemoveHiddenMeshes(Selection.activeGameObject);
    }

    [MenuItem("Tools/Mesh/Remove Fully Hidden Meshes", true)]
    private static bool RemoveHiddenMeshesMenuValidate()
    {
        return Selection.activeGameObject != null;
    }

    public static void RemoveHiddenMeshes(GameObject root, float boundsPadding = DefaultBoundsPadding)
    {
        if (root == null)
        {
            Debug.LogWarning("Root GameObject is null.");
            return;
        }

        var renderers = root.GetComponentsInChildren<Renderer>(true);
        if (renderers.Length == 0)
        {
            Debug.LogWarning($"No renderers found under '{root.name}'.");
            return;
        }

        var geometryCache = new Dictionary<Renderer, MeshGeometry>();
        foreach (var renderer in renderers)
        {
            var geometry = BuildMeshGeometry(renderer, boundsPadding);
            if (geometry != null && geometry.Vertices.Length > 0)
                geometryCache[renderer] = geometry;
        }

        if (geometryCache.Count == 0)
        {
            Debug.LogWarning($"No mesh data could be generated under '{root.name}'.");
            return;
        }

        var toRemove = new List<Renderer>();
        foreach (var pair in geometryCache)
        {
            if (IsRendererFullyHidden(pair.Key, pair.Value, geometryCache, boundsPadding))
                toRemove.Add(pair.Key);
        }

        if (toRemove.Count == 0)
        {
            Debug.Log($"No fully hidden meshes detected under '{root.name}'.");
            return;
        }

        foreach (var renderer in toRemove)
        {
            if (renderer == null)
                continue;

            if (renderer is SkinnedMeshRenderer)
            {
                Undo.DestroyObjectImmediate(renderer);
            }
            else if (renderer is MeshRenderer meshRenderer)
            {
                var filter = meshRenderer.GetComponent<MeshFilter>();
                Undo.DestroyObjectImmediate(meshRenderer);
                if (filter != null)
                    Undo.DestroyObjectImmediate(filter);
            }
            else
            {
                Undo.DestroyObjectImmediate(renderer);
            }
        }

        Debug.Log($"Removed {toRemove.Count} fully hidden mesh renderers under '{root.name}'.");
    }

    private static MeshGeometry BuildMeshGeometry(Renderer renderer, float boundsPadding)
    {
        if (renderer == null)
            return null;

        Mesh mesh = null;
        Matrix4x4 matrix = renderer.localToWorldMatrix;
        bool baked = false;

        if (renderer is SkinnedMeshRenderer skinned)
        {
            if (skinned.sharedMesh == null)
                return null;

            mesh = new Mesh();
            skinned.BakeMesh(mesh, true);
            baked = true;
        }
        else if (renderer is MeshRenderer)
        {
            var filter = renderer.GetComponent<MeshFilter>();
            if (filter == null || filter.sharedMesh == null)
                return null;

            mesh = filter.sharedMesh;
        }

        if (mesh == null)
            return null;

        var worldVertices = new Vector3[mesh.vertexCount];
        for (int i = 0; i < mesh.vertexCount; i++)
            worldVertices[i] = matrix.MultiplyPoint3x4(mesh.vertices[i]);

        var triangles = mesh.triangles;
        var bounds = CalculateBounds(worldVertices, boundsPadding);

        if (baked)
            Object.DestroyImmediate(mesh);

        return new MeshGeometry(worldVertices, triangles, bounds);
    }

    private static Bounds CalculateBounds(Vector3[] vertices, float padding)
    {
        if (vertices == null || vertices.Length == 0)
            return new Bounds(Vector3.zero, Vector3.zero);

        var min = vertices[0];
        var max = vertices[0];
        for (int i = 1; i < vertices.Length; i++)
        {
            var v = vertices[i];
            min = Vector3.Min(min, v);
            max = Vector3.Max(max, v);
        }

        var size = max - min;
        var center = min + size * 0.5f;
        var bounds = new Bounds(center, size);
        bounds.Expand(padding * 2f);
        return bounds;
    }

    private static bool IsRendererFullyHidden(Renderer targetRenderer, MeshGeometry targetGeometry, Dictionary<Renderer, MeshGeometry> geometryCache, float boundsPadding)
    {
        if (targetRenderer == null || targetGeometry == null)
            return false;

        var vertices = targetGeometry.Vertices;
        for (int i = 0; i < vertices.Length; i++)
        {
            var vertex = vertices[i];
            bool covered = false;
            foreach (var pair in geometryCache)
            {
                if (pair.Key == targetRenderer)
                    continue;

                var occluder = pair.Value;
                if (!ContainsWithPadding(occluder.Bounds, vertex, boundsPadding))
                    continue;

                if (IsPointInsideMesh(vertex, occluder))
                {
                    covered = true;
                    break;
                }
            }

            if (!covered)
                return false;
        }

        return true;
    }

    private static bool ContainsWithPadding(Bounds bounds, Vector3 point, float padding)
    {
        bounds.Expand(padding * 2f);
        return bounds.Contains(point);
    }

    private static bool IsPointInsideMesh(Vector3 point, MeshGeometry geometry)
    {
        foreach (var direction in RayDirections)
        {
            if (IsPointInsideMesh(point, direction, geometry))
                return true;
        }

        return false;
    }

    private static bool IsPointInsideMesh(Vector3 point, Vector3 direction, MeshGeometry geometry)
    {
        const float epsilon = 1e-5f;
        int hits = 0;
        var origin = point + direction * epsilon;
        var triangles = geometry.Triangles;
        var vertices = geometry.Vertices;

        for (int i = 0; i < triangles.Length; i += 3)
        {
            var v0 = vertices[triangles[i]];
            var v1 = vertices[triangles[i + 1]];
            var v2 = vertices[triangles[i + 2]];

            if (RayTriangleIntersection(origin, direction, v0, v1, v2, out _))
                hits++;
        }

        return (hits & 1) == 1;
    }

    private static bool RayTriangleIntersection(Vector3 origin, Vector3 direction, Vector3 v0, Vector3 v1, Vector3 v2, out float distance)
    {
        const float epsilon = 1e-7f;
        var edge1 = v1 - v0;
        var edge2 = v2 - v0;
        var pvec = Vector3.Cross(direction, edge2);
        float det = Vector3.Dot(edge1, pvec);

        if (det > -epsilon && det < epsilon)
        {
            distance = 0f;
            return false;
        }

        float invDet = 1f / det;
        var tvec = origin - v0;
        float u = Vector3.Dot(tvec, pvec) * invDet;
        if (u < 0f || u > 1f)
        {
            distance = 0f;
            return false;
        }

        var qvec = Vector3.Cross(tvec, edge1);
        float v = Vector3.Dot(direction, qvec) * invDet;
        if (v < 0f || u + v > 1f)
        {
            distance = 0f;
            return false;
        }

        float t = Vector3.Dot(edge2, qvec) * invDet;
        if (t > epsilon)
        {
            distance = t;
            return true;
        }

        distance = 0f;
        return false;
    }

    private sealed class MeshGeometry
    {
        public MeshGeometry(Vector3[] vertices, int[] triangles, Bounds bounds)
        {
            Vertices = vertices;
            Triangles = triangles;
            Bounds = bounds;
        }

        public Vector3[] Vertices { get; }
        public int[] Triangles { get; }
        public Bounds Bounds { get; }
    }
}
