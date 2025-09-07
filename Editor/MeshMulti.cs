using System.Collections.Generic;
using System.Linq;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

public static class MeshMulti
{
    public static void SubdivideSelected(GameObject selected, bool smooth, int smoothIterations = 10)
    {
        if (selected == null)
        {
            Debug.LogWarning("No GameObject selected.");
            return;
        }

        var renderers = selected.GetComponentsInChildren<SkinnedMeshRenderer>();
        int total = renderers.Length;
        for (int i = 0; i < total; i++)
        {
            var renderer = renderers[i];
            EditorUtility.DisplayProgressBar("Subdivide Meshes", $"Processing {i + 1}/{total}: {renderer.name}", (float)i / total);

            var originalMesh = renderer.sharedMesh;
            if (originalMesh == null)
                continue;

            var newMesh = SubdivideMesh(originalMesh);
            if (smooth)
                ThinPlateSmooth(newMesh, smoothIterations, 0.1f, i, total);

            float percent = ((float)(i + 1) / total) * 100f;
            percent = Mathf.Floor(percent * 1000f) / 1000f;
            EditorUtility.DisplayProgressBar("Subdivide Meshes", $"Processed {i + 1}/{total}: {renderer.name} ({percent:F3}%)", (float)(i + 1) / total);
            newMesh.name = originalMesh.name + "_subdivided";

            var meshPath = AssetDatabase.GetAssetPath(originalMesh);
            if (!string.IsNullOrEmpty(meshPath))
            {
                var directory = Path.GetDirectoryName(meshPath);
                var name = Path.GetFileNameWithoutExtension(meshPath) + "_subdivided.asset";
                var newPath = AssetDatabase.GenerateUniqueAssetPath(Path.Combine(directory, name));
                AssetDatabase.CreateAsset(newMesh, newPath);
                AssetDatabase.SaveAssets();
                renderer.sharedMesh = AssetDatabase.LoadAssetAtPath<Mesh>(newPath);
            }
            else
            {
                renderer.sharedMesh = newMesh;
            }

            EditorUtility.SetDirty(renderer);
        }
        EditorUtility.ClearProgressBar();

        float finalPercent = Mathf.Floor(100f * 1000f) / 1000f;
        Debug.Log(string.Format("Subdivided {0} meshes under '{1}' ({2:F3}%).", renderers.Length, selected.name, finalPercent));
    }

    public static int PredictSubdividedVertexCount(Mesh mesh)
    {
        int[] triangles = mesh.triangles;
        var edges = new HashSet<Edge>();
        for (int i = 0; i < triangles.Length; i += 3)
        {
            int v0 = triangles[i];
            int v1 = triangles[i + 1];
            int v2 = triangles[i + 2];
            edges.Add(new Edge(v0, v1));
            edges.Add(new Edge(v1, v2));
            edges.Add(new Edge(v2, v0));
        }
        return mesh.vertexCount + edges.Count;
    }

    public static int PredictSubdividedTriangleCount(Mesh mesh)
    {
        return mesh.triangles.Length / 3 * 4;
    }

    private static void ThinPlateSmooth(Mesh mesh, int iterations = 10, float lambda = 0.1f, int meshIndex = 0, int meshTotal = 1)
    {
        var vertices = mesh.vertices;
        var originalVertices = (Vector3[])vertices.Clone();
        var triangles = mesh.triangles;
        var adjacency = new HashSet<int>[vertices.Length];
        for (int i = 0; i < adjacency.Length; i++)
            adjacency[i] = new HashSet<int>();
        for (int i = 0; i < triangles.Length; i += 3)
        {
            int a = triangles[i];
            int b = triangles[i + 1];
            int c = triangles[i + 2];
            adjacency[a].Add(b); adjacency[a].Add(c);
            adjacency[b].Add(a); adjacency[b].Add(c);
            adjacency[c].Add(a); adjacency[c].Add(b);
        }

        // Group vertices by position to account for duplicated vertices
        // along seams. Without this, smoothing each copy independently
        // can introduce visible gaps.
        var positionGroups = new Dictionary<Vector3, List<int>>();
        for (int i = 0; i < vertices.Length; i++)
        {
            List<int> list;
            if (!positionGroups.TryGetValue(vertices[i], out list))
            {
                list = new List<int>();
                positionGroups.Add(vertices[i], list);
            }
            list.Add(i);
        }
        foreach (var group in positionGroups.Values)
        {
            if (group.Count < 2) continue;
            var merged = new HashSet<int>();
            foreach (var idx in group)
                merged.UnionWith(adjacency[idx]);
            foreach (var idx in group)
                foreach (var idx2 in group)
                    if (idx != idx2) merged.Add(idx2);
            foreach (var idx in group)
            {
                adjacency[idx] = new HashSet<int>(merged);
                adjacency[idx].Remove(idx);
            }
        }

        var lap = new Vector3[vertices.Length];
        var lap2 = new Vector3[vertices.Length];
        for (int it = 0; it < iterations; it++)
        {
            EditorUtility.DisplayProgressBar(
                "Subdivide Meshes",
                $"Smoothing mesh {meshIndex + 1}/{meshTotal}, iteration {it + 1}/{iterations}",
                (meshIndex + (float)(it + 1) / iterations) / meshTotal);

            for (int i = 0; i < vertices.Length; i++)
            {
                if (adjacency[i].Count == 0) { lap[i] = Vector3.zero; continue; }
                Vector3 sum = Vector3.zero;
                foreach (var n in adjacency[i]) sum += vertices[n];
                lap[i] = sum / adjacency[i].Count - vertices[i];
            }
            for (int i = 0; i < vertices.Length; i++)
            {
                if (adjacency[i].Count == 0) { lap2[i] = Vector3.zero; continue; }
                Vector3 sum = Vector3.zero;
                foreach (var n in adjacency[i]) sum += lap[n];
                lap2[i] = sum / adjacency[i].Count - lap[i];
            }
            for (int i = 0; i < vertices.Length; i++)
                vertices[i] += lambda * lap2[i];

            // Keep duplicate-position vertices welded together
            foreach (var group in positionGroups.Values)
            {
                if (group.Count < 2) continue;
                Vector3 avg = Vector3.zero;
                foreach (var idx in group) avg += vertices[idx];
                avg /= group.Count;
                foreach (var idx in group) vertices[idx] = avg;
            }
        }

        for (int i = 0; i < vertices.Length; i++)
        {
            var v = vertices[i];
            if (float.IsNaN(v.x) || float.IsNaN(v.y) || float.IsNaN(v.z) ||
                float.IsInfinity(v.x) || float.IsInfinity(v.y) || float.IsInfinity(v.z))
            {
                vertices[i] = originalVertices[i];
            }
        }

        mesh.vertices = vertices;
        mesh.RecalculateBounds();
        mesh.RecalculateNormals();
        mesh.RecalculateTangents();
    }

    private struct Edge : System.IEquatable<Edge>
    {
        public int a;
        public int b;

        public Edge(int a, int b)
        {
            if (a < b)
            {
                this.a = a;
                this.b = b;
            }
            else
            {
                this.a = b;
                this.b = a;
            }
        }

        public bool Equals(Edge other)
        {
            return a == other.a && b == other.b;
        }

        public override bool Equals(object obj)
        {
            if (!(obj is Edge)) return false;
            return Equals((Edge)obj);
        }

        public override int GetHashCode()
        {
            return a * 397 ^ b;
        }
    }

    private class BlendShape
    {
        public string name;
        public List<BlendShapeFrame> frames = new List<BlendShapeFrame>();
    }

    private class BlendShapeFrame
    {
        public float weight;
        public List<Vector3> deltaVertices;
        public List<Vector3> deltaNormals;
        public List<Vector3> deltaTangents;
    }

    private static Mesh SubdivideMesh(Mesh mesh)
    {
        var vertices = mesh.vertices;
        int vertexCount = vertices.Length;

        Vector2[][] uvSets = new Vector2[4][];
        uvSets[0] = mesh.uv;
        uvSets[1] = mesh.uv2;
        uvSets[2] = mesh.uv3;
        uvSets[3] = mesh.uv4;

        var normals = mesh.normals;
        var tangents = mesh.tangents;
        var colors = mesh.colors;
        var boneWeights = mesh.boneWeights;
        int subMeshCount = mesh.subMeshCount;

        var blendShapes = new List<BlendShape>();
        for (int i = 0; i < mesh.blendShapeCount; i++)
        {
            var bs = new BlendShape { name = mesh.GetBlendShapeName(i) };
            int frameCount = mesh.GetBlendShapeFrameCount(i);
            for (int j = 0; j < frameCount; j++)
            {
                float weight = mesh.GetBlendShapeFrameWeight(i, j);
                var dv = new Vector3[vertexCount];
                var dn = new Vector3[vertexCount];
                var dt = new Vector3[vertexCount];
                mesh.GetBlendShapeFrameVertices(i, j, dv, dn, dt);
                bs.frames.Add(new BlendShapeFrame
                {
                    weight = weight,
                    deltaVertices = new List<Vector3>(dv),
                    deltaNormals = new List<Vector3>(dn),
                    deltaTangents = new List<Vector3>(dt)
                });
            }
            blendShapes.Add(bs);
        }

        var newVertices = new List<Vector3>(vertices);
        var newUVs = new List<Vector2>[4];
        for (int u = 0; u < 4; u++)
        {
            var set = uvSets[u];
            newUVs[u] = new List<Vector2>(set.Length == 0 ? new Vector2[vertexCount] : set);
        }
        var newNormals = new List<Vector3>(normals.Length == 0 ? new Vector3[vertexCount] : normals);
        var newTangents = new List<Vector4>(tangents.Length == 0 ? new Vector4[vertexCount] : tangents);
        var newColors = new List<Color>(colors.Length == 0 ? new Color[vertexCount] : colors);
        var newBoneWeights = new List<BoneWeight>(boneWeights.Length == 0 ? new BoneWeight[vertexCount] : boneWeights);

        var midpointCache = new Dictionary<Edge, int>();
        var newSubTriangles = new List<int>[subMeshCount];
        for (int s = 0; s < subMeshCount; s++) newSubTriangles[s] = new List<int>();

        System.Func<int, int, int> getMidpoint = null;
        getMidpoint = delegate (int i0, int i1)
        {
            Edge edge = new Edge(i0, i1);
            int index;
            if (midpointCache.TryGetValue(edge, out index))
                return index;

            Vector3 v = (vertices[i0] + vertices[i1]) * 0.5f;
            Vector2[] uv = new Vector2[4];
            for (int u = 0; u < 4; u++)
                uv[u] = uvSets[u].Length > 0 ? (uvSets[u][i0] + uvSets[u][i1]) * 0.5f : Vector2.zero;
            Vector3 normal = (normals.Length > 0) ? (normals[i0] + normals[i1]).normalized : Vector3.zero;
            Vector4 tangent = (tangents.Length > 0) ? (tangents[i0] + tangents[i1]) * 0.5f : Vector4.zero;
            Color color = (colors.Length > 0) ? (colors[i0] + colors[i1]) * 0.5f : Color.white;
            BoneWeight bw = (boneWeights.Length > 0) ? AverageBoneWeight(boneWeights[i0], boneWeights[i1]) : new BoneWeight();

            index = newVertices.Count;
            newVertices.Add(v);
            for (int u = 0; u < 4; u++)
                if (uvSets[u].Length > 0) newUVs[u].Add(uv[u]);
            if (normals.Length > 0) newNormals.Add(normal);
            if (tangents.Length > 0) newTangents.Add(tangent);
            if (colors.Length > 0) newColors.Add(color);
            if (boneWeights.Length > 0) newBoneWeights.Add(bw);

            foreach (var bs in blendShapes)
            {
                foreach (var frame in bs.frames)
                {
                    Vector3 dv = (frame.deltaVertices[i0] + frame.deltaVertices[i1]) * 0.5f;
                    Vector3 dn = (frame.deltaNormals[i0] + frame.deltaNormals[i1]) * 0.5f;
                    Vector3 dt = (frame.deltaTangents[i0] + frame.deltaTangents[i1]) * 0.5f;
                    frame.deltaVertices.Add(dv);
                    frame.deltaNormals.Add(dn);
                    frame.deltaTangents.Add(dt);
                }
            }

            midpointCache.Add(edge, index);
            return index;
        };

        for (int s = 0; s < subMeshCount; s++)
        {
            int[] triangles = mesh.GetTriangles(s);
            List<int> list = newSubTriangles[s];
            for (int i = 0; i < triangles.Length; i += 3)
            {
                int v0 = triangles[i];
                int v1 = triangles[i + 1];
                int v2 = triangles[i + 2];

                int m0 = getMidpoint(v0, v1);
                int m1 = getMidpoint(v1, v2);
                int m2 = getMidpoint(v2, v0);

                list.Add(v0); list.Add(m0); list.Add(m2);
                list.Add(v1); list.Add(m1); list.Add(m0);
                list.Add(v2); list.Add(m2); list.Add(m1);
                list.Add(m0); list.Add(m1); list.Add(m2);
            }
        }

        Mesh newMesh = new Mesh();
        if (newVertices.Count > 65535)
            newMesh.indexFormat = IndexFormat.UInt32;
        else
            newMesh.indexFormat = mesh.indexFormat;
        newMesh.subMeshCount = subMeshCount;
        newMesh.vertices = newVertices.ToArray();
        if (uvSets[0].Length > 0) newMesh.uv = newUVs[0].ToArray();
        if (uvSets[1].Length > 0) newMesh.uv2 = newUVs[1].ToArray();
        if (uvSets[2].Length > 0) newMesh.uv3 = newUVs[2].ToArray();
        if (uvSets[3].Length > 0) newMesh.uv4 = newUVs[3].ToArray();
        if (normals.Length > 0) newMesh.normals = newNormals.ToArray();
        if (tangents.Length > 0) newMesh.tangents = newTangents.ToArray();
        if (colors.Length > 0) newMesh.colors = newColors.ToArray();
        if (boneWeights.Length > 0) newMesh.boneWeights = newBoneWeights.ToArray();
        if (mesh.bindposes != null && mesh.bindposes.Length > 0) newMesh.bindposes = mesh.bindposes;

        for (int s = 0; s < subMeshCount; s++)
            newMesh.SetTriangles(newSubTriangles[s], s);

        foreach (var bs in blendShapes)
        {
            foreach (var frame in bs.frames)
            {
                newMesh.AddBlendShapeFrame(bs.name, frame.weight,
                    frame.deltaVertices.ToArray(),
                    frame.deltaNormals.ToArray(),
                    frame.deltaTangents.ToArray());
            }
        }

        newMesh.RecalculateBounds();
        if (normals.Length == 0) newMesh.RecalculateNormals();
        if (tangents.Length == 0) newMesh.RecalculateTangents();
        newMesh.name = mesh.name;
        return newMesh;
    }

    private static BoneWeight AverageBoneWeight(BoneWeight a, BoneWeight b)
    {
        Dictionary<int, float> dict = new Dictionary<int, float>();
        AddBone(ref dict, a.boneIndex0, a.weight0);
        AddBone(ref dict, a.boneIndex1, a.weight1);
        AddBone(ref dict, a.boneIndex2, a.weight2);
        AddBone(ref dict, a.boneIndex3, a.weight3);
        AddBone(ref dict, b.boneIndex0, b.weight0);
        AddBone(ref dict, b.boneIndex1, b.weight1);
        AddBone(ref dict, b.boneIndex2, b.weight2);
        AddBone(ref dict, b.boneIndex3, b.weight3);

        List<int> keys = new List<int>(dict.Keys);
        for (int i = 0; i < keys.Count; i++)
            dict[keys[i]] *= 0.5f;

        var ordered = dict.OrderByDescending(delegate (KeyValuePair<int, float> kv) { return kv.Value; }).Take(4).ToArray();
        BoneWeight result = new BoneWeight();
        for (int i = 0; i < ordered.Length; i++)
        {
            switch (i)
            {
                case 0:
                    result.boneIndex0 = ordered[i].Key;
                    result.weight0 = ordered[i].Value;
                    break;
                case 1:
                    result.boneIndex1 = ordered[i].Key;
                    result.weight1 = ordered[i].Value;
                    break;
                case 2:
                    result.boneIndex2 = ordered[i].Key;
                    result.weight2 = ordered[i].Value;
                    break;
                case 3:
                    result.boneIndex3 = ordered[i].Key;
                    result.weight3 = ordered[i].Value;
                    break;
            }
        }

        float total = result.weight0 + result.weight1 + result.weight2 + result.weight3;
        if (total > 0f)
        {
            result.weight0 /= total;
            result.weight1 /= total;
            result.weight2 /= total;
            result.weight3 /= total;
        }

        return result;
    }

    private static void AddBone(ref Dictionary<int, float> dict, int boneIndex, float weight)
    {
        if (weight == 0f) return;
        float existing;
        if (dict.TryGetValue(boneIndex, out existing))
            dict[boneIndex] = existing + weight;
        else
            dict[boneIndex] = weight;
    }
}

public class MeshMultiWindow : EditorWindow
{
    private bool smoothAppearance;
    private int smoothIterations = 10;

    [MenuItem("yussy/Subdivide Skinned Meshes")]
    private static void ShowWindow()
    {
        GetWindow<MeshMultiWindow>("Subdivide Meshes");
    }

    private void OnGUI()
    {
        var selected = Selection.activeGameObject;
        if (selected == null)
        {
            EditorGUILayout.HelpBox("No GameObject selected.", MessageType.Info);
            return;
        }

        var renderers = selected.GetComponentsInChildren<SkinnedMeshRenderer>();
        foreach (var renderer in renderers)
        {
            var mesh = renderer.sharedMesh;
            if (mesh == null) continue;
            int predictedVertices = MeshMulti.PredictSubdividedVertexCount(mesh);
            int predictedTriangles = MeshMulti.PredictSubdividedTriangleCount(mesh);
            EditorGUILayout.LabelField(
                mesh.name,
                string.Format(
                    "Vertices: {0} → {1}, Triangles: {2} → {3}",
                    mesh.vertexCount,
                    predictedVertices,
                    mesh.triangles.Length / 3,
                    predictedTriangles));
        }

        smoothAppearance = EditorGUILayout.Toggle("見た目をなめらかにする(実験的)", smoothAppearance);
        if (smoothAppearance)
            smoothIterations = EditorGUILayout.IntSlider("スムージング強度", smoothIterations, 1, 50);

        if (GUILayout.Button("Subdivide"))
        {
            MeshMulti.SubdivideSelected(selected, smoothAppearance, smoothIterations);
        }
    }
}
