using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

public static class MeshPolygonReducer
{
    public static void ReduceSelected(GameObject selected, float reductionPercent, Bounds? limitBounds = null, int? seed = null)
    {
        if (selected == null)
        {
            Debug.LogWarning("No GameObject selected.");
            return;
        }

        reductionPercent = Mathf.Clamp(reductionPercent, 0f, 100f);
        float reductionRatio = reductionPercent / 100f;
        if (reductionRatio <= 0f)
        {
            Debug.Log("Reduction ratio is zero. Nothing to do.");
            return;
        }

        var renderers = selected.GetComponentsInChildren<SkinnedMeshRenderer>();
        var targetRenderers = new List<SkinnedMeshRenderer>();
        if (limitBounds.HasValue)
        {
            Bounds bounds = limitBounds.Value;
            for (int i = 0; i < renderers.Length; i++)
            {
                var renderer = renderers[i];
                var mesh = renderer.sharedMesh;
                if (mesh == null)
                    continue;

                var insideMask = MeshMulti.CalculateVerticesInsideBounds(renderer, mesh, bounds);
                bool anyInside = false;
                for (int v = 0; v < insideMask.Length; v++)
                {
                    if (insideMask[v])
                    {
                        anyInside = true;
                        break;
                    }
                }

                if (anyInside)
                    targetRenderers.Add(renderer);
            }

            if (targetRenderers.Count == 0)
            {
                Debug.LogWarning("No skinned mesh vertices found within the specified bounds.");
                return;
            }
        }
        else
        {
            for (int i = 0; i < renderers.Length; i++)
            {
                var mesh = renderers[i].sharedMesh;
                if (mesh == null)
                    continue;
                targetRenderers.Add(renderers[i]);
            }
        }

        int total = targetRenderers.Count;
        for (int i = 0; i < total; i++)
        {
            var renderer = targetRenderers[i];
            EditorUtility.DisplayProgressBar("Reduce Mesh Polygons", $"Processing {i + 1}/{total}: {renderer.name}", (float)i / total);

            var originalMesh = renderer.sharedMesh;
            if (originalMesh == null)
                continue;

            bool[] vertexMask = null;
            if (limitBounds.HasValue)
                vertexMask = MeshMulti.CalculateVerticesInsideBounds(renderer, originalMesh, limitBounds.Value);

            var newMesh = ReduceMesh(originalMesh, reductionRatio, vertexMask, seed);
            if (newMesh == null)
                continue;

            float percent = ((float)(i + 1) / total) * 100f;
            percent = Mathf.Floor(percent * 1000f) / 1000f;
            EditorUtility.DisplayProgressBar("Reduce Mesh Polygons", $"Processed {i + 1}/{total}: {renderer.name} ({percent:F3}%)", (float)(i + 1) / total);
            newMesh.name = originalMesh.name + "_reduced";

            var meshPath = AssetDatabase.GetAssetPath(originalMesh);
            if (!string.IsNullOrEmpty(meshPath))
            {
                var directory = Path.GetDirectoryName(meshPath);
                var name = Path.GetFileNameWithoutExtension(meshPath) + "_reduced.asset";
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
        Debug.Log(string.Format("Reduced polygons for {0} meshes under '{1}' ({2:F3}%).", targetRenderers.Count, selected.name, finalPercent));
    }

    private static Mesh ReduceMesh(Mesh mesh, float reductionRatio, bool[] vertexMask, int? seed)
    {
        if (mesh == null)
            return null;

        int vertexCount = mesh.vertexCount;
        if (vertexCount < 3)
            return null;

        bool useMask = vertexMask != null && vertexMask.Length == vertexCount;
        int candidateVertices = 0;
        if (useMask)
        {
            for (int i = 0; i < vertexMask.Length; i++)
            {
                if (vertexMask[i])
                    candidateVertices++;
            }
            if (candidateVertices < 3)
                return null;
        }
        else
        {
            candidateVertices = vertexCount;
        }

        reductionRatio = Mathf.Clamp01(reductionRatio);
        if (reductionRatio <= 0f)
            return null;

        int desiredCandidateVertices = Mathf.Max(3, candidateVertices - Mathf.RoundToInt(candidateVertices * reductionRatio));
        if (desiredCandidateVertices >= candidateVertices)
            return null;

        var simplifier = new ArapMeshSimplifier(mesh, vertexMask);
        var simplifierResult = simplifier.Simplify(desiredCandidateVertices);
        if (!simplifierResult.HasValue)
            return null;

        if (!simplifierResult.Value.RemovedTriangles)
            return null;

        return simplifierResult.Value.Mesh;
    }

    private class ArapMeshSimplifier
    {
        private const float LengthTolerance = 1e-6f;
        private const float RelaxationStep = 0.35f;
        private const int RelaxationIterations = 6;

        private readonly Mesh sourceMesh;
        private readonly bool[] vertexMask;

        private readonly Vector3[] sourceVertices;
        private readonly Vector3[] sourceNormals;
        private readonly Vector4[] sourceTangents;
        private readonly Color[] sourceColors;
        private readonly BoneWeight[] sourceBoneWeights;
        private readonly Vector2[][] sourceUVs;

        private VertexData[] vertices;
        private List<Triangle>[] submeshTriangles;
        private Dictionary<EdgeKey, float> restLengths;
        private int activeVertexCount;

        public struct Result
        {
            public Mesh Mesh;
            public bool RemovedTriangles;
        }

        private class VertexData
        {
            public Vector3 Position;
            public Vector3 RestPosition;
            public Vector3 NormalSum;
            public Vector4 TangentSum;
            public Color ColorSum;
            public Vector2[] UVSum;
            public bool[] UVHasData;
            public Dictionary<int, float> BoneWeights;
            public HashSet<int> Neighbors = new HashSet<int>();
            public int Count;
            public bool Locked;
            public bool Removed;
            public List<int> SourceIndices = new List<int>();
        }

        private struct Triangle
        {
            public int A;
            public int B;
            public int C;
            public bool Removed;
        }

        private readonly struct EdgeKey : IEquatable<EdgeKey>
        {
            public readonly int A;
            public readonly int B;

            public EdgeKey(int a, int b)
            {
                if (a < b)
                {
                    A = a;
                    B = b;
                }
                else
                {
                    A = b;
                    B = a;
                }
            }

            public bool Equals(EdgeKey other)
            {
                return A == other.A && B == other.B;
            }

            public override bool Equals(object obj)
            {
                return obj is EdgeKey other && Equals(other);
            }

            public override int GetHashCode()
            {
                unchecked
                {
                    return (A * 397) ^ B;
                }
            }
        }

        public ArapMeshSimplifier(Mesh mesh, bool[] mask)
        {
            sourceMesh = mesh;
            vertexMask = mask;

            sourceVertices = mesh.vertices;
            sourceNormals = mesh.normals;
            sourceTangents = mesh.tangents;
            sourceColors = mesh.colors;
            sourceBoneWeights = mesh.boneWeights;
            sourceUVs = GetUVChannels(mesh);
        }

        public Result? Simplify(int targetVertexCount)
        {
            InitializeData();

            if (targetVertexCount < 3)
                targetVertexCount = 3;

            if (targetVertexCount >= activeVertexCount)
                return null;

            bool removedAnyTriangle = false;

            while (activeVertexCount > targetVertexCount)
            {
                var bestEdge = FindBestEdge();
                if (!bestEdge.HasValue)
                    break;

                if (CollapseEdge(bestEdge.Value.A, bestEdge.Value.B))
                {
                    activeVertexCount--;
                    removedAnyTriangle = true;
                }
                else
                {
                    // Unable to collapse this edge safely. Remove it from consideration.
                    RemoveEdge(bestEdge.Value.A, bestEdge.Value.B);
                    continue;
                }
            }

            if (!removedAnyTriangle)
                return null;

            RunArapRelaxation();

            var mesh = BuildMesh(out bool removedTrianglesDuringBuild);
            removedAnyTriangle |= removedTrianglesDuringBuild;

            return new Result
            {
                Mesh = mesh,
                RemovedTriangles = removedAnyTriangle
            };
        }

        private void InitializeData()
        {
            int vertexCount = sourceVertices.Length;
            vertices = new VertexData[vertexCount];
            activeVertexCount = vertexCount;

            for (int i = 0; i < vertexCount; i++)
            {
                vertices[i] = CreateVertexData(i);
            }

            int subMeshCount = Mathf.Max(1, sourceMesh.subMeshCount);
            submeshTriangles = new List<Triangle>[subMeshCount];
            for (int sub = 0; sub < subMeshCount; sub++)
            {
                submeshTriangles[sub] = new List<Triangle>();
                int[] tris = sourceMesh.subMeshCount == 0 && sub == 0 ? sourceMesh.triangles : sourceMesh.GetTriangles(sub);
                for (int i = 0; i < tris.Length; i += 3)
                {
                    var tri = new Triangle
                    {
                        A = tris[i],
                        B = tris[i + 1],
                        C = tris[i + 2],
                        Removed = false
                    };
                    submeshTriangles[sub].Add(tri);
                    AddNeighbor(tri.A, tri.B);
                    AddNeighbor(tri.B, tri.C);
                    AddNeighbor(tri.C, tri.A);
                }
            }

            restLengths = new Dictionary<EdgeKey, float>();
            for (int i = 0; i < vertexCount; i++)
            {
                foreach (var neighbor in vertices[i].Neighbors)
                {
                    EdgeKey key = new EdgeKey(i, neighbor);
                    if (!restLengths.ContainsKey(key))
                    {
                        float length = (sourceVertices[key.A] - sourceVertices[key.B]).magnitude;
                        restLengths.Add(key, Mathf.Max(length, LengthTolerance));
                    }
                }
            }
        }

        private VertexData CreateVertexData(int index)
        {
            var data = new VertexData
            {
                Position = sourceVertices[index],
                RestPosition = sourceVertices[index],
                Count = 1,
                Locked = vertexMask != null && vertexMask.Length == sourceVertices.Length && !vertexMask[index],
                UVSum = new Vector2[sourceUVs.Length],
                UVHasData = new bool[sourceUVs.Length]
            };

            data.SourceIndices.Add(index);

            if (sourceNormals != null && sourceNormals.Length == sourceVertices.Length)
                data.NormalSum = sourceNormals[index];
            if (sourceTangents != null && sourceTangents.Length == sourceVertices.Length)
                data.TangentSum = sourceTangents[index];
            if (sourceColors != null && sourceColors.Length == sourceVertices.Length)
                data.ColorSum = sourceColors[index];

            for (int channel = 0; channel < sourceUVs.Length; channel++)
            {
                var uvChannel = sourceUVs[channel];
                if (uvChannel != null && uvChannel.Length == sourceVertices.Length)
                {
                    data.UVSum[channel] = uvChannel[index];
                    data.UVHasData[channel] = true;
                }
            }

            if (sourceBoneWeights != null && sourceBoneWeights.Length == sourceVertices.Length)
            {
                data.BoneWeights = new Dictionary<int, float>(4);
                AddBoneWeight(data.BoneWeights, sourceBoneWeights[index].boneIndex0, sourceBoneWeights[index].weight0);
                AddBoneWeight(data.BoneWeights, sourceBoneWeights[index].boneIndex1, sourceBoneWeights[index].weight1);
                AddBoneWeight(data.BoneWeights, sourceBoneWeights[index].boneIndex2, sourceBoneWeights[index].weight2);
                AddBoneWeight(data.BoneWeights, sourceBoneWeights[index].boneIndex3, sourceBoneWeights[index].weight3);
            }

            return data;
        }

        private void AddNeighbor(int a, int b)
        {
            if (a == b)
                return;
            vertices[a].Neighbors.Add(b);
            vertices[b].Neighbors.Add(a);
        }

        private static void AddBoneWeight(Dictionary<int, float> dict, int index, float weight)
        {
            if (weight <= 0f)
                return;
            if (!dict.ContainsKey(index))
                dict[index] = weight;
            else
                dict[index] += weight;
        }

        private EdgeKey? FindBestEdge()
        {
            float bestCost = float.MaxValue;
            EdgeKey? bestEdge = null;

            for (int i = 0; i < vertices.Length; i++)
            {
                if (vertices[i] == null || vertices[i].Removed)
                    continue;

                foreach (var neighbor in vertices[i].Neighbors)
                {
                    if (neighbor <= i)
                        continue;
                    if (vertices[neighbor] == null || vertices[neighbor].Removed)
                        continue;

                    float cost = ComputeEdgeCost(i, neighbor);
                    if (cost < bestCost)
                    {
                        bestCost = cost;
                        bestEdge = new EdgeKey(i, neighbor);
                    }
                }
            }

            return bestEdge;
        }

        private float ComputeEdgeCost(int a, int b)
        {
            var va = vertices[a];
            var vb = vertices[b];
            if (va.Locked && vb.Locked)
                return float.MaxValue;

            HashSet<int> union = new HashSet<int>(va.Neighbors);
            union.UnionWith(vb.Neighbors);
            union.Remove(a);
            union.Remove(b);

            if (union.Count == 0)
                return float.MaxValue;

            Vector3 merged = EstimateMergedPosition(a, b);
            float cost = 0f;
            foreach (int n in union)
            {
                float rest = GetCombinedRestLength(a, b, n);
                Vector3 dir = vertices[n].Position - merged;
                float length = dir.magnitude;
                if (rest <= 0f)
                    rest = Mathf.Max(length, LengthTolerance);
                float diff = length - rest;
                cost += diff * diff;
            }

            cost /= union.Count;
            return cost;
        }

        private Vector3 EstimateMergedPosition(int a, int b)
        {
            var va = vertices[a];
            var vb = vertices[b];
            Vector3 merged = (va.Position * va.Count + vb.Position * vb.Count) / Mathf.Max(1, va.Count + vb.Count);

            HashSet<int> union = new HashSet<int>(va.Neighbors);
            union.UnionWith(vb.Neighbors);
            union.Remove(a);
            union.Remove(b);

            Vector3 correction = Vector3.zero;
            float weight = 0f;
            foreach (int n in union)
            {
                float rest = GetCombinedRestLength(a, b, n);
                Vector3 neighborPos = vertices[n].Position;
                Vector3 dir = neighborPos - merged;
                float len = dir.magnitude;
                if (len <= LengthTolerance || rest <= 0f)
                    continue;
                float diff = rest - len;
                correction += dir.normalized * diff;
                weight += 1f;
            }

            if (weight > 0f)
                merged += correction / weight * 0.5f;

            return merged;
        }

        private float GetCombinedRestLength(int a, int b, int neighbor)
        {
            float restA = GetRestLength(a, neighbor);
            float restB = GetRestLength(b, neighbor);
            var va = vertices[a];
            var vb = vertices[b];

            if (restA > 0f && restB > 0f)
                return (restA * va.Count + restB * vb.Count) / Mathf.Max(1, va.Count + vb.Count);
            if (restA > 0f)
                return restA;
            if (restB > 0f)
                return restB;
            return 0f;
        }

        private float GetRestLength(int a, int b)
        {
            EdgeKey key = new EdgeKey(a, b);
            if (restLengths.TryGetValue(key, out float length))
                return length;
            return 0f;
        }

        private bool CollapseEdge(int a, int b)
        {
            var va = vertices[a];
            var vb = vertices[b];
            if (va == null || vb == null || va.Removed || vb.Removed)
                return false;
            if (va.Locked && vb.Locked)
                return false;

            int target = va.Locked ? a : (vb.Locked ? b : a);
            int source = target == a ? b : a;

            if (target != a)
            {
                int temp = a;
                a = b;
                b = temp;
                va = vertices[a];
                vb = vertices[b];
            }

            Vector3 mergedPosition = EstimateMergedPosition(a, b);

            MergeVertexData(a, b, mergedPosition);
            UpdateTrianglesAfterMerge(a, b);
            UpdateNeighborsAfterMerge(a, b);
            UpdateRestLengthsAfterMerge(a, b);

            vb.Removed = true;
            vb.Neighbors.Clear();

            return true;
        }

        private void MergeVertexData(int target, int source, Vector3 newPosition)
        {
            var vt = vertices[target];
            var vs = vertices[source];

            int combinedCount = Mathf.Max(1, vt.Count + vs.Count);
            vt.Position = newPosition;
            vt.RestPosition = (vt.RestPosition * vt.Count + vs.RestPosition * vs.Count) / combinedCount;
            vt.Count = combinedCount;
            vt.Locked = vt.Locked || vs.Locked;

            vt.NormalSum += vs.NormalSum;
            vt.TangentSum += vs.TangentSum;
            vt.ColorSum += vs.ColorSum;
            vt.SourceIndices.AddRange(vs.SourceIndices);

            for (int channel = 0; channel < vt.UVSum.Length; channel++)
            {
                if (vs.UVHasData[channel])
                {
                    vt.UVSum[channel] += vs.UVSum[channel];
                    vt.UVHasData[channel] = true;
                }
            }

            if (vs.BoneWeights != null)
            {
                if (vt.BoneWeights == null)
                    vt.BoneWeights = new Dictionary<int, float>(vs.BoneWeights.Count);
                foreach (var kv in vs.BoneWeights)
                {
                    if (vt.BoneWeights.ContainsKey(kv.Key))
                        vt.BoneWeights[kv.Key] += kv.Value;
                    else
                        vt.BoneWeights.Add(kv.Key, kv.Value);
                }
            }
        }

        private void UpdateTrianglesAfterMerge(int target, int source)
        {
            for (int sub = 0; sub < submeshTriangles.Length; sub++)
            {
                var list = submeshTriangles[sub];
                for (int i = 0; i < list.Count; i++)
                {
                    var tri = list[i];
                    if (tri.Removed)
                        continue;

                    bool changed = false;
                    if (tri.A == source)
                    {
                        tri.A = target;
                        changed = true;
                    }
                    if (tri.B == source)
                    {
                        tri.B = target;
                        changed = true;
                    }
                    if (tri.C == source)
                    {
                        tri.C = target;
                        changed = true;
                    }

                    if (tri.A == tri.B || tri.B == tri.C || tri.C == tri.A)
                    {
                        tri.Removed = true;
                        list[i] = tri;
                        continue;
                    }

                    if (changed)
                        list[i] = tri;
                }
            }
        }

        private void UpdateNeighborsAfterMerge(int target, int source)
        {
            var vt = vertices[target];
            var vs = vertices[source];

            vt.Neighbors.Remove(target);
            vt.Neighbors.Remove(source);

            foreach (var neighbor in vs.Neighbors)
            {
                if (neighbor == target)
                    continue;

                vt.Neighbors.Add(neighbor);
                var vn = vertices[neighbor];
                vn.Neighbors.Remove(source);
                vn.Neighbors.Add(target);
            }
        }

        private void UpdateRestLengthsAfterMerge(int target, int source)
        {
            var vt = vertices[target];
            var vs = vertices[source];

            foreach (var neighbor in vs.Neighbors)
            {
                EdgeKey keySource = new EdgeKey(source, neighbor);
                if (!restLengths.TryGetValue(keySource, out float sourceRest))
                    sourceRest = (vertices[source].RestPosition - vertices[neighbor].RestPosition).magnitude;

                EdgeKey keyTarget = new EdgeKey(target, neighbor);
                float targetRest = 0f;
                restLengths.TryGetValue(keyTarget, out targetRest);

                float combined = sourceRest;
                if (targetRest > 0f)
                {
                    combined = (targetRest * vt.Count + sourceRest * vs.Count) / Mathf.Max(1, vt.Count + vs.Count);
                }

                restLengths[keyTarget] = Mathf.Max(combined, LengthTolerance);
                restLengths.Remove(keySource);
            }

            EdgeKey collapsed = new EdgeKey(target, source);
            restLengths.Remove(collapsed);
        }

        private void RemoveEdge(int a, int b)
        {
            vertices[a].Neighbors.Remove(b);
            vertices[b].Neighbors.Remove(a);
            EdgeKey key = new EdgeKey(a, b);
            restLengths.Remove(key);
        }

        private void RunArapRelaxation()
        {
            for (int iteration = 0; iteration < RelaxationIterations; iteration++)
            {
                Vector3[] updated = new Vector3[vertices.Length];
                for (int i = 0; i < vertices.Length; i++)
                {
                    var v = vertices[i];
                    if (v == null || v.Removed)
                        continue;
                    if (v.Locked)
                    {
                        updated[i] = v.Position;
                        continue;
                    }

                    Vector3 delta = Vector3.zero;
                    float totalWeight = 0f;
                    foreach (var neighbor in v.Neighbors)
                    {
                        var n = vertices[neighbor];
                        if (n == null || n.Removed)
                            continue;

                        float rest = GetRestLength(i, neighbor);
                        Vector3 dir = v.Position - n.Position;
                        float length = dir.magnitude;
                        if (length <= LengthTolerance || rest <= 0f)
                            continue;

                        float factor = 1f - (rest / length);
                        delta += dir * factor;
                        totalWeight += 1f;
                    }

                    if (totalWeight > 0f)
                        updated[i] = v.Position - (delta / totalWeight) * RelaxationStep;
                    else
                        updated[i] = v.Position;
                }

                for (int i = 0; i < vertices.Length; i++)
                {
                    if (vertices[i] == null || vertices[i].Removed || vertices[i].Locked)
                        continue;
                    vertices[i].Position = updated[i];
                }
            }
        }

        private Mesh BuildMesh(out bool removedAnyTriangle)
        {
            removedAnyTriangle = false;
            var mesh = new Mesh();
            mesh.name = sourceMesh.name;

            int vertexCount = 0;
            int[] map = new int[vertices.Length];
            for (int i = 0; i < vertices.Length; i++)
            {
                var v = vertices[i];
                if (v == null || v.Removed)
                {
                    map[i] = -1;
                    continue;
                }
                map[i] = vertexCount++;
            }

            var newVertices = new Vector3[vertexCount];
            Vector3[] newNormals = (sourceNormals != null && sourceNormals.Length == sourceVertices.Length) ? new Vector3[vertexCount] : null;
            Vector4[] newTangents = (sourceTangents != null && sourceTangents.Length == sourceVertices.Length) ? new Vector4[vertexCount] : null;
            Color[] newColors = (sourceColors != null && sourceColors.Length == sourceVertices.Length) ? new Color[vertexCount] : null;
            BoneWeight[] newBoneWeights = (sourceBoneWeights != null && sourceBoneWeights.Length == sourceVertices.Length) ? new BoneWeight[vertexCount] : null;
            List<int>[] vertexSources = new List<int>[vertexCount];

            List<Vector2>[] uvChannels = new List<Vector2>[sourceUVs.Length];
            for (int channel = 0; channel < sourceUVs.Length; channel++)
            {
                if (sourceUVs[channel] != null && sourceUVs[channel].Length == sourceVertices.Length)
                    uvChannels[channel] = new List<Vector2>(vertexCount);
            }

            for (int i = 0; i < vertices.Length; i++)
            {
                var v = vertices[i];
                if (v == null || v.Removed)
                    continue;
                int mapped = map[i];

                newVertices[mapped] = v.Position;

                if (newNormals != null)
                {
                    Vector3 normal = v.NormalSum / Mathf.Max(1, v.Count);
                    newNormals[mapped] = normal.sqrMagnitude > 0f ? normal.normalized : normal;
                }

                if (newTangents != null)
                {
                    Vector4 tangent = v.TangentSum / Mathf.Max(1, v.Count);
                    Vector3 xyz = new Vector3(tangent.x, tangent.y, tangent.z);
                    if (xyz.sqrMagnitude > 0f)
                        xyz.Normalize();
                    float w = tangent.w >= 0f ? 1f : -1f;
                    newTangents[mapped] = new Vector4(xyz.x, xyz.y, xyz.z, w);
                }

                if (newColors != null)
                {
                    newColors[mapped] = v.ColorSum / Mathf.Max(1, v.Count);
                }

                if (newBoneWeights != null)
                {
                    newBoneWeights[mapped] = BuildBoneWeight(v.BoneWeights);
                }

                vertexSources[mapped] = v.SourceIndices;

                for (int channel = 0; channel < uvChannels.Length; channel++)
                {
                    if (uvChannels[channel] == null)
                        continue;
                    Vector2 uv = Vector2.zero;
                    if (v.UVHasData[channel])
                        uv = v.UVSum[channel] / Mathf.Max(1, v.Count);
                    uvChannels[channel].Add(uv);
                }
            }

            mesh.indexFormat = vertexCount > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
            mesh.vertices = newVertices;
            if (newNormals != null)
                mesh.normals = newNormals;
            if (newTangents != null)
                mesh.tangents = newTangents;
            if (newColors != null)
                mesh.colors = newColors;
            if (newBoneWeights != null)
                mesh.boneWeights = newBoneWeights;
            if (sourceMesh.bindposes != null && sourceMesh.bindposes.Length > 0)
                mesh.bindposes = sourceMesh.bindposes;

            for (int channel = 0; channel < uvChannels.Length; channel++)
            {
                if (uvChannels[channel] != null)
                    mesh.SetUVs(channel, uvChannels[channel]);
            }

            CopyBlendShapesToMesh(mesh, vertexSources);

            bool anyTriangles = false;
            mesh.subMeshCount = submeshTriangles.Length;
            for (int sub = 0; sub < submeshTriangles.Length; sub++)
            {
                List<int> indices = new List<int>();
                foreach (var tri in submeshTriangles[sub])
                {
                    if (tri.Removed)
                    {
                        removedAnyTriangle = true;
                        continue;
                    }
                    int a = map[tri.A];
                    int b = map[tri.B];
                    int c = map[tri.C];
                    if (a < 0 || b < 0 || c < 0)
                    {
                        removedAnyTriangle = true;
                        continue;
                    }
                    if (a == b || b == c || c == a)
                    {
                        removedAnyTriangle = true;
                        continue;
                    }
                    indices.Add(a);
                    indices.Add(b);
                    indices.Add(c);
                    anyTriangles = true;
                }

                mesh.SetTriangles(indices, sub, true);
            }

            mesh.RecalculateBounds();
            if (mesh.normals == null || mesh.normals.Length == 0)
                mesh.RecalculateNormals();
            if (mesh.tangents == null || mesh.tangents.Length == 0)
                mesh.RecalculateTangents();

            if (!anyTriangles)
                removedAnyTriangle = true;

            return mesh;
        }

        private void CopyBlendShapesToMesh(Mesh mesh, List<int>[] vertexSources)
        {
            int blendShapeCount = sourceMesh.blendShapeCount;
            if (blendShapeCount == 0)
                return;

            var deltaVertices = new Vector3[sourceVertices.Length];
            var deltaNormals = new Vector3[sourceVertices.Length];
            var deltaTangents = new Vector3[sourceVertices.Length];

            var newDeltaVertices = new Vector3[mesh.vertexCount];
            var newDeltaNormals = new Vector3[mesh.vertexCount];
            var newDeltaTangents = new Vector3[mesh.vertexCount];

            for (int shapeIndex = 0; shapeIndex < blendShapeCount; shapeIndex++)
            {
                string shapeName = sourceMesh.GetBlendShapeName(shapeIndex);
                int frameCount = sourceMesh.GetBlendShapeFrameCount(shapeIndex);
                for (int frameIndex = 0; frameIndex < frameCount; frameIndex++)
                {
                    float weight = sourceMesh.GetBlendShapeFrameWeight(shapeIndex, frameIndex);
                    sourceMesh.GetBlendShapeFrameVertices(shapeIndex, frameIndex, deltaVertices, deltaNormals, deltaTangents);

                    Array.Clear(newDeltaVertices, 0, newDeltaVertices.Length);
                    Array.Clear(newDeltaNormals, 0, newDeltaNormals.Length);
                    Array.Clear(newDeltaTangents, 0, newDeltaTangents.Length);

                    for (int v = 0; v < mesh.vertexCount; v++)
                    {
                        var sources = vertexSources[v];
                        if (sources == null || sources.Count == 0)
                            continue;

                        Vector3 accumDelta = Vector3.zero;
                        Vector3 accumNormal = Vector3.zero;
                        Vector3 accumTangent = Vector3.zero;
                        foreach (int src in sources)
                        {
                            accumDelta += deltaVertices[src];
                            accumNormal += deltaNormals[src];
                            accumTangent += deltaTangents[src];
                        }

                        float inv = 1f / Mathf.Max(1, sources.Count);
                        newDeltaVertices[v] = accumDelta * inv;
                        newDeltaNormals[v] = accumNormal * inv;
                        newDeltaTangents[v] = accumTangent * inv;
                    }

                    mesh.AddBlendShapeFrame(shapeName, weight, newDeltaVertices, newDeltaNormals, newDeltaTangents);
                }
            }
        }

        private static BoneWeight BuildBoneWeight(Dictionary<int, float> weights)
        {
            if (weights == null || weights.Count == 0)
                return default;

            var ordered = new List<KeyValuePair<int, float>>(weights);
            ordered.Sort((a, b) => b.Value.CompareTo(a.Value));

            BoneWeight boneWeight = new BoneWeight();
            float total = 0f;
            int influenceCount = Mathf.Min(4, ordered.Count);
            for (int i = 0; i < influenceCount; i++)
                total += ordered[i].Value;
            if (total <= 0f)
                total = 1f;

            for (int i = 0; i < influenceCount; i++)
            {
                float value = ordered[i].Value / total;
                switch (i)
                {
                    case 0:
                        boneWeight.boneIndex0 = ordered[i].Key;
                        boneWeight.weight0 = value;
                        break;
                    case 1:
                        boneWeight.boneIndex1 = ordered[i].Key;
                        boneWeight.weight1 = value;
                        break;
                    case 2:
                        boneWeight.boneIndex2 = ordered[i].Key;
                        boneWeight.weight2 = value;
                        break;
                    case 3:
                        boneWeight.boneIndex3 = ordered[i].Key;
                        boneWeight.weight3 = value;
                        break;
                }
            }

            return boneWeight;
        }
    }

    private static Bounds CalculateBounds(Vector3[] vertices, bool[] vertexMask)
    {
        if (vertices == null || vertices.Length == 0)
            return new Bounds(Vector3.zero, Vector3.one);

        bool useMask = vertexMask != null && vertexMask.Length == vertices.Length;
        bool initialized = false;
        Vector3 min = Vector3.zero;
        Vector3 max = Vector3.zero;
        for (int i = 0; i < vertices.Length; i++)
        {
            if (useMask && !vertexMask[i])
                continue;
            if (!initialized)
            {
                min = vertices[i];
                max = vertices[i];
                initialized = true;
            }
            else
            {
                min = Vector3.Min(min, vertices[i]);
                max = Vector3.Max(max, vertices[i]);
            }
        }

        if (!initialized)
            return new Bounds(Vector3.zero, Vector3.one);

        Bounds bounds = new Bounds();
        bounds.SetMinMax(min, max);
        return bounds;
    }

    private static Vector2[][] GetUVChannels(Mesh mesh)
    {
        Vector2[][] channels = new Vector2[8][];
        channels[0] = mesh.uv;
        channels[1] = mesh.uv2;
        channels[2] = mesh.uv3;
        channels[3] = mesh.uv4;
#if UNITY_2018_2_OR_NEWER
        channels[4] = mesh.uv5;
        channels[5] = mesh.uv6;
        channels[6] = mesh.uv7;
        channels[7] = mesh.uv8;
#endif
        return channels;
    }
}

public class MeshPolygonReducerWindow : EditorWindow
{
    private bool restrictToBounds;
    private Vector3 boundsCenter;
    private Vector3 boundsSize = Vector3.one;
    private float reductionPercent = 50f;
    private static readonly Color BoundsFillColor = new Color(0f, 0.6f, 1f, 0.15f);
    private static readonly Color BoundsOutlineColor = new Color(0f, 0.6f, 1f, 0.6f);

    [MenuItem("yussy/Reduce Skinned Mesh Polygons")]
    private static void ShowWindow()
    {
        GetWindow<MeshPolygonReducerWindow>("Reduce Polygons");
    }

    private void OnEnable()
    {
        SceneView.duringSceneGui += OnSceneGUI;
        SyncBoundsWithSelection();
        Selection.selectionChanged += OnSelectionChanged;
    }

    private void OnDisable()
    {
        SceneView.duringSceneGui -= OnSceneGUI;
        Selection.selectionChanged -= OnSelectionChanged;
    }

    private void OnGUI()
    {
        var selected = Selection.activeGameObject;
        if (selected == null)
        {
            EditorGUILayout.HelpBox("No GameObject selected.", MessageType.Info);
            return;
        }

        EditorGUI.BeginChangeCheck();
        restrictToBounds = EditorGUILayout.Toggle("座標範囲を指定する", restrictToBounds);
        if (restrictToBounds)
        {
            boundsCenter = EditorGUILayout.Vector3Field("範囲中心", boundsCenter);
            Vector3 sizeInput = EditorGUILayout.Vector3Field("範囲サイズ", boundsSize);
            boundsSize = new Vector3(Mathf.Max(0f, Mathf.Abs(sizeInput.x)), Mathf.Max(0f, Mathf.Abs(sizeInput.y)), Mathf.Max(0f, Mathf.Abs(sizeInput.z)));
            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("選択から範囲を取得"))
                {
                    var calculated = CalculateSelectionBounds(selected);
                    boundsCenter = calculated.center;
                    boundsSize = calculated.size;
                }
                if (GUILayout.Button("範囲をリセット"))
                {
                    boundsCenter = Vector3.zero;
                    boundsSize = Vector3.one;
                }
            }
        }
        if (EditorGUI.EndChangeCheck())
        {
            SceneView.RepaintAll();
        }

        reductionPercent = EditorGUILayout.Slider("削減率(%)", reductionPercent, 0f, 100f);

        var renderers = selected.GetComponentsInChildren<SkinnedMeshRenderer>();
        Bounds activeBounds = new Bounds(boundsCenter, boundsSize);
        bool boundsValid = boundsSize.x > 0f && boundsSize.y > 0f && boundsSize.z > 0f;
        int includedRenderers = 0;
        foreach (var renderer in renderers)
        {
            var mesh = renderer.sharedMesh;
            if (mesh == null) continue;

            bool hasVerticesInBounds = false;
            bool[] mask = null;
            if (restrictToBounds && boundsValid)
            {
                mask = MeshMulti.CalculateVerticesInsideBounds(renderer, mesh, activeBounds);
                for (int v = 0; v < mask.Length; v++)
                {
                    if (mask[v])
                    {
                        hasVerticesInBounds = true;
                        break;
                    }
                }
            }

            bool inRange = !restrictToBounds || (boundsValid && hasVerticesInBounds);
            if (inRange) includedRenderers++;
            int totalTriangles = mesh.triangles.Length / 3;
            int affectedTriangles = totalTriangles;
            if (restrictToBounds && boundsValid && mask != null)
            {
                affectedTriangles = 0;
                var triangles = mesh.triangles;
                for (int i = 0; i < triangles.Length; i += 3)
                {
                    if (mask[triangles[i]] && mask[triangles[i + 1]] && mask[triangles[i + 2]])
                        affectedTriangles++;
                }
            }
            int predictedRemoval = Mathf.RoundToInt(affectedTriangles * (Mathf.Clamp01(reductionPercent / 100f)));
            int predictedTriangles = Mathf.Max(0, totalTriangles - predictedRemoval);
            string label = string.Format(
                "Triangles: {0} → {1} (削減 {2})",
                totalTriangles,
                predictedTriangles,
                predictedRemoval);
            if (!inRange && restrictToBounds)
            {
                label += "（範囲外）";
            }
            else if (restrictToBounds && boundsValid)
            {
                label += string.Format("（範囲内の三角形 {0} 個）", affectedTriangles);
            }
            EditorGUILayout.LabelField(mesh.name, label);
        }

        if (restrictToBounds)
        {
            if (!boundsValid)
            {
                EditorGUILayout.HelpBox("範囲サイズがゼロの軸があるため、現在の設定ではメッシュが選択されません。", MessageType.Warning);
            }
            else
            {
                EditorGUILayout.HelpBox(string.Format("範囲内に頂点を持つメッシュ {0}/{1} 個が対象になります。", includedRenderers, renderers.Length), MessageType.Info);
            }
        }

        if (GUILayout.Button("Reduce"))
        {
            if (restrictToBounds && !boundsValid)
            {
                EditorUtility.DisplayDialog("範囲が無効です", "範囲サイズのいずれかがゼロのため、削減を実行できません。サイズを調整してください。", "OK");
                return;
            }

            Bounds? bounds = null;
            if (restrictToBounds && boundsValid)
                bounds = new Bounds(boundsCenter, boundsSize);
            MeshPolygonReducer.ReduceSelected(selected, reductionPercent, bounds);
        }
    }

    private void OnSceneGUI(SceneView sceneView)
    {
        if (!restrictToBounds)
            return;

        if (boundsSize.x <= 0f || boundsSize.y <= 0f || boundsSize.z <= 0f)
            return;

        DrawBounds(boundsCenter, boundsSize);
    }

    private void DrawBounds(Vector3 center, Vector3 size)
    {
        Vector3 extents = size * 0.5f;
        var prevColor = Handles.color;
        var prevZTest = Handles.zTest;
        Handles.zTest = CompareFunction.LessEqual;

        Vector3[] faceNormals =
        {
            Vector3.up,
            Vector3.down,
            Vector3.left,
            Vector3.right,
            Vector3.forward,
            Vector3.back
        };
        Vector3[] faceRight =
        {
            Vector3.right,
            Vector3.right,
            Vector3.forward,
            Vector3.forward,
            Vector3.right,
            Vector3.right
        };
        Vector3[] faceUp =
        {
            Vector3.forward,
            Vector3.forward,
            Vector3.up,
            Vector3.up,
            Vector3.up,
            Vector3.up
        };

        for (int i = 0; i < faceNormals.Length; i++)
        {
            Vector3 normal = faceNormals[i];
            Vector3 right = faceRight[i];
            Vector3 up = faceUp[i];

            Vector3 faceCenter = center + Vector3.Scale(normal, extents);
            Vector3 rightOffset = Vector3.Scale(right, extents);
            Vector3 upOffset = Vector3.Scale(up, extents);

            Vector3[] verts = new Vector3[4];
            verts[0] = faceCenter + rightOffset + upOffset;
            verts[1] = faceCenter + rightOffset - upOffset;
            verts[2] = faceCenter - rightOffset - upOffset;
            verts[3] = faceCenter - rightOffset + upOffset;

            Handles.DrawSolidRectangleWithOutline(verts, BoundsFillColor, BoundsOutlineColor);
        }

        Handles.color = BoundsOutlineColor;
        Handles.DrawWireCube(center, size);

        Handles.color = prevColor;
        Handles.zTest = prevZTest;
    }

    private void SyncBoundsWithSelection()
    {
        var selected = Selection.activeGameObject;
        if (selected == null)
            return;

        var bounds = CalculateSelectionBounds(selected);
        boundsCenter = bounds.center;
        boundsSize = bounds.size;
    }

    private void OnSelectionChanged()
    {
        SyncBoundsWithSelection();
        Repaint();
        SceneView.RepaintAll();
    }

    private Bounds CalculateSelectionBounds(GameObject root)
    {
        var renderers = root.GetComponentsInChildren<SkinnedMeshRenderer>();
        if (renderers.Length == 0)
            return new Bounds(root.transform.position, Vector3.one);

        Bounds combined = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++)
            combined.Encapsulate(renderers[i].bounds);

        return combined;
    }
}
