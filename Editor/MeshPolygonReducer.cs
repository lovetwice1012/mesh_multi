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

        var clusterResult = BuildClusteredMesh(mesh, vertexMask, desiredCandidateVertices);
        if (clusterResult == null)
            return null;

        var newMesh = clusterResult.Value.NewMesh;
        bool removedAnyTriangles = clusterResult.Value.RemovedTriangle;
        if (!removedAnyTriangles)
            return null;

        return newMesh;
    }

    private static ClusterBuildResult? BuildClusteredMesh(Mesh mesh, bool[] vertexMask, int desiredCandidateVertices)
    {
        var vertices = mesh.vertices;
        int vertexCount = vertices.Length;

        Bounds bounds = CalculateBounds(vertices, vertexMask);
        Vector3 boundsSize = bounds.size;
        if (boundsSize.x < 1e-5f) boundsSize.x = 1e-5f;
        if (boundsSize.y < 1e-5f) boundsSize.y = 1e-5f;
        if (boundsSize.z < 1e-5f) boundsSize.z = 1e-5f;

        int desiredClusters = Mathf.Clamp(desiredCandidateVertices, 3, vertexCount);
        float approxCells = Mathf.Pow(desiredClusters, 1f / 3f);
        Vector3 baseCellSize = new Vector3(
            boundsSize.x / Mathf.Max(1f, approxCells),
            boundsSize.y / Mathf.Max(1f, approxCells),
            boundsSize.z / Mathf.Max(1f, approxCells));
        baseCellSize.x = Mathf.Max(baseCellSize.x, 1e-5f);
        baseCellSize.y = Mathf.Max(baseCellSize.y, 1e-5f);
        baseCellSize.z = Mathf.Max(baseCellSize.z, 1e-5f);

        ClusterBuildResult? bestResult = null;
        float bestDifference = float.MaxValue;
        float scale = 1f;

        for (int iteration = 0; iteration < 12; iteration++)
        {
            Vector3 cellSize = new Vector3(
                Mathf.Max(baseCellSize.x * scale, 1e-5f),
                Mathf.Max(baseCellSize.y * scale, 1e-5f),
                Mathf.Max(baseCellSize.z * scale, 1e-5f));
            var result = GenerateClusters(mesh, vertexMask, cellSize);
            int clusterCount = result.ClusterCount;
            float difference = Mathf.Abs(clusterCount - desiredClusters);
            if (difference < bestDifference)
            {
                bestDifference = difference;
                bestResult = result;
            }

            if (clusterCount > desiredClusters)
                scale *= 1.3f;
            else if (clusterCount < desiredClusters)
                scale *= 0.7f;
            else
                break;
        }

        if (!bestResult.HasValue)
            return null;

        return bestResult;
    }

    private static ClusterBuildResult GenerateClusters(Mesh mesh, bool[] vertexMask, Vector3 cellSize)
    {
        var vertices = mesh.vertices;
        var normals = mesh.normals;
        var tangents = mesh.tangents;
        var colors = mesh.colors;
        var boneWeights = mesh.boneWeights;
        var bindposes = mesh.bindposes;

        Vector2[][] uvChannels = GetUVChannels(mesh);

        int vertexCount = vertices.Length;
        Bounds bounds = CalculateBounds(vertices, vertexMask);
        Vector3 min = bounds.min;

        var clusters = new List<ClusterData>();
        var clusterLookup = new Dictionary<ClusterKey, int>();
        var vertexToCluster = new int[vertexCount];

        bool useMask = vertexMask != null && vertexMask.Length == vertexCount;

        for (int i = 0; i < vertexCount; i++)
        {
            bool reduce = !useMask || vertexMask[i];
            ClusterData data;
            int clusterIndex;
            if (!reduce)
            {
                data = new ClusterData();
                data.AddVertex(i, vertices, normals, tangents, colors, uvChannels, boneWeights);
                clusters.Add(data);
                clusterIndex = clusters.Count - 1;
                vertexToCluster[i] = clusterIndex;
                continue;
            }

            ClusterKey key = new ClusterKey(vertices[i], min, cellSize);
            if (!clusterLookup.TryGetValue(key, out clusterIndex))
            {
                clusterIndex = clusters.Count;
                clusterLookup.Add(key, clusterIndex);
                data = new ClusterData();
                clusters.Add(data);
            }
            data = clusters[clusterIndex];
            data.AddVertex(i, vertices, normals, tangents, colors, uvChannels, boneWeights);
            clusters[clusterIndex] = data;
            vertexToCluster[i] = clusterIndex;
        }

        int clusterCount = clusters.Count;
        var newVertices = new Vector3[clusterCount];
        Vector3[] newNormals = normals != null && normals.Length == vertexCount ? new Vector3[clusterCount] : null;
        Vector4[] newTangents = tangents != null && tangents.Length == vertexCount ? new Vector4[clusterCount] : null;
        Color[] newColors = colors != null && colors.Length == vertexCount ? new Color[clusterCount] : null;
        BoneWeight[] newBoneWeights = boneWeights != null && boneWeights.Length == vertexCount ? new BoneWeight[clusterCount] : null;
        List<Vector2>[] newUVs = new List<Vector2>[uvChannels.Length];
        for (int channel = 0; channel < uvChannels.Length; channel++)
        {
            if (uvChannels[channel] != null && uvChannels[channel].Length == vertexCount)
                newUVs[channel] = new List<Vector2>(clusterCount);
        }

        for (int clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++)
        {
            var cluster = clusters[clusterIndex];
            newVertices[clusterIndex] = cluster.GetAveragePosition();
            if (newNormals != null)
                newNormals[clusterIndex] = cluster.GetAverageNormal();
            if (newTangents != null)
                newTangents[clusterIndex] = cluster.GetAverageTangent();
            if (newColors != null)
                newColors[clusterIndex] = cluster.GetAverageColor();
            if (newBoneWeights != null)
                newBoneWeights[clusterIndex] = cluster.GetAverageBoneWeight();
            for (int channel = 0; channel < uvChannels.Length; channel++)
            {
                if (newUVs[channel] != null)
                    newUVs[channel].Add(cluster.GetAverageUV(channel));
            }
        }

        var newMesh = new Mesh();
        newMesh.name = mesh.name;
        newMesh.indexFormat = clusterCount > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
        newMesh.vertices = newVertices;
        if (newNormals != null)
            newMesh.normals = newNormals;
        if (newTangents != null)
            newMesh.tangents = newTangents;
        if (newColors != null)
            newMesh.colors = newColors;
        if (newBoneWeights != null)
            newMesh.boneWeights = newBoneWeights;
        if (bindposes != null && bindposes.Length > 0)
            newMesh.bindposes = bindposes;

        for (int channel = 0; channel < uvChannels.Length; channel++)
        {
            if (newUVs[channel] != null)
                newMesh.SetUVs(channel, newUVs[channel]);
        }

        int originalSubMeshCount = mesh.subMeshCount;
        int subMeshCount = Mathf.Max(1, originalSubMeshCount);
        newMesh.subMeshCount = subMeshCount;
        bool removedAnyTriangle = false;
        for (int subMesh = 0; subMesh < subMeshCount; subMesh++)
        {
            int[] triangles = originalSubMeshCount == 0 && subMesh == 0 ? mesh.triangles : mesh.GetTriangles(subMesh);
            var newTriangles = new List<int>(triangles.Length);
            for (int t = 0; t < triangles.Length; t += 3)
            {
                int a = vertexToCluster[triangles[t]];
                int b = vertexToCluster[triangles[t + 1]];
                int c = vertexToCluster[triangles[t + 2]];
                if (a == b || b == c || c == a)
                {
                    removedAnyTriangle = true;
                    continue;
                }
                newTriangles.Add(a);
                newTriangles.Add(b);
                newTriangles.Add(c);
            }
            newMesh.SetTriangles(newTriangles, subMesh);
            if (newTriangles.Count == 0)
                removedAnyTriangle = true;
        }

        CopyBlendShapes(mesh, newMesh, clusters, vertexToCluster);

        newMesh.RecalculateBounds();
        if ((mesh.normals == null || mesh.normals.Length == 0) && newMesh.normals == null)
            newMesh.RecalculateNormals();
        if ((mesh.tangents == null || mesh.tangents.Length == 0) && newMesh.tangents == null)
            newMesh.RecalculateTangents();

        bool anyReduction = removedAnyTriangle || clusterCount != vertexCount;

        return new ClusterBuildResult
        {
            ClusterCount = clusterCount,
            NewMesh = newMesh,
            RemovedTriangle = anyReduction
        };
    }

    private static void CopyBlendShapes(Mesh originalMesh, Mesh newMesh, List<ClusterData> clusters, int[] vertexToCluster)
    {
        int blendShapeCount = originalMesh.blendShapeCount;
        if (blendShapeCount == 0)
            return;

        var clusterCounts = new int[clusters.Count];
        for (int i = 0; i < vertexToCluster.Length; i++)
            clusterCounts[vertexToCluster[i]]++;

        var deltaVertices = new Vector3[originalMesh.vertexCount];
        var deltaNormals = new Vector3[originalMesh.vertexCount];
        var deltaTangents = new Vector3[originalMesh.vertexCount];

        for (int shapeIndex = 0; shapeIndex < blendShapeCount; shapeIndex++)
        {
            string shapeName = originalMesh.GetBlendShapeName(shapeIndex);
            int frameCount = originalMesh.GetBlendShapeFrameCount(shapeIndex);
            for (int frameIndex = 0; frameIndex < frameCount; frameIndex++)
            {
                float weight = originalMesh.GetBlendShapeFrameWeight(shapeIndex, frameIndex);
                originalMesh.GetBlendShapeFrameVertices(shapeIndex, frameIndex, deltaVertices, deltaNormals, deltaTangents);

                var newDeltaVertices = new Vector3[clusters.Count];
                var newDeltaNormals = new Vector3[clusters.Count];
                var newDeltaTangents = new Vector3[clusters.Count];

                for (int v = 0; v < vertexToCluster.Length; v++)
                {
                    int clusterIndex = vertexToCluster[v];
                    newDeltaVertices[clusterIndex] += deltaVertices[v];
                    newDeltaNormals[clusterIndex] += deltaNormals[v];
                    newDeltaTangents[clusterIndex] += deltaTangents[v];
                }

                for (int clusterIndex = 0; clusterIndex < clusters.Count; clusterIndex++)
                {
                    int count = Mathf.Max(1, clusterCounts[clusterIndex]);
                    newDeltaVertices[clusterIndex] /= count;
                    newDeltaNormals[clusterIndex] /= count;
                    newDeltaTangents[clusterIndex] /= count;
                }

                newMesh.AddBlendShapeFrame(shapeName, weight, newDeltaVertices, newDeltaNormals, newDeltaTangents);
            }
        }
    }

    private struct ClusterKey : IEquatable<ClusterKey>
    {
        private readonly int x;
        private readonly int y;
        private readonly int z;

        public ClusterKey(Vector3 position, Vector3 min, Vector3 cellSize)
        {
            x = Mathf.FloorToInt((position.x - min.x) / cellSize.x);
            y = Mathf.FloorToInt((position.y - min.y) / cellSize.y);
            z = Mathf.FloorToInt((position.z - min.z) / cellSize.z);
        }

        public bool Equals(ClusterKey other)
        {
            return x == other.x && y == other.y && z == other.z;
        }

        public override bool Equals(object obj)
        {
            return obj is ClusterKey other && Equals(other);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = x;
                hash = (hash * 397) ^ y;
                hash = (hash * 397) ^ z;
                return hash;
            }
        }
    }

    private struct ClusterData
    {
        private Vector3 positionSum;
        private Vector3 normalSum;
        private Vector4 tangentSum;
        private Color colorSum;
        private int count;
        private Dictionary<int, float> boneWeights;
        private Vector2[] uvSum;
        private bool[] uvHasData;

        public void AddVertex(int index, Vector3[] vertices, Vector3[] normals, Vector4[] tangents, Color[] colors, Vector2[][] uvChannels, BoneWeight[] boneWeightArray)
        {
            positionSum += vertices[index];
            if (normals != null && normals.Length == vertices.Length)
                normalSum += normals[index];
            if (tangents != null && tangents.Length == vertices.Length)
                tangentSum += tangents[index];
            if (colors != null && colors.Length == vertices.Length)
                colorSum += colors[index];

            if (uvChannels != null)
            {
                if (uvSum == null)
                {
                    uvSum = new Vector2[uvChannels.Length];
                    uvHasData = new bool[uvChannels.Length];
                }
                for (int channel = 0; channel < uvChannels.Length; channel++)
                {
                    var channelData = uvChannels[channel];
                    if (channelData != null && channelData.Length == vertices.Length)
                    {
                        uvSum[channel] += channelData[index];
                        uvHasData[channel] = true;
                    }
                }
            }

            if (boneWeightArray != null && boneWeightArray.Length == vertices.Length)
            {
                if (boneWeights == null)
                    boneWeights = new Dictionary<int, float>();

                AddBoneWeight(boneWeightArray[index].boneIndex0, boneWeightArray[index].weight0);
                AddBoneWeight(boneWeightArray[index].boneIndex1, boneWeightArray[index].weight1);
                AddBoneWeight(boneWeightArray[index].boneIndex2, boneWeightArray[index].weight2);
                AddBoneWeight(boneWeightArray[index].boneIndex3, boneWeightArray[index].weight3);
            }

            count++;
        }

        private void AddBoneWeight(int boneIndex, float weight)
        {
            if (weight <= 0f)
                return;
            if (boneWeights == null)
                boneWeights = new Dictionary<int, float>();
            if (!boneWeights.ContainsKey(boneIndex))
                boneWeights.Add(boneIndex, weight);
            else
                boneWeights[boneIndex] += weight;
        }

        public Vector3 GetAveragePosition()
        {
            return positionSum / Mathf.Max(1, count);
        }

        public Vector3 GetAverageNormal()
        {
            Vector3 normal = normalSum / Mathf.Max(1, count);
            return normal == Vector3.zero ? normal : normal.normalized;
        }

        public Vector4 GetAverageTangent()
        {
            Vector4 tangent = tangentSum / Mathf.Max(1, count);
            Vector3 xyz = new Vector3(tangent.x, tangent.y, tangent.z);
            if (xyz != Vector3.zero)
                xyz.Normalize();
            float w = tangent.w >= 0f ? 1f : -1f;
            return new Vector4(xyz.x, xyz.y, xyz.z, w);
        }

        public Color GetAverageColor()
        {
            return colorSum / Mathf.Max(1, count);
        }

        public Vector2 GetAverageUV(int channel)
        {
            if (uvSum == null || uvSum.Length <= channel || uvHasData == null || !uvHasData[channel])
                return Vector2.zero;
            return uvSum[channel] / Mathf.Max(1, count);
        }

        public BoneWeight GetAverageBoneWeight()
        {
            if (boneWeights == null || boneWeights.Count == 0)
                return default;

            var ordered = new List<KeyValuePair<int, float>>(boneWeights.Count);
            foreach (var kv in boneWeights)
                ordered.Add(kv);
            ordered.Sort((a, b) => b.Value.CompareTo(a.Value));

            BoneWeight result = new BoneWeight();
            float total = 0f;
            int influenceCount = Mathf.Min(4, ordered.Count);
            for (int i = 0; i < influenceCount; i++)
                total += ordered[i].Value;
            if (total <= 0f)
                total = 1f;

            for (int i = 0; i < influenceCount; i++)
            {
                float normalized = ordered[i].Value / total;
                switch (i)
                {
                    case 0:
                        result.boneIndex0 = ordered[i].Key;
                        result.weight0 = normalized;
                        break;
                    case 1:
                        result.boneIndex1 = ordered[i].Key;
                        result.weight1 = normalized;
                        break;
                    case 2:
                        result.boneIndex2 = ordered[i].Key;
                        result.weight2 = normalized;
                        break;
                    case 3:
                        result.boneIndex3 = ordered[i].Key;
                        result.weight3 = normalized;
                        break;
                }
            }
            return result;
        }
    }

    private struct ClusterBuildResult
    {
        public int ClusterCount;
        public Mesh NewMesh;
        public bool RemovedTriangle;
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
