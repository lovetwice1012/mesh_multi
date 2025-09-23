using System;
using System.Collections.Generic;
using Unity.Collections;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEditor.SceneManagement;

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

        var renderers = selected.GetComponentsInChildren<SkinnedMeshRenderer>(includeInactive: true);
        var targetRenderers = new List<SkinnedMeshRenderer>();
        Dictionary<MaskCacheKey, bool[]> vertexMaskCache = null;
        if (limitBounds.HasValue)
        {
            vertexMaskCache = new Dictionary<MaskCacheKey, bool[]>();
            Bounds bounds = limitBounds.Value;
            for (int i = 0; i < renderers.Length; i++)
            {
                var renderer = renderers[i];
                var mesh = renderer.sharedMesh;
                if (mesh == null)
                    continue;

                bool evaluateInCurrentPose = ShouldEvaluateInCurrentPose(renderer);
                var mask = GetOrCreateMask(renderer, mesh, bounds, vertexMaskCache, evaluateInCurrentPose);
                bool anyInside = false;
                for (int v = 0; v < mask.Length; v++)
                {
                    if (mask[v])
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
        bool assetCreated = false;
        bool anyRendererModified = false;
        bool assetEditingStarted = false;
        int totalOriginalTriangles = 0;
        int totalReducedTriangles = 0;
        int successfulRenderers = 0;

        try
        {
            for (int i = 0; i < total; i++)
            {
                var renderer = targetRenderers[i];
                EditorUtility.DisplayProgressBar("Reduce Mesh Polygons", $"Processing {i + 1}/{total}: {renderer.name}", (float)i / total);

                var originalMesh = renderer.sharedMesh;
                if (originalMesh == null)
                    continue;

                int originalTriangleCount = CountTotalTriangles(originalMesh);
                if (originalTriangleCount <= 0)
                    continue;

                bool[] vertexMask = null;
                if (limitBounds.HasValue)
                {
                    bool evaluateInCurrentPose = ShouldEvaluateInCurrentPose(renderer);
                    vertexMask = GetOrCreateMask(renderer, originalMesh, limitBounds.Value, vertexMaskCache, evaluateInCurrentPose);
                }

                Undo.RegisterCompleteObjectUndo(renderer, "Reduce Skinned Mesh Polygons");

                var newMesh = ReduceMesh(originalMesh, reductionRatio, vertexMask, seed);
                if (newMesh == null)
                    continue;

                float percent = ((float)(i + 1) / total) * 100f;
                percent = Mathf.Floor(percent * 1000f) / 1000f;
                EditorUtility.DisplayProgressBar("Reduce Mesh Polygons", $"Processed {i + 1}/{total}: {renderer.name} ({percent:F3}%)", (float)(i + 1) / total);
                newMesh.name = originalMesh.name + "_reduced";

                if (!assetEditingStarted)
                {
                    AssetDatabase.StartAssetEditing();
                    assetEditingStarted = true;
                }

                bool createdAsset = MeshAssetUtility.TryCreateDerivedMeshAsset(newMesh, originalMesh, "reduced", out _);
                if (!createdAsset)
                {
                    Debug.LogError($"Failed to create reduced mesh asset for '{renderer.name}'. The original mesh will remain unchanged.");
                    renderer.sharedMesh = originalMesh;
                    UnityEngine.Object.DestroyImmediate(newMesh);
                    continue;
                }

                assetCreated = true;
                renderer.sharedMesh = newMesh;
                EditorUtility.SetDirty(renderer);
                if (renderer.sharedMesh != null)
                    EditorUtility.SetDirty(renderer.sharedMesh);
                PrefabUtility.RecordPrefabInstancePropertyModifications(renderer);
                EditorSceneManager.MarkSceneDirty(renderer.gameObject.scene);
                anyRendererModified = true;
                successfulRenderers++;
                totalOriginalTriangles += originalTriangleCount;
                totalReducedTriangles += Mathf.Max(0, CountTotalTriangles(newMesh));
            }
        }
        finally
        {
            if (assetEditingStarted)
                AssetDatabase.StopAssetEditing();
            EditorUtility.ClearProgressBar();
        }

        if (assetCreated)
        {
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
        }

        if (anyRendererModified)
            SceneView.RepaintAll();

        float finalPercent = 0f;
        if (totalOriginalTriangles > 0)
        {
            float reduction = 1f - (float)totalReducedTriangles / totalOriginalTriangles;
            finalPercent = Mathf.Clamp(reduction * 100f, 0f, 100f);
        }
        finalPercent = Mathf.Floor(finalPercent * 1000f) / 1000f;
        Debug.Log(string.Format("Reduced polygons for {0}/{1} meshes under '{2}' ({3:F3}% reduction).", successfulRenderers, targetRenderers.Count, selected.name, finalPercent));
    }

    private static Mesh ReduceMesh(Mesh mesh, float reductionRatio, bool[] vertexMask, int? seed)
    {
        if (mesh == null)
        {
            Debug.LogWarning("Cannot reduce polygons because the mesh is null.");
            return null;
        }

        int vertexCount = mesh.vertexCount;
        if (vertexCount < 3)
        {
            Debug.LogWarning($"Mesh '{mesh.name}' does not have enough vertices to perform reduction.");
            return null;
        }

        int originalTriangles = CountTotalTriangles(mesh);
        if (originalTriangles <= 0)
        {
            Debug.LogWarning($"Mesh '{mesh.name}' does not contain any triangles to reduce.");
            return null;
        }

        bool useMask = vertexMask != null;
        if (vertexMask != null && vertexMask.Length != vertexCount)
        {
            Debug.LogWarning($"Vertex mask length ({vertexMask.Length}) does not match vertex count ({vertexCount}) on mesh '{mesh.name}'. Ignoring mask.");
            useMask = false;
        }
        bool[] effectiveMask = useMask ? vertexMask : null;
        int candidateVertices = 0;
        if (effectiveMask != null)
        {
            for (int i = 0; i < effectiveMask.Length; i++)
            {
                if (effectiveMask[i])
                    candidateVertices++;
            }
            if (candidateVertices < 3)
            {
                Debug.LogWarning($"Bounds restriction leaves fewer than three vertices eligible for reduction on mesh '{mesh.name}'.");
                return null;
            }
        }
        else
        {
            candidateVertices = vertexCount;
        }

        reductionRatio = Mathf.Clamp01(reductionRatio);
        if (reductionRatio <= 0f)
        {
            Debug.LogWarning($"Reduction ratio of {reductionRatio:P0} does not remove any polygons for mesh '{mesh.name}'.");
            return null;
        }

        float targetTriangleFloat = Mathf.Max(1f, originalTriangles * (1f - reductionRatio));
        int targetTriangles = Mathf.Max(1, Mathf.RoundToInt(targetTriangleFloat));
        int lowerTriangleBound = Mathf.Max(1, Mathf.RoundToInt(targetTriangleFloat * 0.95f));
        int upperTriangleBound = Mathf.Max(lowerTriangleBound, Mathf.RoundToInt(targetTriangleFloat * 1.05f));
        upperTriangleBound = Mathf.Min(upperTriangleBound, originalTriangles);

        var simplifier = new ArapMeshSimplifier(mesh, effectiveMask, seed);
        int actualCandidateVertices = simplifier.GetCandidateVertexCount();
        if (actualCandidateVertices < 3)
        {
            Debug.LogWarning($"Only {actualCandidateVertices} vertices are available for reduction on mesh '{mesh.name}'. The reduction bounds or topology constraints leave too few vertices to simplify.");
            return null;
        }

        candidateVertices = Mathf.Min(candidateVertices, actualCandidateVertices);
        int desiredCandidateVertices = Mathf.Max(3, candidateVertices - Mathf.RoundToInt(candidateVertices * reductionRatio));
        if (desiredCandidateVertices >= candidateVertices)
        {
            Debug.LogWarning($"Requested reduction keeps all {candidateVertices} candidate vertices on mesh '{mesh.name}'. Increase the reduction percent or relax the bounds.");
            return null;
        }

        if (candidateVertices <= 3)
        {
            Debug.LogWarning($"Only {candidateVertices} vertices are available for reduction on mesh '{mesh.name}'. The reduction percent is too small for the selected bounds.");
            return null;
        }

        int maxCandidateVertices = Mathf.Min(candidateVertices - 1, actualCandidateVertices - 1);
        int minTarget = Mathf.Clamp(desiredCandidateVertices, 3, maxCandidateVertices);
        int low = minTarget;
        int high = maxCandidateVertices;
        Mesh bestMesh = null;
        bool success = false;
        float bestDeviation = float.MaxValue;
        int bestTriangleCount = -1;
        Mesh fallbackMesh = null;
        float fallbackDeviation = float.MaxValue;
        int fallbackTriangleCount = int.MaxValue;

        bool TryStoreFallback(Mesh candidateMesh, int triangleCount)
        {
            if (candidateMesh == null)
                return false;

            if (triangleCount >= originalTriangles)
            {
                UnityEngine.Object.DestroyImmediate(candidateMesh);
                return false;
            }

            float deviation = Mathf.Abs(triangleCount - targetTriangles);
            bool replace = fallbackMesh == null || deviation < fallbackDeviation ||
                           (Mathf.Approximately(deviation, fallbackDeviation) && triangleCount < fallbackTriangleCount);
            if (replace)
            {
                if (fallbackMesh != null)
                    UnityEngine.Object.DestroyImmediate(fallbackMesh);
                fallbackMesh = candidateMesh;
                fallbackDeviation = deviation;
                fallbackTriangleCount = triangleCount;
                return true;
            }

            UnityEngine.Object.DestroyImmediate(candidateMesh);
            return false;
        }

        while (low <= high)
        {
            int target = (low + high) / 2;
            if (TrySimplify(simplifier, target, out var simplified))
            {
                int simplifiedTriangles = CountTotalTriangles(simplified);
                if (simplifiedTriangles < lowerTriangleBound)
                {
                    TryStoreFallback(simplified, simplifiedTriangles);
                    low = target + 1;
                    continue;
                }

                if (simplifiedTriangles > upperTriangleBound)
                {
                    TryStoreFallback(simplified, simplifiedTriangles);
                    high = target - 1;
                    continue;
                }

                float deviation = Mathf.Abs(simplifiedTriangles - targetTriangles);
                bool shouldReplace = !success || deviation < bestDeviation ||
                                     (Mathf.Approximately(deviation, bestDeviation) && simplifiedTriangles > bestTriangleCount);
                if (shouldReplace)
                {
                    if (bestMesh != null)
                        UnityEngine.Object.DestroyImmediate(bestMesh);
                    bestMesh = simplified;
                    bestDeviation = deviation;
                    bestTriangleCount = simplifiedTriangles;
                }
                else
                {
                    TryStoreFallback(simplified, simplifiedTriangles);
                }

                success = true;
                high = target - 1;
            }
            else
            {
                low = target + 1;
            }
        }

        if (success)
        {
            if (fallbackMesh != null)
                UnityEngine.Object.DestroyImmediate(fallbackMesh);
            return bestMesh;
        }

        return fallbackMesh;
    }

    private static bool TrySimplify(ArapMeshSimplifier simplifier, int targetCandidateCount, out Mesh simplified)
    {
        var result = simplifier.Simplify(targetCandidateCount);
        if (result.HasValue)
        {
            if (result.Value.RemovedTriangles && result.Value.Mesh != null)
            {
                simplified = result.Value.Mesh;
                return true;
            }

            if (result.Value.Mesh != null)
                UnityEngine.Object.DestroyImmediate(result.Value.Mesh);
        }

        simplified = null;
        return false;
    }

    private static bool[] GetOrCreateMask(SkinnedMeshRenderer renderer, Mesh mesh, Bounds bounds, Dictionary<MaskCacheKey, bool[]> cache, bool evaluateInCurrentPose)
    {
        if (evaluateInCurrentPose)
            return CalculateVerticesInsideBounds(renderer, mesh, bounds, evaluateInCurrentPose: true);

        if (cache == null)
            return CalculateVerticesInsideBounds(renderer, mesh, bounds);

        var key = new MaskCacheKey(renderer, mesh, bounds);
        if (!cache.TryGetValue(key, out var mask))
        {
            mask = CalculateVerticesInsideBounds(renderer, mesh, bounds);
            cache[key] = mask;
        }

        return mask;
    }

    internal static bool ShouldEvaluateInCurrentPose(SkinnedMeshRenderer renderer)
    {
        if (renderer == null)
            return false;

        if (renderer.bones != null && renderer.bones.Length > 0)
            return true;

        if (renderer.rootBone != null)
            return true;

        var mesh = renderer.sharedMesh;
        return mesh != null && mesh.bindposes != null && mesh.bindposes.Length > 0;
    }

    internal static bool[] CalculateVerticesInsideBounds(SkinnedMeshRenderer renderer, Mesh mesh, Bounds bounds, bool evaluateInCurrentPose = false)
    {
        if (mesh == null)
            return Array.Empty<bool>();

        if (!TryGetMeshVertices(mesh, out var vertices) || vertices == null)
            return Array.Empty<bool>();

        var inside = new bool[vertices.Length];
        if (vertices.Length == 0)
            return inside;

        var worldPositions = new Vector3[vertices.Length];
        string failureMessage = null;
        bool usedSkinning = evaluateInCurrentPose && TryComputeSkinnedWorldPositions(renderer, mesh, vertices, worldPositions, out failureMessage);
        if (!usedSkinning)
        {
            if (evaluateInCurrentPose && !string.IsNullOrEmpty(failureMessage))
            {
                string rendererName = renderer != null ? renderer.name : mesh.name;
                Debug.LogWarning($"Failed to evaluate animated vertex positions for '{rendererName}': {failureMessage}. Falling back to the mesh rest pose.");
            }

            FillWorldPositionsFromTransform(renderer, vertices, worldPositions);
        }

        Vector3 boundsMin = bounds.min;
        Vector3 boundsMax = bounds.max;
        for (int i = 0; i < worldPositions.Length; i++)
        {
            Vector3 w = worldPositions[i];
            inside[i] = w.x >= boundsMin.x && w.x <= boundsMax.x &&
                        w.y >= boundsMin.y && w.y <= boundsMax.y &&
                        w.z >= boundsMin.z && w.z <= boundsMax.z;
        }

        return inside;
    }

    private static void FillWorldPositionsFromTransform(SkinnedMeshRenderer renderer, Vector3[] vertices, Vector3[] worldPositions)
    {
        Matrix4x4 localToWorld = renderer != null ? renderer.transform.localToWorldMatrix : Matrix4x4.identity;
        for (int i = 0; i < vertices.Length; i++)
            worldPositions[i] = localToWorld.MultiplyPoint3x4(vertices[i]);
    }

    private static bool TryComputeSkinnedWorldPositions(SkinnedMeshRenderer renderer, Mesh mesh, Vector3[] vertices, Vector3[] worldPositions, out string errorMessage)
    {
        errorMessage = null;
        if (renderer == null)
        {
            errorMessage = "Renderer is null.";
            return false;
        }

        if (!TryGetBoneWeights(mesh, out var boneWeights) || boneWeights == null)
        {
            errorMessage = "The mesh does not expose CPU bone weights.";
            return false;
        }
        var bindPoses = mesh.bindposes;
        var bones = renderer.bones;

        if (boneWeights == null || boneWeights.Length != vertices.Length)
        {
            errorMessage = "Bone weight count does not match vertex count.";
            return false;
        }

        if (bindPoses == null || bindPoses.Length == 0)
        {
            errorMessage = "Mesh is missing bind poses.";
            return false;
        }

        if (bones == null || bones.Length == 0)
        {
            errorMessage = "Renderer does not reference any bones.";
            return false;
        }

        int matrixCount = Math.Min(bindPoses.Length, bones.Length);
        if (matrixCount == 0)
        {
            errorMessage = "No matching bones were found between the renderer and mesh bind poses.";
            return false;
        }

        var boneMatrices = new Matrix4x4[matrixCount];
        Matrix4x4 fallbackMatrix = renderer.transform.localToWorldMatrix;
        for (int i = 0; i < matrixCount; i++)
        {
            Transform bone = bones[i];
            Matrix4x4 boneMatrix = bone != null ? bone.localToWorldMatrix : fallbackMatrix;
            boneMatrices[i] = boneMatrix * bindPoses[i];
        }

        bool anySkinningApplied = false;
        bool encounteredInvalidInfluence = false;
        for (int i = 0; i < vertices.Length; i++)
        {
            BoneWeight weight = boneWeights[i];
            Vector3 local = vertices[i];
            float totalWeight = 0f;
            Vector3 skinned = Vector3.zero;

            ApplyBoneWeight(ref skinned, ref totalWeight, weight.weight0, weight.boneIndex0, boneMatrices, matrixCount, local, ref encounteredInvalidInfluence);
            ApplyBoneWeight(ref skinned, ref totalWeight, weight.weight1, weight.boneIndex1, boneMatrices, matrixCount, local, ref encounteredInvalidInfluence);
            ApplyBoneWeight(ref skinned, ref totalWeight, weight.weight2, weight.boneIndex2, boneMatrices, matrixCount, local, ref encounteredInvalidInfluence);
            ApplyBoneWeight(ref skinned, ref totalWeight, weight.weight3, weight.boneIndex3, boneMatrices, matrixCount, local, ref encounteredInvalidInfluence);

            if (encounteredInvalidInfluence)
            {
                errorMessage = "Bone weights reference indices outside the available bone or bind-pose range.";
                return false;
            }

            if (totalWeight > 0f)
            {
                float remainder = Mathf.Clamp01(1f - totalWeight);
                if (remainder > 0f)
                    skinned += fallbackMatrix.MultiplyPoint3x4(local) * remainder;
                worldPositions[i] = skinned;
                anySkinningApplied = true;
            }
            else
            {
                worldPositions[i] = fallbackMatrix.MultiplyPoint3x4(local);
            }
        }

        if (!anySkinningApplied)
        {
            errorMessage = "No vertices contained valid skinning information.";
            return false;
        }

        return anySkinningApplied;
    }

    private static void ApplyBoneWeight(ref Vector3 accum, ref float totalWeight, float weight, int boneIndex, Matrix4x4[] boneMatrices, int matrixCount, Vector3 localPosition, ref bool invalidIndexEncountered)
    {
        if (weight <= 0f)
            return;

        if (boneIndex < 0 || boneIndex >= matrixCount)
        {
            invalidIndexEncountered = true;
            return;
        }

        accum += boneMatrices[boneIndex].MultiplyPoint3x4(localPosition) * weight;
        totalWeight += weight;
    }

    private static bool TryGetMeshVertices(Mesh mesh, out Vector3[] vertices)
    {
        vertices = null;
        if (mesh == null)
            return false;

        try
        {
            using (var meshDataArray = Mesh.AcquireReadOnlyMeshData(mesh))
            {
                if (meshDataArray.Length == 0)
                    return false;

                var meshData = meshDataArray[0];
                int count = meshData.vertexCount;
                vertices = new Vector3[count];
                if (count == 0)
                    return true;

                using (var native = new NativeArray<Vector3>(count, Allocator.Temp))
                {
                    meshData.GetVertices(native);
                    native.CopyTo(vertices);
                }
                return true;
            }
        }
        catch
        {
            vertices = null;
            return false;
        }
    }

    private static bool TryGetBoneWeights(Mesh mesh, out BoneWeight[] boneWeights)
    {
        boneWeights = null;
        if (mesh == null)
            return false;

#if UNITY_2020_2_OR_NEWER
        try
        {
            using (var meshDataArray = Mesh.AcquireReadOnlyMeshData(mesh))
            {
                if (meshDataArray.Length == 0)
                    return false;

                var meshData = meshDataArray[0];
                if (!meshData.HasVertexAttribute(VertexAttribute.BlendWeight) ||
                    !meshData.HasVertexAttribute(VertexAttribute.BlendIndices))
                    return false;

                int vertexCount = meshData.vertexCount;
                var result = new BoneWeight[vertexCount];
                var bonesPerVertex = mesh.GetBonesPerVertex();
                var allWeights = mesh.GetAllBoneWeights();

                try
                {
                    if (!bonesPerVertex.IsCreated || !allWeights.IsCreated)
                        return false;

                    int offset = 0;
                    for (int i = 0; i < vertexCount; i++)
                    {
                        int influenceCount = i < bonesPerVertex.Length ? bonesPerVertex[i] : 0;
                        influenceCount = Mathf.Clamp(influenceCount, 0, Mathf.Max(0, allWeights.Length - offset));

                        BoneWeight weight = new BoneWeight();
                        for (int j = 0; j < influenceCount && j < 4; j++)
                        {
                            var w = allWeights[offset + j];
                            switch (j)
                            {
                                case 0:
                                    weight.boneIndex0 = w.boneIndex;
                                    weight.weight0 = w.weight;
                                    break;
                                case 1:
                                    weight.boneIndex1 = w.boneIndex;
                                    weight.weight1 = w.weight;
                                    break;
                                case 2:
                                    weight.boneIndex2 = w.boneIndex;
                                    weight.weight2 = w.weight;
                                    break;
                                case 3:
                                    weight.boneIndex3 = w.boneIndex;
                                    weight.weight3 = w.weight;
                                    break;
                            }
                        }

                        NormalizeBoneWeight(ref weight);
                        result[i] = weight;
                        offset += influenceCount;
                    }
                }
                finally
                {
                    if (bonesPerVertex.IsCreated)
                        bonesPerVertex.Dispose();
                    if (allWeights.IsCreated)
                        allWeights.Dispose();
                }

                boneWeights = result;
                return true;
            }
        }
        catch
        {
            boneWeights = null;
            return false;
        }
#else
        var legacyWeights = mesh != null ? mesh.boneWeights : null;
        if (legacyWeights == null || legacyWeights.Length == 0)
        {
            boneWeights = null;
            return false;
        }

        var result = new BoneWeight[legacyWeights.Length];
        Array.Copy(legacyWeights, result, legacyWeights.Length);
        boneWeights = result;
        return true;
#endif
    }

    private static void NormalizeBoneWeight(ref BoneWeight weight)
    {
        float total = weight.weight0 + weight.weight1 + weight.weight2 + weight.weight3;
        if (total <= 0f)
            return;
        float inv = 1f / total;
        weight.weight0 *= inv;
        weight.weight1 *= inv;
        weight.weight2 *= inv;
        weight.weight3 *= inv;
    }

    internal static int CountTotalTriangles(Mesh mesh)
    {
        if (mesh == null)
            return 0;

        using (var meshDataArray = Mesh.AcquireReadOnlyMeshData(mesh))
        {
            if (meshDataArray.Length == 0)
                return 0;

            var meshData = meshDataArray[0];
            int subMeshCount = Mathf.Max(1, Mathf.Max(mesh.subMeshCount, meshData.subMeshCount));
            int total = 0;
            for (int i = 0; i < subMeshCount; i++)
            {
                MeshTopology topology;
                if (mesh.subMeshCount > 0 && i < mesh.subMeshCount)
                    topology = mesh.GetTopology(i);
                else if (i < meshData.subMeshCount)
                    topology = meshData.GetSubMesh(i).topology;
                else
                    topology = MeshTopology.Triangles;

                if (i < meshData.subMeshCount)
                {
                    var descriptor = meshData.GetSubMesh(i);
                    total += CountTrianglesForTopology(meshData, i, descriptor, topology);
                }
            }

            return total;
        }
    }

    private static int CountTrianglesForTopology(Mesh.MeshData meshData, int submeshIndex, SubMeshDescriptor descriptor, MeshTopology topology)
    {
        switch (topology)
        {
            case MeshTopology.Triangles:
                return descriptor.indexCount / 3;
            case MeshTopology.Quads:
                return (descriptor.indexCount / 4) * 2;
            case MeshTopology.TriangleStrip:
                return CountTriangleStripTriangles(meshData, submeshIndex);
            case MeshTopology.TriangleFan:
                return CountTriangleFanTriangles(meshData, submeshIndex);
            default:
                return 0;
        }
    }

    private static int CountTriangleStripTriangles(Mesh.MeshData meshData, int submeshIndex)
    {
        NativeArray<int> indices = default;
        try
        {
            indices = meshData.GetIndices<int>(submeshIndex);
            if (!indices.IsCreated || indices.Length < 3)
                return 0;

            int a = indices[0];
            int b = indices[1];
            int count = 0;
            for (int i = 2; i < indices.Length; i++)
            {
                int c = indices[i];
                if (a != b && b != c && c != a)
                    count++;
                a = b;
                b = c;
            }

            return count;
        }
        finally
        {
            if (indices.IsCreated)
                indices.Dispose();
        }
    }

    private static int CountTriangleFanTriangles(Mesh.MeshData meshData, int submeshIndex)
    {
        NativeArray<int> indices = default;
        try
        {
            indices = meshData.GetIndices<int>(submeshIndex);
            if (!indices.IsCreated || indices.Length < 3)
                return 0;

            int center = indices[0];
            int previous = indices[1];
            int count = 0;
            for (int i = 2; i < indices.Length; i++)
            {
                int current = indices[i];
                if (center != previous && previous != current && current != center)
                    count++;
                previous = current;
            }

            return count;
        }
        finally
        {
            if (indices.IsCreated)
                indices.Dispose();
        }
    }
}

internal readonly struct MaskCacheKey : IEquatable<MaskCacheKey>
{
    private readonly int rendererId;
    private readonly int meshId;
    private readonly Vector3 center;
    private readonly Vector3 extents;

    public MaskCacheKey(SkinnedMeshRenderer renderer, Mesh mesh, Bounds bounds)
    {
        rendererId = renderer != null ? renderer.GetInstanceID() : 0;
        meshId = mesh != null ? mesh.GetInstanceID() : 0;
        center = bounds.center;
        extents = bounds.extents;
    }

    public bool Equals(MaskCacheKey other)
    {
        return rendererId == other.rendererId &&
               meshId == other.meshId &&
               center.Equals(other.center) &&
               extents.Equals(other.extents);
    }

    public override bool Equals(object obj)
    {
        return obj is MaskCacheKey other && Equals(other);
    }

    public override int GetHashCode()
    {
        unchecked
        {
            int hash = rendererId;
            hash = (hash * 397) ^ meshId;
            hash = (hash * 397) ^ center.GetHashCode();
            hash = (hash * 397) ^ extents.GetHashCode();
            return hash;
        }
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

        var renderers = selected.GetComponentsInChildren<SkinnedMeshRenderer>(includeInactive: true);
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
                bool evaluateInCurrentPose = MeshPolygonReducer.ShouldEvaluateInCurrentPose(renderer);
                mask = MeshPolygonReducer.CalculateVerticesInsideBounds(renderer, mesh, activeBounds, evaluateInCurrentPose);
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
            int totalTriangles = MeshPolygonReducer.CountTotalTriangles(mesh);
            int affectedTriangles = totalTriangles;
            if (restrictToBounds && boundsValid && mask != null)
            {
                affectedTriangles = CountTrianglesWithinMask(mesh, mask);
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

    private static int CountTrianglesWithinMask(Mesh mesh, bool[] mask)
    {
        if (mesh == null || mask == null)
            return 0;

        using (var meshDataArray = Mesh.AcquireReadOnlyMeshData(mesh))
        {
            if (meshDataArray.Length == 0)
                return 0;

            var meshData = meshDataArray[0];
            int subMeshCount = Mathf.Max(1, Mathf.Max(mesh.subMeshCount, meshData.subMeshCount));
            int total = 0;

            for (int subMesh = 0; subMesh < subMeshCount; subMesh++)
            {
                MeshTopology topology;
                if (mesh.subMeshCount > 0 && subMesh < mesh.subMeshCount)
                    topology = mesh.GetTopology(subMesh);
                else if (subMesh < meshData.subMeshCount)
                    topology = meshData.GetSubMesh(subMesh).topology;
                else
                    topology = MeshTopology.Triangles;

                if (topology != MeshTopology.Triangles || subMesh >= meshData.subMeshCount)
                    continue;

                var descriptor = meshData.GetSubMesh(subMesh);
                if (descriptor.indexCount == 0)
                    continue;

                using (var indices = new NativeArray<int>(descriptor.indexCount, Allocator.Temp))
                {
                    meshData.GetIndices(indices, subMesh);
                    for (int i = 0; i + 2 < indices.Length; i += 3)
                    {
                        if (IsTriangleInsideMask(mask, indices[i], indices[i + 1], indices[i + 2]))
                            total++;
                    }
                }
            }

            return total;
        }
    }

    private static bool IsTriangleInsideMask(bool[] mask, int a, int b, int c)
    {
        int maskLength = mask.Length;
        if (a < 0 || a >= maskLength || b < 0 || b >= maskLength || c < 0 || c >= maskLength)
            return false;

        return mask[a] && mask[b] && mask[c];
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
        var renderers = root.GetComponentsInChildren<SkinnedMeshRenderer>(includeInactive: true);
        if (renderers.Length == 0)
            return new Bounds(root.transform.position, Vector3.one);

        Bounds combined = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++)
            combined.Encapsulate(renderers[i].bounds);

        return combined;
    }
}
