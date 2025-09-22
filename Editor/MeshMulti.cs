using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

public static class MeshMulti
{
    public static void SubdivideSelected(GameObject selected, bool smooth, int smoothIterations = 10, Bounds? limitBounds = null)
    {
        if (selected == null)
        {
            Debug.LogWarning("No GameObject selected.");
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

                var insideMask = CalculateVerticesInsideBounds(renderer, mesh, bounds);
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
        bool createdAsset = false;
        try
        {
            for (int i = 0; i < total; i++)
            {
                var renderer = targetRenderers[i];
                EditorUtility.DisplayProgressBar("Subdivide Meshes", $"Processing {i + 1}/{total}: {renderer.name}", (float)i / total);

                var originalMesh = renderer.sharedMesh;
                if (originalMesh == null)
                    continue;

                bool[] subdivisionMaskInput = null;
                if (limitBounds.HasValue)
                    subdivisionMaskInput = CalculateVerticesInsideBounds(renderer, originalMesh, limitBounds.Value);

                bool[] subdivisionMask;
                var newMesh = SubdivideMesh(originalMesh, subdivisionMaskInput, out subdivisionMask);
                if (smooth)
                {
                    bool[] smoothedVertices = null;
                    if (limitBounds.HasValue)
                    {
                        smoothedVertices = CalculateVerticesInsideBounds(renderer, newMesh, limitBounds.Value);
                        if (subdivisionMask != null && smoothedVertices != null)
                        {
                            int count = Mathf.Min(subdivisionMask.Length, smoothedVertices.Length);
                            for (int v = 0; v < count; v++)
                                smoothedVertices[v] = smoothedVertices[v] || subdivisionMask[v];
                        }
                    }
                    ThinPlateSmooth(newMesh, smoothIterations, 0.1f, i, total, smoothedVertices);
                }

                float percent = ((float)(i + 1) / total) * 100f;
                percent = Mathf.Floor(percent * 1000f) / 1000f;
                EditorUtility.DisplayProgressBar("Subdivide Meshes", $"Processed {i + 1}/{total}: {renderer.name} ({percent:F3}%)", (float)(i + 1) / total);
                newMesh.name = originalMesh.name + "_subdivided";

                if (MeshAssetUtility.TryCreateDerivedMeshAsset(newMesh, originalMesh, "subdivided", out var newPath))
                {
                    createdAsset = true;
                    renderer.sharedMesh = AssetDatabase.LoadAssetAtPath<Mesh>(newPath);
                }
                else
                {
                    renderer.sharedMesh = newMesh;
                }

                EditorUtility.SetDirty(renderer);
            }
        }
        finally
        {
            EditorUtility.ClearProgressBar();
        }

        if (createdAsset)
        {
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
        }

        float finalPercent = Mathf.Floor(100f * 1000f) / 1000f;
        Debug.Log(string.Format("Subdivided {0} meshes under '{1}' ({2:F3}%).", targetRenderers.Count, selected.name, finalPercent));
    }

    public static int PredictSubdividedVertexCount(Mesh mesh, bool[] vertexMask = null)
    {
        int[] triangles = mesh.triangles;
        var edges = new HashSet<Edge>();
        bool useMask = vertexMask != null && vertexMask.Length == mesh.vertexCount;
        bool anyMasked = false;
        if (useMask)
        {
            for (int i = 0; i < vertexMask.Length; i++)
            {
                if (vertexMask[i])
                {
                    anyMasked = true;
                    break;
                }
            }
            if (!anyMasked)
                return mesh.vertexCount;
        }

        for (int i = 0; i < triangles.Length; i += 3)
        {
            int v0 = triangles[i];
            int v1 = triangles[i + 1];
            int v2 = triangles[i + 2];

            if (useMask && anyMasked)
            {
                if (!vertexMask[v0] && !vertexMask[v1] && !vertexMask[v2])
                    continue;
            }

            edges.Add(new Edge(v0, v1));
            edges.Add(new Edge(v1, v2));
            edges.Add(new Edge(v2, v0));
        }
        return mesh.vertexCount + edges.Count;
    }

    public static int PredictSubdividedTriangleCount(Mesh mesh, bool[] vertexMask = null)
    {
        int triangleCount = mesh.triangles.Length / 3;
        bool useMask = vertexMask != null && vertexMask.Length == mesh.vertexCount;
        bool anyMasked = false;
        if (useMask)
        {
            for (int i = 0; i < vertexMask.Length; i++)
            {
                if (vertexMask[i])
                {
                    anyMasked = true;
                    break;
                }
            }
            if (!anyMasked)
                return triangleCount;
        }

        if (!useMask)
            return triangleCount * 4;

        int[] triangles = mesh.triangles;
        int predicted = 0;
        for (int i = 0; i < triangles.Length; i += 3)
        {
            int v0 = triangles[i];
            int v1 = triangles[i + 1];
            int v2 = triangles[i + 2];

            bool e01 = vertexMask[v0] || vertexMask[v1];
            bool e12 = vertexMask[v1] || vertexMask[v2];
            bool e20 = vertexMask[v2] || vertexMask[v0];
            int flagged = (e01 ? 1 : 0) + (e12 ? 1 : 0) + (e20 ? 1 : 0);

            switch (flagged)
            {
                case 0:
                    predicted += 1;
                    break;
                case 1:
                    predicted += 2;
                    break;
                case 2:
                    predicted += 3;
                    break;
                default:
                    predicted += 4;
                    break;
            }
        }

        return predicted;
    }

    private static void ThinPlateSmooth(Mesh mesh, int iterations = 10, float lambda = 0.1f, int meshIndex = 0, int meshTotal = 1, bool[] vertexMask = null)
    {
        var vertices = mesh.vertices;
        var originalVertices = (Vector3[])vertices.Clone();
        var triangles = mesh.triangles;
        var boneWeights = mesh.boneWeights;
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

        // Group vertices by position (with a small tolerance) to account for
        // duplicated vertices along seams. Without this, smoothing each copy
        // independently can introduce visible gaps.
        var positionGroups = new Dictionary<Vector3, List<int>>(new Vector3EqualityComparer(1e-6f));
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
                if (vertexMask != null && !vertexMask[i]) { lap[i] = Vector3.zero; continue; }
                if (adjacency[i].Count == 0) { lap[i] = Vector3.zero; continue; }
                Vector3 sum = Vector3.zero;
                foreach (var n in adjacency[i]) sum += vertices[n];
                lap[i] = sum / adjacency[i].Count - vertices[i];
            }
            for (int i = 0; i < vertices.Length; i++)
            {
                if (vertexMask != null && !vertexMask[i]) { lap2[i] = Vector3.zero; continue; }
                if (adjacency[i].Count == 0) { lap2[i] = Vector3.zero; continue; }
                Vector3 sum = Vector3.zero;
                foreach (var n in adjacency[i]) sum += lap[n];
                lap2[i] = sum / adjacency[i].Count - lap[i];
            }
            // Subtracting moves vertices toward lower curvature. Using addition here
            // would amplify curvature and can explode the mesh during smoothing.
            for (int i = 0; i < vertices.Length; i++)
            {
                if (vertexMask != null && !vertexMask[i])
                    continue;
                vertices[i] -= lambda * lap2[i];
            }

            // Keep duplicate-position vertices welded together
            foreach (var group in positionGroups.Values)
            {
                if (group.Count < 2) continue;
                if (vertexMask == null)
                {
                    Vector3 avg = Vector3.zero;
                    foreach (var idx in group) avg += vertices[idx];
                    avg /= group.Count;
                    foreach (var idx in group) vertices[idx] = avg;
                }
                else
                {
                    var insideMembers = new List<int>();
                    for (int g = 0; g < group.Count; g++)
                    {
                        int idx = group[g];
                        if (vertexMask[idx]) insideMembers.Add(idx);
                    }
                    if (insideMembers.Count < 2) continue;
                    Vector3 avg = Vector3.zero;
                    foreach (var idx in insideMembers) avg += vertices[idx];
                    avg /= insideMembers.Count;
                    foreach (var idx in insideMembers) vertices[idx] = avg;
                }
            }
        }

        // Ensure vertices sharing a position also share averaged bone weights
        if (boneWeights.Length > 0)
        {
            foreach (var group in positionGroups.Values)
            {
                if (group.Count < 2) continue;
                if (vertexMask == null)
                {
                    var weights = new List<BoneWeight>(group.Count);
                    foreach (var idx in group) weights.Add(boneWeights[idx]);
                    var avgWeight = AverageBoneWeights(weights);
                    foreach (var idx in group) boneWeights[idx] = avgWeight;
                }
                else
                {
                    var insideWeights = new List<BoneWeight>();
                    foreach (var idx in group)
                    {
                        if (vertexMask[idx]) insideWeights.Add(boneWeights[idx]);
                    }
                    if (insideWeights.Count < 2) continue;
                    var avgWeight = AverageBoneWeights(insideWeights);
                    foreach (var idx in group)
                    {
                        if (vertexMask[idx])
                            boneWeights[idx] = avgWeight;
                    }
                }
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
        if (boneWeights.Length > 0) mesh.boneWeights = boneWeights;
        mesh.RecalculateBounds();
        mesh.RecalculateNormals();
        mesh.RecalculateTangents();
    }

    internal static bool[] CalculateVerticesInsideBounds(SkinnedMeshRenderer renderer, Mesh mesh, Bounds bounds)
    {
        var vertices = mesh.vertices;
        var inside = new bool[vertices.Length];
        Matrix4x4 localToWorld = renderer.transform.localToWorldMatrix;
        for (int i = 0; i < vertices.Length; i++)
        {
            Vector3 world = localToWorld.MultiplyPoint3x4(vertices[i]);
            inside[i] = bounds.Contains(world);
        }
        return inside;
    }

    private struct TriangleData
    {
        public int a;
        public int b;
        public int c;
        public int subMesh;

        public bool Contains(int v)
        {
            return a == v || b == v || c == v;
        }
    }

    private static void OptimizeMesh(Mesh mesh, float planarTolerance = 1e-4f, float linearTolerance = 1e-6f)
    {
        if (mesh == null || mesh.vertexCount == 0) return;

        Vector3[] vertices = mesh.vertices;
        Vector2[] uv = mesh.uv;
        Vector2[] uv2 = mesh.uv2;
        Vector2[] uv3 = mesh.uv3;
        Vector2[] uv4 = mesh.uv4;
        Vector3[] normals = mesh.normals;
        Vector4[] tangents = mesh.tangents;
        Color[] colors = mesh.colors;
        BoneWeight[] boneWeights = mesh.boneWeights;
        Matrix4x4[] bindposes = mesh.bindposes;

        int subMeshCount = mesh.subMeshCount;
        var triangles = new List<TriangleData>();
        for (int s = 0; s < subMeshCount; s++)
        {
            int[] subTriangles = mesh.GetTriangles(s);
            for (int i = 0; i < subTriangles.Length; i += 3)
            {
                triangles.Add(new TriangleData
                {
                    a = subTriangles[i],
                    b = subTriangles[i + 1],
                    c = subTriangles[i + 2],
                    subMesh = s
                });
            }
        }

        if (triangles.Count == 0) return;

        RemoveDegenerateTriangles(triangles, vertices, linearTolerance);

        bool[] removedVertices = new bool[vertices.Length];
        bool changed = true;
        while (changed)
        {
            changed = false;
            var vertexTriangles = BuildVertexTriangleMap(vertices.Length, triangles, removedVertices);
            for (int v = 0; v < vertices.Length; v++)
            {
                if (removedVertices[v]) continue;
                var connected = vertexTriangles[v];
                if (connected == null || connected.Count < 3) continue;

                List<int> orderedNeighbors;
                int subMesh;
                if (!TryGetNeighborLoop(v, connected, triangles, removedVertices, out orderedNeighbors, out subMesh))
                    continue;

                if (!IsPlanarRegion(v, orderedNeighbors, vertices, planarTolerance))
                    continue;

                if (!IsConvexPolygon(orderedNeighbors, vertices, planarTolerance))
                    continue;

                RemoveTrianglesWithVertex(triangles, v);
                TriangulatePolygon(orderedNeighbors, triangles, vertices, subMesh);
                removedVertices[v] = true;
                changed = true;
                break;
            }
        }

        RemoveDegenerateTriangles(triangles, vertices, linearTolerance);
        RemoveUnusedVertices(triangles, removedVertices);

        int[] remap = new int[vertices.Length];
        int newCount = 0;
        for (int i = 0; i < vertices.Length; i++)
        {
            if (removedVertices[i])
                remap[i] = -1;
            else
            {
                remap[i] = newCount;
                newCount++;
            }
        }

        var newVertices = new List<Vector3>(newCount);
        var newUV = (uv.Length == vertices.Length) ? new List<Vector2>(newCount) : null;
        var newUV2 = (uv2.Length == vertices.Length) ? new List<Vector2>(newCount) : null;
        var newUV3 = (uv3.Length == vertices.Length) ? new List<Vector2>(newCount) : null;
        var newUV4 = (uv4.Length == vertices.Length) ? new List<Vector2>(newCount) : null;
        var newNormals = (normals.Length == vertices.Length) ? new List<Vector3>(newCount) : null;
        var newTangents = (tangents.Length == vertices.Length) ? new List<Vector4>(newCount) : null;
        var newColors = (colors.Length == vertices.Length) ? new List<Color>(newCount) : null;
        var newBoneWeights = (boneWeights.Length == vertices.Length) ? new List<BoneWeight>(newCount) : null;

        for (int i = 0; i < vertices.Length; i++)
        {
            if (remap[i] < 0) continue;
            newVertices.Add(vertices[i]);
            if (newUV != null) newUV.Add(uv[i]);
            if (newUV2 != null) newUV2.Add(uv2[i]);
            if (newUV3 != null) newUV3.Add(uv3[i]);
            if (newUV4 != null) newUV4.Add(uv4[i]);
            if (newNormals != null) newNormals.Add(normals[i]);
            if (newTangents != null) newTangents.Add(tangents[i]);
            if (newColors != null) newColors.Add(colors[i]);
            if (newBoneWeights != null) newBoneWeights.Add(boneWeights[i]);
        }

        var blendShapes = CaptureBlendShapes(mesh);
        foreach (var bs in blendShapes)
        {
            foreach (var frame in bs.frames)
            {
                frame.deltaVertices = Remap(frame.deltaVertices, remap);
                frame.deltaNormals = Remap(frame.deltaNormals, remap);
                frame.deltaTangents = Remap(frame.deltaTangents, remap);
            }
        }

        var trianglesPerSubMesh = new List<int>[subMeshCount];
        for (int s = 0; s < subMeshCount; s++)
            trianglesPerSubMesh[s] = new List<int>();

        for (int i = 0; i < triangles.Count; i++)
        {
            TriangleData tri = triangles[i];
            if (remap[tri.a] < 0 || remap[tri.b] < 0 || remap[tri.c] < 0) continue;
            trianglesPerSubMesh[tri.subMesh].Add(remap[tri.a]);
            trianglesPerSubMesh[tri.subMesh].Add(remap[tri.b]);
            trianglesPerSubMesh[tri.subMesh].Add(remap[tri.c]);
        }

        mesh.Clear();
        mesh.indexFormat = (newVertices.Count > 65535) ? IndexFormat.UInt32 : IndexFormat.UInt16;
        mesh.vertices = newVertices.ToArray();
        if (newUV != null) mesh.uv = newUV.ToArray();
        if (newUV2 != null) mesh.uv2 = newUV2.ToArray();
        if (newUV3 != null) mesh.uv3 = newUV3.ToArray();
        if (newUV4 != null) mesh.uv4 = newUV4.ToArray();
        if (newNormals != null) mesh.normals = newNormals.ToArray();
        if (newTangents != null) mesh.tangents = newTangents.ToArray();
        if (newColors != null) mesh.colors = newColors.ToArray();
        if (newBoneWeights != null) mesh.boneWeights = newBoneWeights.ToArray();
        if (bindposes != null && bindposes.Length > 0) mesh.bindposes = bindposes;

        mesh.subMeshCount = subMeshCount;
        for (int s = 0; s < subMeshCount; s++)
            mesh.SetTriangles(trianglesPerSubMesh[s], s);

        foreach (var bs in blendShapes)
        {
            foreach (var frame in bs.frames)
            {
                mesh.AddBlendShapeFrame(bs.name, frame.weight,
                    frame.deltaVertices.ToArray(),
                    frame.deltaNormals.ToArray(),
                    frame.deltaTangents.ToArray());
            }
        }

        mesh.RecalculateBounds();
        mesh.RecalculateNormals();
        mesh.RecalculateTangents();
    }

    private static void RemoveDegenerateTriangles(List<TriangleData> triangles, Vector3[] vertices, float tolerance)
    {
        float threshold = tolerance * tolerance;
        for (int i = triangles.Count - 1; i >= 0; i--)
        {
            TriangleData tri = triangles[i];
            Vector3 a = vertices[tri.a];
            Vector3 b = vertices[tri.b];
            Vector3 c = vertices[tri.c];
            float area = Vector3.Cross(b - a, c - a).sqrMagnitude;
            if (area <= threshold)
                triangles.RemoveAt(i);
        }
    }

    private static List<int>[] BuildVertexTriangleMap(int vertexCount, List<TriangleData> triangles, bool[] removed)
    {
        var map = new List<int>[vertexCount];
        for (int i = 0; i < triangles.Count; i++)
        {
            TriangleData tri = triangles[i];
            if (removed[tri.a] || removed[tri.b] || removed[tri.c]) continue;
            if (map[tri.a] == null) map[tri.a] = new List<int>();
            if (map[tri.b] == null) map[tri.b] = new List<int>();
            if (map[tri.c] == null) map[tri.c] = new List<int>();
            map[tri.a].Add(i);
            map[tri.b].Add(i);
            map[tri.c].Add(i);
        }
        return map;
    }

    private static bool TryGetNeighborLoop(int vertexIndex, List<int> connectedTriangles, List<TriangleData> triangles, bool[] removedVertices, out List<int> orderedNeighbors, out int subMesh)
    {
        orderedNeighbors = null;
        subMesh = -1;
        var adjacency = new Dictionary<int, List<int>>();

        for (int i = 0; i < connectedTriangles.Count; i++)
        {
            TriangleData tri = triangles[connectedTriangles[i]];
            if (removedVertices[tri.a] || removedVertices[tri.b] || removedVertices[tri.c])
                continue;

            if (subMesh == -1)
                subMesh = tri.subMesh;
            else if (subMesh != tri.subMesh)
                return false;

            int n0, n1;
            if (tri.a == vertexIndex)
            {
                n0 = tri.b; n1 = tri.c;
            }
            else if (tri.b == vertexIndex)
            {
                n0 = tri.c; n1 = tri.a;
            }
            else
            {
                n0 = tri.a; n1 = tri.b;
            }

            if (removedVertices[n0] || removedVertices[n1])
                return false;

            AddNeighbor(adjacency, n0, n1);
            AddNeighbor(adjacency, n1, n0);
        }

        if (adjacency.Count < 3) return false;

        foreach (var kv in adjacency)
        {
            if (kv.Value.Count != 2)
                return false;
        }

        int start = -1;
        foreach (var kv in adjacency)
        {
            start = kv.Key;
            break;
        }
        if (start == -1) return false;

        orderedNeighbors = new List<int>(adjacency.Count);
        int current = start;
        int prev = -1;
        for (int count = 0; count < adjacency.Count; count++)
        {
            orderedNeighbors.Add(current);
            var neighbors = adjacency[current];
            int next = (neighbors[0] == prev) ? neighbors[1] : neighbors[0];
            prev = current;
            current = next;
        }

        if (current != start)
            return false;

        return true;
    }

    private static void AddNeighbor(Dictionary<int, List<int>> adjacency, int key, int value)
    {
        List<int> list;
        if (!adjacency.TryGetValue(key, out list))
        {
            list = new List<int>(2);
            adjacency[key] = list;
        }
        if (!list.Contains(value))
            list.Add(value);
    }

    private static bool IsPlanarRegion(int center, List<int> neighbors, Vector3[] vertices, float tolerance)
    {
        if (neighbors.Count < 3) return false;

        Vector3 normal = Vector3.zero;
        bool normalValid = false;
        float areaThreshold = tolerance * tolerance;
        for (int i = 0; i < neighbors.Count; i++)
        {
            Vector3 p0 = vertices[neighbors[i]];
            Vector3 p1 = vertices[neighbors[(i + 1) % neighbors.Count]];
            Vector3 p2 = vertices[center];
            Vector3 n = Vector3.Cross(p1 - p0, p2 - p0);
            if (n.sqrMagnitude > areaThreshold)
            {
                normal = n.normalized;
                normalValid = true;
                break;
            }
        }
        if (!normalValid) return false;

        float d = -Vector3.Dot(normal, vertices[center]);
        for (int i = 0; i < neighbors.Count; i++)
        {
            Vector3 p = vertices[neighbors[i]];
            float dist = Mathf.Abs(Vector3.Dot(normal, p) + d);
            if (dist > tolerance)
                return false;
        }

        return true;
    }

    private static bool IsConvexPolygon(List<int> neighbors, Vector3[] vertices, float tolerance)
    {
        if (neighbors.Count < 3) return false;

        Vector3 normal = Vector3.zero;
        float threshold = tolerance * tolerance;
        for (int i = 0; i < neighbors.Count; i++)
        {
            Vector3 p0 = vertices[neighbors[i]];
            Vector3 p1 = vertices[neighbors[(i + 1) % neighbors.Count]];
            Vector3 p2 = vertices[neighbors[(i + 2) % neighbors.Count]];
            Vector3 cross = Vector3.Cross(p1 - p0, p2 - p1);
            if (cross.sqrMagnitude > threshold)
            {
                normal = cross.normalized;
                break;
            }
        }
        if (normal == Vector3.zero) return false;

        for (int i = 0; i < neighbors.Count; i++)
        {
            Vector3 p0 = vertices[neighbors[i]];
            Vector3 p1 = vertices[neighbors[(i + 1) % neighbors.Count]];
            Vector3 p2 = vertices[neighbors[(i + 2) % neighbors.Count]];
            Vector3 cross = Vector3.Cross(p1 - p0, p2 - p1);
            if (Vector3.Dot(cross, normal) < -tolerance)
                return false;
        }

        return true;
    }

    private static void RemoveTrianglesWithVertex(List<TriangleData> triangles, int vertex)
    {
        for (int i = triangles.Count - 1; i >= 0; i--)
        {
            if (triangles[i].Contains(vertex))
                triangles.RemoveAt(i);
        }
    }

    private static void TriangulatePolygon(List<int> neighbors, List<TriangleData> triangles, Vector3[] vertices, int subMesh)
    {
        if (neighbors.Count < 3) return;

        Vector3 normal = Vector3.zero;
        for (int i = 0; i < neighbors.Count; i++)
        {
            Vector3 p0 = vertices[neighbors[i]];
            Vector3 p1 = vertices[neighbors[(i + 1) % neighbors.Count]];
            Vector3 p2 = vertices[neighbors[(i + 2) % neighbors.Count]];
            Vector3 cross = Vector3.Cross(p1 - p0, p2 - p1);
            if (cross.sqrMagnitude > 1e-12f)
            {
                normal = cross.normalized;
                break;
            }
        }

        int anchor = neighbors[0];
        for (int i = 1; i < neighbors.Count - 1; i++)
        {
            int b = neighbors[i];
            int c = neighbors[i + 1];
            Vector3 pa = vertices[anchor];
            Vector3 pb = vertices[b];
            Vector3 pc = vertices[c];
            Vector3 triNormal = Vector3.Cross(pb - pa, pc - pa);
            if (Vector3.Dot(triNormal, normal) < 0f)
            {
                int temp = b;
                b = c;
                c = temp;
            }
            triangles.Add(new TriangleData { a = anchor, b = b, c = c, subMesh = subMesh });
        }
    }

    private static void RemoveUnusedVertices(List<TriangleData> triangles, bool[] removed)
    {
        var used = new bool[removed.Length];
        for (int i = 0; i < triangles.Count; i++)
        {
            TriangleData tri = triangles[i];
            used[tri.a] = true;
            used[tri.b] = true;
            used[tri.c] = true;
        }
        for (int i = 0; i < removed.Length; i++)
            if (!used[i]) removed[i] = true;
    }

    private static List<Vector3> Remap(List<Vector3> source, int[] remap)
    {
        var result = new List<Vector3>(remap.Length);
        if (source == null || source.Count != remap.Length) return result;
        for (int i = 0; i < remap.Length; i++)
        {
            if (remap[i] >= 0)
                result.Add(source[i]);
        }
        return result;
    }

    private static List<BlendShape> CaptureBlendShapes(Mesh mesh)
    {
        var blendShapes = new List<BlendShape>();
        int vertexCount = mesh.vertexCount;
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
        return blendShapes;
    }

    private class Vector3EqualityComparer : IEqualityComparer<Vector3>
    {
        private readonly float tolerance;
        public Vector3EqualityComparer(float tolerance) { this.tolerance = tolerance; }
        public bool Equals(Vector3 a, Vector3 b)
        {
            return Mathf.Abs(a.x - b.x) <= tolerance &&
                   Mathf.Abs(a.y - b.y) <= tolerance &&
                   Mathf.Abs(a.z - b.z) <= tolerance;
        }
        public int GetHashCode(Vector3 v)
        {
            int hx = Mathf.RoundToInt(v.x / tolerance);
            int hy = Mathf.RoundToInt(v.y / tolerance);
            int hz = Mathf.RoundToInt(v.z / tolerance);
            int hash = 17;
            hash = hash * 31 + hx;
            hash = hash * 31 + hy;
            hash = hash * 31 + hz;
            return hash;
        }
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

    private static Mesh SubdivideMesh(Mesh mesh, bool[] vertexMask, out bool[] newVertexMask)
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
        List<bool> maskList = null;
        if (vertexMask != null)
            maskList = new List<bool>(vertexMask);

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
            if (maskList != null)
            {
                bool inside = vertexMask[i0] || vertexMask[i1];
                maskList.Add(inside);
            }

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
            if (vertexMask == null)
            {
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
            else
            {
                for (int i = 0; i < triangles.Length; i += 3)
                {
                    int v0 = triangles[i];
                    int v1 = triangles[i + 1];
                    int v2 = triangles[i + 2];

                    bool e01 = vertexMask[v0] || vertexMask[v1];
                    bool e12 = vertexMask[v1] || vertexMask[v2];
                    bool e20 = vertexMask[v2] || vertexMask[v0];
                    int flagged = (e01 ? 1 : 0) + (e12 ? 1 : 0) + (e20 ? 1 : 0);

                    if (flagged == 0)
                    {
                        list.Add(v0); list.Add(v1); list.Add(v2);
                        continue;
                    }

                    if (flagged == 3)
                    {
                        int m0 = getMidpoint(v0, v1);
                        int m1 = getMidpoint(v1, v2);
                        int m2 = getMidpoint(v2, v0);

                        list.Add(v0); list.Add(m0); list.Add(m2);
                        list.Add(v1); list.Add(m1); list.Add(m0);
                        list.Add(v2); list.Add(m2); list.Add(m1);
                        list.Add(m0); list.Add(m1); list.Add(m2);
                        continue;
                    }

                    if (flagged == 1)
                    {
                        if (e01)
                        {
                            int m0 = getMidpoint(v0, v1);
                            list.Add(v0); list.Add(m0); list.Add(v2);
                            list.Add(m0); list.Add(v1); list.Add(v2);
                        }
                        else if (e12)
                        {
                            int m1 = getMidpoint(v1, v2);
                            list.Add(v1); list.Add(m1); list.Add(v0);
                            list.Add(m1); list.Add(v2); list.Add(v0);
                        }
                        else if (e20)
                        {
                            int m2 = getMidpoint(v2, v0);
                            list.Add(v2); list.Add(m2); list.Add(v1);
                            list.Add(m2); list.Add(v0); list.Add(v1);
                        }
                        continue;
                    }

                    // flagged == 2 -> exactly one vertex inside the bounds
                    if (!e20)
                    {
                        int m0 = getMidpoint(v0, v1);
                        int m1 = getMidpoint(v1, v2);
                        list.Add(v1); list.Add(m1); list.Add(m0);
                        list.Add(m0); list.Add(m1); list.Add(v2);
                        list.Add(v0); list.Add(m0); list.Add(v2);
                    }
                    else if (!e01)
                    {
                        int m1 = getMidpoint(v1, v2);
                        int m2 = getMidpoint(v2, v0);
                        list.Add(v2); list.Add(m2); list.Add(m1);
                        list.Add(m1); list.Add(m2); list.Add(v0);
                        list.Add(v1); list.Add(m1); list.Add(v0);
                    }
                    else // !e12
                    {
                        int m2 = getMidpoint(v2, v0);
                        int m0 = getMidpoint(v0, v1);
                        list.Add(v0); list.Add(m0); list.Add(m2);
                        list.Add(m2); list.Add(m0); list.Add(v1);
                        list.Add(v2); list.Add(m2); list.Add(v1);
                    }
                }
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
        if (maskList != null)
            newVertexMask = maskList.ToArray();
        else
            newVertexMask = null;
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

    private static BoneWeight AverageBoneWeights(IList<BoneWeight> weights)
    {
        Dictionary<int, float> dict = new Dictionary<int, float>();
        int count = 0;
        foreach (var w in weights)
        {
            count++;
            AddBone(ref dict, w.boneIndex0, w.weight0);
            AddBone(ref dict, w.boneIndex1, w.weight1);
            AddBone(ref dict, w.boneIndex2, w.weight2);
            AddBone(ref dict, w.boneIndex3, w.weight3);
        }
        if (count == 0) return new BoneWeight();

        List<int> keys = new List<int>(dict.Keys);
        for (int i = 0; i < keys.Count; i++)
            dict[keys[i]] /= count;

        var ordered = dict.OrderByDescending(kv => kv.Value).Take(4).ToArray();
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
    private bool restrictToBounds;
    private Vector3 boundsCenter;
    private Vector3 boundsSize = Vector3.one;
    private static readonly Color BoundsFillColor = new Color(1f, 0.6f, 0f, 0.15f);
    private static readonly Color BoundsOutlineColor = new Color(1f, 0.6f, 0f, 0.6f);

    [MenuItem("yussy/Subdivide Skinned Meshes")]
    private static void ShowWindow()
    {
        GetWindow<MeshMultiWindow>("Subdivide Meshes");
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
            int predictedVertices = MeshMulti.PredictSubdividedVertexCount(mesh, (restrictToBounds && boundsValid) ? mask : null);
            int predictedTriangles = MeshMulti.PredictSubdividedTriangleCount(mesh, (restrictToBounds && boundsValid) ? mask : null);
            string label = string.Format(
                "Vertices: {0} → {1}, Triangles: {2} → {3}",
                mesh.vertexCount,
                predictedVertices,
                mesh.triangles.Length / 3,
                predictedTriangles);
            if (!inRange && restrictToBounds)
            {
                label += "（範囲外）";
            }
            else if (restrictToBounds && boundsValid)
            {
                int insideCount = 0;
                if (mask != null)
                {
                    for (int v = 0; v < mask.Length; v++)
                        if (mask[v]) insideCount++;
                }
                label += string.Format("（範囲内の頂点 {0} 個）", insideCount);
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

        smoothAppearance = EditorGUILayout.Toggle("見た目をなめらかにする(実験的)", smoothAppearance);
        if (smoothAppearance)
            smoothIterations = EditorGUILayout.IntSlider("スムージング強度", smoothIterations, 1, 50);

        if (GUILayout.Button("Subdivide"))
        {
            if (restrictToBounds && !boundsValid)
            {
                EditorUtility.DisplayDialog("範囲が無効です", "範囲サイズのいずれかがゼロのため、細分化を実行できません。サイズを調整してください。", "OK");
                return;
            }

            Bounds? bounds = null;
            if (restrictToBounds && boundsValid)
                bounds = new Bounds(boundsCenter, boundsSize);
            MeshMulti.SubdivideSelected(selected, smoothAppearance, smoothIterations, bounds);
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
