using System;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

internal sealed class ArapMeshSimplifier
{
    // 改良: トポロジ判定を厳しくし、極端な潰れや法線反転を防ぐ
    private const float MinimumTriangleArea = 1e-8f;
    private const float NormalAlignmentThreshold = 0.0f;
    private const float CotangentEpsilon = 1e-6f;
    private const float PolarTolerance = 1e-6f;
    private const int PolarIterations = 10;
    // 改良: ARAP の反復と収束閾値を強化して形状復元を優先
    private const int MaxArapIterations = 30;
    private const int MaxConjugateGradientIterations = 512;
    private const float ConjugateGradientTolerance = 1e-8f;
    private const float BoneWeightCompatibilityThreshold = 0.1f;
    // 改良: 元座標との差を強く抑制するためのペナルティ係数
    private const float OriginalPositionPenaltyWeight = 250f;

    private readonly Mesh sourceMesh;
    private readonly bool[] vertexMask;
    private readonly int? randomSeed;

    private readonly Vector3[] sourceVertices;
    private readonly Vector3[] sourceNormals;
    private readonly Vector4[] sourceTangents;
    private readonly Color[] sourceColors;
    private readonly BoneWeight[] sourceBoneWeights;
    private readonly Vector2[][] sourceUVs;

    private VertexData[] vertices;
    private List<TriangleData> triangles;
    private List<int>[] submeshTriangleIndices;
    private PriorityQueue<EdgeCandidate> edgeQueue;
    private int activeVertexCount;
    private int candidateVertexCount;
    private System.Random random;
    private int candidateTieBreakerCounter;

    private VertexData[] baseVertices;
    private List<TriangleData> baseTriangles;
    private List<int>[] baseSubmeshTriangleIndices;
    private int baseCandidateVertexCount;
    private int baseActiveVertexCount;
    private bool baseInitialized;

    public struct Result
    {
        public Mesh Mesh;
        public bool RemovedTriangles;
    }

    private sealed class VertexData
    {
        public Vector3 Position;
        public Vector3 RestPosition;
        public Vector3 NormalSum;
        public Vector4 TangentSum;
        public Color ColorSum;
        public Vector2[] UVSum;
        public int[] UVSampleCount;
        public Dictionary<int, float> BoneWeights;
        public HashSet<int> Neighbors = new HashSet<int>();
        public HashSet<int> IncidentTriangles = new HashSet<int>();
        public List<int> SourceIndices = new List<int>();
        public int AggregateWeight = 1;
        public Vector3 OriginalPositionSum;
        public float OriginalPositionSquaredSum;
        public bool Removed;
        public bool Locked;
        public bool Candidate;
        public SymmetricMatrix Quadric;
        public int Revision;
    }

    private struct TriangleData
    {
        public int Submesh;
        public int A;
        public int B;
        public int C;
        public bool Removed;
        public Vector3 RestNormal;
    }

    private struct TriangleInfo
    {
        public int A;
        public int B;
        public int C;
        public bool Removed;
    }

    [BurstCompile]
    private struct TriangleQuadricJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<TriangleInfo> Triangles;
        [ReadOnly] public NativeArray<float3> RestPositions;
        public float MinimumTriangleArea;
        public NativeArray<SymmetricMatrix> Results;

        public void Execute(int index)
        {
            var tri = Triangles[index];
            if (tri.Removed)
            {
                Results[index] = SymmetricMatrix.Zero;
                return;
            }

            float3 pa = RestPositions[tri.A];
            float3 pb = RestPositions[tri.B];
            float3 pc = RestPositions[tri.C];
            float3 ab = pb - pa;
            float3 ac = pc - pa;
            float3 cross = math.cross(ab, ac);
            float area = math.length(cross) * 0.5f;
            if (area < MinimumTriangleArea)
            {
                Results[index] = SymmetricMatrix.Zero;
                return;
            }

            float3 normal = math.normalize(cross);
            double nx = normal.x;
            double ny = normal.y;
            double nz = normal.z;
            double d = -(nx * pa.x + ny * pa.y + nz * pa.z);
            var quadric = SymmetricMatrix.FromPlane(new double3(nx, ny, nz), d) * area;
            Results[index] = quadric;
        }
    }

    private struct EdgeCandidate : IComparable<EdgeCandidate>
    {
        public int A;
        public int B;
        public float Cost;
        public Vector3 OptimalPosition;
        public int RevisionA;
        public int RevisionB;
        public int TieBreaker;

        public int CompareTo(EdgeCandidate other)
        {
            int costComparison = Cost.CompareTo(other.Cost);
            if (costComparison != 0)
                return costComparison;
            int tieComparison = TieBreaker.CompareTo(other.TieBreaker);
            if (tieComparison != 0)
                return tieComparison;
            int aComparison = A.CompareTo(other.A);
            if (aComparison != 0)
                return aComparison;
            return B.CompareTo(other.B);
        }
    }

    private struct SymmetricMatrix
    {
        public double m00, m01, m02, m03;
        public double m11, m12, m13;
        public double m22, m23;
        public double m33;

        public static SymmetricMatrix Zero => new SymmetricMatrix();

        public static SymmetricMatrix FromPlane(Vector3 normal, float d)
        {
            return FromPlane(new double3(normal.x, normal.y, normal.z), d);
        }

        public static SymmetricMatrix FromPlane(double3 normal, double d)
        {
            double nx = normal.x;
            double ny = normal.y;
            double nz = normal.z;
            double nd = d;
            return new SymmetricMatrix
            {
                m00 = nx * nx,
                m01 = nx * ny,
                m02 = nx * nz,
                m03 = nx * nd,
                m11 = ny * ny,
                m12 = ny * nz,
                m13 = ny * nd,
                m22 = nz * nz,
                m23 = nz * nd,
                m33 = nd * nd
            };
        }

        public static SymmetricMatrix operator +(SymmetricMatrix a, SymmetricMatrix b)
        {
            return new SymmetricMatrix
            {
                m00 = a.m00 + b.m00,
                m01 = a.m01 + b.m01,
                m02 = a.m02 + b.m02,
                m03 = a.m03 + b.m03,
                m11 = a.m11 + b.m11,
                m12 = a.m12 + b.m12,
                m13 = a.m13 + b.m13,
                m22 = a.m22 + b.m22,
                m23 = a.m23 + b.m23,
                m33 = a.m33 + b.m33
            };
        }

        public static SymmetricMatrix operator *(SymmetricMatrix m, double scalar)
        {
            return new SymmetricMatrix
            {
                m00 = m.m00 * scalar,
                m01 = m.m01 * scalar,
                m02 = m.m02 * scalar,
                m03 = m.m03 * scalar,
                m11 = m.m11 * scalar,
                m12 = m.m12 * scalar,
                m13 = m.m13 * scalar,
                m22 = m.m22 * scalar,
                m23 = m.m23 * scalar,
                m33 = m.m33 * scalar
            };
        }

        public static SymmetricMatrix operator *(double scalar, SymmetricMatrix m) => m * scalar;

        public double Evaluate(Vector3 v)
        {
            double x = v.x;
            double y = v.y;
            double z = v.z;
            return m00 * x * x + 2.0 * m01 * x * y + 2.0 * m02 * x * z + 2.0 * m03 * x +
                   m11 * y * y + 2.0 * m12 * y * z + 2.0 * m13 * y +
                   m22 * z * z + 2.0 * m23 * z + m33;
        }

        public bool TryGetOptimalPosition(out Vector3 position)
        {
            double det = m00 * (m11 * m22 - m12 * m12) -
                         m01 * (m01 * m22 - m12 * m02) +
                         m02 * (m01 * m12 - m11 * m02);
            if (Math.Abs(det) < 1e-12)
            {
                position = Vector3.zero;
                return false;
            }

            double invDet = 1.0 / det;
            double bx = -m03;
            double by = -m13;
            double bz = -m23;

            double x = (bx * (m11 * m22 - m12 * m12) +
                        by * (m02 * m12 - m01 * m22) +
                        bz * (m01 * m12 - m02 * m11)) * invDet;
            double y = (bx * (m12 * m02 - m01 * m22) +
                        by * (m00 * m22 - m02 * m02) +
                        bz * (m01 * m02 - m00 * m12)) * invDet;
            double z = (bx * (m01 * m12 - m11 * m02) +
                        by * (m01 * m02 - m00 * m12) +
                        bz * (m00 * m11 - m01 * m01)) * invDet;

            position = new Vector3((float)x, (float)y, (float)z);
            return true;
        }
    }

    private sealed class PriorityQueue<T> where T : IComparable<T>
    {
        private readonly List<T> heap = new List<T>();

        public int Count => heap.Count;

        public void Enqueue(T value)
        {
            heap.Add(value);
            SiftUp(heap.Count - 1);
        }

        public bool TryDequeue(out T value)
        {
            if (heap.Count == 0)
            {
                value = default;
                return false;
            }

            value = heap[0];
            int last = heap.Count - 1;
            heap[0] = heap[last];
            heap.RemoveAt(last);
            if (heap.Count > 0)
                SiftDown(0);
            return true;
        }

        private void SiftUp(int index)
        {
            while (index > 0)
            {
                int parent = (index - 1) >> 1;
                if (heap[index].CompareTo(heap[parent]) >= 0)
                    break;
                (heap[index], heap[parent]) = (heap[parent], heap[index]);
                index = parent;
            }
        }

        private void SiftDown(int index)
        {
            int count = heap.Count;
            while (true)
            {
                int left = (index << 1) + 1;
                int right = left + 1;
                int smallest = index;
                if (left < count && heap[left].CompareTo(heap[smallest]) < 0)
                    smallest = left;
                if (right < count && heap[right].CompareTo(heap[smallest]) < 0)
                    smallest = right;
                if (smallest == index)
                    break;
                (heap[index], heap[smallest]) = (heap[smallest], heap[index]);
                index = smallest;
            }
        }
    }

    private struct Matrix3x3
    {
        public float m00, m01, m02;
        public float m10, m11, m12;
        public float m20, m21, m22;

        public static Matrix3x3 Zero => new Matrix3x3();
        public static Matrix3x3 Identity => new Matrix3x3 { m00 = 1f, m11 = 1f, m22 = 1f };

        public static Matrix3x3 operator +(Matrix3x3 a, Matrix3x3 b)
        {
            return new Matrix3x3
            {
                m00 = a.m00 + b.m00,
                m01 = a.m01 + b.m01,
                m02 = a.m02 + b.m02,
                m10 = a.m10 + b.m10,
                m11 = a.m11 + b.m11,
                m12 = a.m12 + b.m12,
                m20 = a.m20 + b.m20,
                m21 = a.m21 + b.m21,
                m22 = a.m22 + b.m22
            };
        }

        public static Matrix3x3 operator -(Matrix3x3 a, Matrix3x3 b)
        {
            return new Matrix3x3
            {
                m00 = a.m00 - b.m00,
                m01 = a.m01 - b.m01,
                m02 = a.m02 - b.m02,
                m10 = a.m10 - b.m10,
                m11 = a.m11 - b.m11,
                m12 = a.m12 - b.m12,
                m20 = a.m20 - b.m20,
                m21 = a.m21 - b.m21,
                m22 = a.m22 - b.m22
            };
        }

        public static Matrix3x3 operator *(Matrix3x3 a, float scalar)
        {
            return new Matrix3x3
            {
                m00 = a.m00 * scalar,
                m01 = a.m01 * scalar,
                m02 = a.m02 * scalar,
                m10 = a.m10 * scalar,
                m11 = a.m11 * scalar,
                m12 = a.m12 * scalar,
                m20 = a.m20 * scalar,
                m21 = a.m21 * scalar,
                m22 = a.m22 * scalar
            };
        }

        public static Matrix3x3 OuterProduct(Vector3 a, Vector3 b)
        {
            return new Matrix3x3
            {
                m00 = a.x * b.x,
                m01 = a.x * b.y,
                m02 = a.x * b.z,
                m10 = a.y * b.x,
                m11 = a.y * b.y,
                m12 = a.y * b.z,
                m20 = a.z * b.x,
                m21 = a.z * b.y,
                m22 = a.z * b.z
            };
        }

        public Matrix3x3 Transpose()
        {
            return new Matrix3x3
            {
                m00 = m00,
                m01 = m10,
                m02 = m20,
                m10 = m01,
                m11 = m11,
                m12 = m21,
                m20 = m02,
                m21 = m12,
                m22 = m22
            };
        }

        public float Determinant()
        {
            return m00 * (m11 * m22 - m12 * m21) -
                   m01 * (m10 * m22 - m12 * m20) +
                   m02 * (m10 * m21 - m11 * m20);
        }

        public Matrix3x3 Inverse()
        {
            float det = Determinant();
            if (Mathf.Abs(det) < 1e-12f)
                return Identity;
            float invDet = 1f / det;
            return new Matrix3x3
            {
                m00 = (m11 * m22 - m12 * m21) * invDet,
                m01 = (m02 * m21 - m01 * m22) * invDet,
                m02 = (m01 * m12 - m02 * m11) * invDet,
                m10 = (m12 * m20 - m10 * m22) * invDet,
                m11 = (m00 * m22 - m02 * m20) * invDet,
                m12 = (m02 * m10 - m00 * m12) * invDet,
                m20 = (m10 * m21 - m11 * m20) * invDet,
                m21 = (m01 * m20 - m00 * m21) * invDet,
                m22 = (m00 * m11 - m01 * m10) * invDet
            };
        }

        public float FrobeniusNorm()
        {
            return Mathf.Sqrt(m00 * m00 + m01 * m01 + m02 * m02 +
                              m10 * m10 + m11 * m11 + m12 * m12 +
                              m20 * m20 + m21 * m21 + m22 * m22);
        }

        public Vector3 Multiply(Vector3 v)
        {
            return new Vector3(
                m00 * v.x + m01 * v.y + m02 * v.z,
                m10 * v.x + m11 * v.y + m12 * v.z,
                m20 * v.x + m21 * v.y + m22 * v.z
            );
        }
    }

    private sealed class LinearSystemRow
    {
        public float Diagonal;
        public List<int> Neighbors = new List<int>();
        public List<float> Weights = new List<float>();
    }

    public ArapMeshSimplifier(Mesh mesh, bool[] mask, int? seed = null)
    {
        sourceMesh = mesh;
        if (mesh == null)
            throw new ArgumentNullException(nameof(mesh));

        if (mask != null && mask.Length != mesh.vertexCount)
        {
            Debug.LogWarning($"Ignoring vertex mask for '{mesh.name}' because its length ({mask.Length}) does not match the mesh vertex count ({mesh.vertexCount}).");
            vertexMask = null;
        }
        else
        {
            vertexMask = mask;
        }
        randomSeed = seed;
        sourceVertices = mesh.vertices;
        sourceNormals = mesh.normals;
        sourceTangents = mesh.tangents;
        sourceColors = mesh.colors;
        sourceBoneWeights = mesh.boneWeights;
        sourceUVs = GetUVChannels(mesh);
    }

    public Result? Simplify(int targetCandidateCount)
    {
        EnsureBaseData();
        ResetWorkingState();
        if (targetCandidateCount < 3)
            targetCandidateCount = 3;
        if (targetCandidateCount >= candidateVertexCount)
            return null;

        BuildTriangleQuadrics();
        InitializeEdgeQueue();

        bool collapsed = PerformEdgeCollapses(targetCandidateCount);
        if (!collapsed)
            return null;

        RunArapRelaxation();

        Mesh mesh = BuildMesh(out bool removedTriangles);
        return new Result { Mesh = mesh, RemovedTriangles = removedTriangles };
    }

    private void EnsureBaseData()
    {
        if (baseInitialized)
            return;

        int vertexCount = sourceVertices.Length;
        baseVertices = new VertexData[vertexCount];
        baseCandidateVertexCount = 0;
        baseActiveVertexCount = vertexCount;

        for (int i = 0; i < vertexCount; i++)
        {
            bool candidate = vertexMask == null || vertexMask[i];
            baseVertices[i] = CreateInitialVertex(i, candidate);
            if (candidate)
                baseCandidateVertexCount++;
        }

        baseTriangles = new List<TriangleData>();
        int subMeshCount = Mathf.Max(1, sourceMesh.subMeshCount);
        baseSubmeshTriangleIndices = new List<int>[subMeshCount];
        for (int sub = 0; sub < subMeshCount; sub++)
        {
            baseSubmeshTriangleIndices[sub] = new List<int>();
            int[] indices = (sourceMesh.subMeshCount == 0 && sub == 0) ? sourceMesh.triangles : sourceMesh.GetTriangles(sub);
            for (int i = 0; i < indices.Length; i += 3)
            {
                int a = indices[i];
                int b = indices[i + 1];
                int c = indices[i + 2];
                if (a == b || b == c || c == a)
                    continue;

                var restNormal = Vector3.Cross(sourceVertices[b] - sourceVertices[a], sourceVertices[c] - sourceVertices[a]);
                if (restNormal.sqrMagnitude < MinimumTriangleArea)
                    continue;

                restNormal.Normalize();
                var tri = new TriangleData
                {
                    Submesh = sub,
                    A = a,
                    B = b,
                    C = c,
                    Removed = false,
                    RestNormal = restNormal
                };
                int triIndex = baseTriangles.Count;
                baseTriangles.Add(tri);
                baseSubmeshTriangleIndices[sub].Add(triIndex);

                baseVertices[a].Neighbors.Add(b);
                baseVertices[a].Neighbors.Add(c);
                baseVertices[b].Neighbors.Add(a);
                baseVertices[b].Neighbors.Add(c);
                baseVertices[c].Neighbors.Add(a);
                baseVertices[c].Neighbors.Add(b);

                baseVertices[a].IncidentTriangles.Add(triIndex);
                baseVertices[b].IncidentTriangles.Add(triIndex);
                baseVertices[c].IncidentTriangles.Add(triIndex);
            }
        }

        baseInitialized = true;
    }

    private VertexData CreateInitialVertex(int index, bool candidate)
    {
        int vertexCount = sourceVertices.Length;
        var data = new VertexData
        {
            Position = sourceVertices[index],
            RestPosition = sourceVertices[index],
            NormalSum = (sourceNormals != null && sourceNormals.Length == vertexCount) ? sourceNormals[index] : Vector3.zero,
            TangentSum = (sourceTangents != null && sourceTangents.Length == vertexCount) ? sourceTangents[index] : Vector4.zero,
            ColorSum = (sourceColors != null && sourceColors.Length == vertexCount) ? sourceColors[index] : Color.black,
            UVSum = new Vector2[sourceUVs.Length],
            UVSampleCount = new int[sourceUVs.Length],
            BoneWeights = (sourceBoneWeights != null && sourceBoneWeights.Length == vertexCount) ? new Dictionary<int, float>(4) : null,
            Locked = !candidate,
            Candidate = candidate,
            AggregateWeight = 1,
            OriginalPositionSum = sourceVertices[index],
            OriginalPositionSquaredSum = sourceVertices[index].sqrMagnitude,
            Quadric = SymmetricMatrix.Zero,
            Revision = 0
        };

        if (data.BoneWeights != null)
        {
            var bw = sourceBoneWeights[index];
            AddBoneWeight(data.BoneWeights, bw.boneIndex0, bw.weight0);
            AddBoneWeight(data.BoneWeights, bw.boneIndex1, bw.weight1);
            AddBoneWeight(data.BoneWeights, bw.boneIndex2, bw.weight2);
            AddBoneWeight(data.BoneWeights, bw.boneIndex3, bw.weight3);
        }

        for (int channel = 0; channel < sourceUVs.Length; channel++)
        {
            var uv = sourceUVs[channel];
            if (uv != null && uv.Length == vertexCount)
            {
                data.UVSum[channel] = uv[index];
                data.UVSampleCount[channel] = 1;
            }
        }

        data.SourceIndices.Add(index);
        return data;
    }

    private void ResetWorkingState()
    {
        if (!baseInitialized)
            EnsureBaseData();

        int vertexCount = baseVertices.Length;
        vertices = new VertexData[vertexCount];
        for (int i = 0; i < vertexCount; i++)
        {
            vertices[i] = CloneVertex(baseVertices[i]);
        }

        triangles = new List<TriangleData>(baseTriangles.Count);
        for (int i = 0; i < baseTriangles.Count; i++)
            triangles.Add(baseTriangles[i]);

        submeshTriangleIndices = new List<int>[baseSubmeshTriangleIndices.Length];
        for (int sub = 0; sub < baseSubmeshTriangleIndices.Length; sub++)
            submeshTriangleIndices[sub] = new List<int>(baseSubmeshTriangleIndices[sub]);

        candidateVertexCount = baseCandidateVertexCount;
        activeVertexCount = baseActiveVertexCount;
        candidateTieBreakerCounter = 0;
        random = randomSeed.HasValue ? new System.Random(randomSeed.Value) : null;
    }

    private static VertexData CloneVertex(VertexData source)
    {
        if (source == null)
            return null;

        var clone = new VertexData
        {
            Position = source.Position,
            RestPosition = source.RestPosition,
            NormalSum = source.NormalSum,
            TangentSum = source.TangentSum,
            ColorSum = source.ColorSum,
            UVSum = source.UVSum != null ? (Vector2[])source.UVSum.Clone() : null,
            UVSampleCount = source.UVSampleCount != null ? (int[])source.UVSampleCount.Clone() : null,
            BoneWeights = source.BoneWeights != null ? new Dictionary<int, float>(source.BoneWeights) : null,
            Locked = source.Locked,
            Candidate = source.Candidate,
            AggregateWeight = source.AggregateWeight,
            OriginalPositionSum = source.OriginalPositionSum,
            OriginalPositionSquaredSum = source.OriginalPositionSquaredSum,
            Quadric = SymmetricMatrix.Zero,
            Revision = 0,
            Removed = source.Removed
        };

        if (clone.UVSum == null)
            clone.UVSum = Array.Empty<Vector2>();
        if (clone.UVSampleCount == null)
            clone.UVSampleCount = Array.Empty<int>();

        clone.Neighbors = new HashSet<int>(source.Neighbors);
        clone.IncidentTriangles = new HashSet<int>(source.IncidentTriangles);
        clone.SourceIndices = new List<int>(source.SourceIndices);
        return clone;
    }

    private void BuildTriangleQuadrics()
    {
        if (triangles == null || triangles.Count == 0)
            return;

        var triangleInfos = new NativeArray<TriangleInfo>(triangles.Count, Allocator.TempJob);
        var restPositions = new NativeArray<float3>(vertices.Length, Allocator.TempJob);
        var results = new NativeArray<SymmetricMatrix>(triangles.Count, Allocator.TempJob);

        try
        {
            for (int i = 0; i < triangles.Count; i++)
            {
                var tri = triangles[i];
                triangleInfos[i] = new TriangleInfo
                {
                    A = tri.A,
                    B = tri.B,
                    C = tri.C,
                    Removed = tri.Removed
                };
            }

            for (int i = 0; i < vertices.Length; i++)
            {
                var vertex = vertices[i];
                if (vertex != null)
                {
                    restPositions[i] = new float3(vertex.RestPosition.x, vertex.RestPosition.y, vertex.RestPosition.z);
                    vertex.Quadric = SymmetricMatrix.Zero;
                }
                else
                {
                    restPositions[i] = float3.zero;
                }
            }

            var job = new TriangleQuadricJob
            {
                Triangles = triangleInfos,
                RestPositions = restPositions,
                MinimumTriangleArea = MinimumTriangleArea,
                Results = results
            };

            JobHandle handle = job.Schedule(triangleInfos.Length, 64);
            handle.Complete();

            for (int i = 0; i < triangles.Count; i++)
            {
                if (triangles[i].Removed)
                    continue;
                var tri = triangles[i];
                var quadric = results[i];
                vertices[tri.A].Quadric = vertices[tri.A].Quadric + quadric;
                vertices[tri.B].Quadric = vertices[tri.B].Quadric + quadric;
                vertices[tri.C].Quadric = vertices[tri.C].Quadric + quadric;
            }
        }
        finally
        {
            if (triangleInfos.IsCreated)
                triangleInfos.Dispose();
            if (restPositions.IsCreated)
                restPositions.Dispose();
            if (results.IsCreated)
                results.Dispose();
        }
    }

    private void InitializeEdgeQueue()
    {
        edgeQueue = new PriorityQueue<EdgeCandidate>();
        var seen = new HashSet<long>();
        for (int i = 0; i < vertices.Length; i++)
        {
            var vi = vertices[i];
            if (vi == null || vi.Removed)
                continue;
            foreach (int neighbor in vi.Neighbors)
            {
                if (neighbor <= i)
                    continue;
                long key = ((long)i << 32) | (uint)neighbor;
                if (seen.Contains(key))
                    continue;
                seen.Add(key);
                if (TryBuildEdgeCandidate(i, neighbor, out var candidate))
                    edgeQueue.Enqueue(candidate);
            }
        }
    }

    private bool PerformEdgeCollapses(int targetCandidateCount)
    {
        bool collapsedAny = false;
        while (candidateVertexCount > targetCandidateCount)
        {
            if (!edgeQueue.TryDequeue(out var candidate))
                break;

            if (!ValidateCandidate(candidate))
                continue;

            if (CollapseEdge(candidate))
            {
                collapsedAny = true;
            }
        }
        return collapsedAny;
    }

    private bool ValidateCandidate(EdgeCandidate candidate)
    {
        if (candidate.A < 0 || candidate.A >= vertices.Length || candidate.B < 0 || candidate.B >= vertices.Length)
            return false;
        var va = vertices[candidate.A];
        var vb = vertices[candidate.B];
        if (va == null || vb == null || va.Removed || vb.Removed)
            return false;
        if (va.Revision != candidate.RevisionA || vb.Revision != candidate.RevisionB)
            return false;
        if (va.Locked && vb.Locked)
            return false;
        return true;
    }

    private bool TryBuildEdgeCandidate(int a, int b, out EdgeCandidate candidate)
    {
        var va = vertices[a];
        var vb = vertices[b];
        candidate = default;
        if (va == null || vb == null || va.Removed || vb.Removed)
            return false;
        if (va.Locked && vb.Locked)
            return false;

        if (!AreBoneWeightsCompatible(va, vb))
            return false;

        SymmetricMatrix quadric = va.Quadric + vb.Quadric;

        // 改良: 元座標からの乖離を強く抑えるコスト評価
        Vector3 bestPosition = Vector3.zero;
        float bestCost = float.PositiveInfinity;

        void ConsiderPosition(Vector3 position)
        {
            if (!IsValidVector(position))
                return;
            float evaluatedCost = EvaluateEdgeCost(quadric, position, va, vb);
            if (evaluatedCost < bestCost)
            {
                bestCost = evaluatedCost;
                bestPosition = position;
            }
        }

        if (quadric.TryGetOptimalPosition(out Vector3 optimal) && IsValidVector(optimal))
            ConsiderPosition(optimal);
        ConsiderPosition((va.Position + vb.Position) * 0.5f);
        ConsiderPosition(va.Position);
        ConsiderPosition(vb.Position);

        if (!IsValidVector(bestPosition) || float.IsInfinity(bestCost))
            return false;

        candidate = new EdgeCandidate
        {
            A = a,
            B = b,
            Cost = bestCost,
            OptimalPosition = bestPosition,
            RevisionA = va.Revision,
            RevisionB = vb.Revision,
            TieBreaker = NextTieBreakerValue()
        };
        return true;
    }

    private bool CollapseEdge(EdgeCandidate candidate)
    {
        int a = candidate.A;
        int b = candidate.B;
        var va = vertices[a];
        var vb = vertices[b];
        bool targetIsA = true;
        if (vb.Locked && !va.Locked)
            targetIsA = false;
        else if (!va.Locked && !vb.Locked)
            targetIsA = va.Candidate || !vb.Candidate;

        int targetIndex = targetIsA ? a : b;
        int sourceIndex = targetIsA ? b : a;
        var target = vertices[targetIndex];
        var source = vertices[sourceIndex];

        Vector3 newPosition = candidate.OptimalPosition;
        if (!IsValidVector(newPosition))
            newPosition = Vector3.Lerp(va.Position, vb.Position, 0.5f);
        if (!IsValidVector(newPosition))
            newPosition = target.Position;
        if (target.Locked)
            newPosition = target.Position;
        if (!IsValidVector(newPosition))
            newPosition = source.Position;
        if (!IsValidVector(newPosition))
            newPosition = Vector3.zero;

        if (!IsCollapseTopologicallyValid(targetIndex, sourceIndex, newPosition))
            return false;

        MergeVertices(targetIndex, sourceIndex, newPosition);
        return true;
    }

    private bool AreBoneWeightsCompatible(VertexData a, VertexData b)
    {
        if (a == null || b == null)
            return false;
        if (a.BoneWeights == null || b.BoneWeights == null)
            return true;
        if (a.BoneWeights.Count == 0 || b.BoneWeights.Count == 0)
            return true;

        float totalA = 0f;
        foreach (var kv in a.BoneWeights)
            totalA += kv.Value;

        float totalB = 0f;
        foreach (var kv in b.BoneWeights)
            totalB += kv.Value;

        if (totalA <= 0f || totalB <= 0f)
            return true;

        float shared = 0f;
        foreach (var kv in a.BoneWeights)
        {
            if (!b.BoneWeights.TryGetValue(kv.Key, out float weightB))
                continue;

            float weightA = kv.Value / totalA;
            float normalisedB = weightB / totalB;
            shared += Mathf.Min(weightA, normalisedB);
        }

        return shared >= BoneWeightCompatibilityThreshold;
    }

    private bool IsCollapseTopologicallyValid(int target, int source, Vector3 newPosition)
    {
        var vt = vertices[target];
        var vs = vertices[source];
        var affectedTriangles = new HashSet<int>(vt.IncidentTriangles);
        affectedTriangles.UnionWith(vs.IncidentTriangles);

        foreach (int triIndex in affectedTriangles)
        {
            var tri = triangles[triIndex];
            if (tri.Removed)
                continue;

            bool usesSource = tri.A == source || tri.B == source || tri.C == source;
            bool usesTargetOriginal = tri.A == target || tri.B == target || tri.C == target;
            bool collapsesTriangle = usesSource && usesTargetOriginal;

            int a = tri.A;
            int b = tri.B;
            int c = tri.C;
            if (a == source) a = target;
            if (b == source) b = target;
            if (c == source) c = target;

            if (collapsesTriangle)
                continue;

            if (a == b || b == c || c == a)
                return false;

            Vector3 pa = (a == target) ? vt.RestPosition : vertices[a].RestPosition;
            Vector3 pb = (b == target) ? vt.RestPosition : vertices[b].RestPosition;
            Vector3 pc = (c == target) ? vt.RestPosition : vertices[c].RestPosition;
            Vector3 restNormal = Vector3.Cross(pb - pa, pc - pa);
            if (restNormal.sqrMagnitude < MinimumTriangleArea)
                return false;
            restNormal.Normalize();
            if (Vector3.Dot(restNormal, tri.RestNormal) < NormalAlignmentThreshold)
                return false;

            Vector3 xa = (a == target) ? newPosition : vertices[a].Position;
            Vector3 xb = (b == target) ? newPosition : vertices[b].Position;
            Vector3 xc = (c == target) ? newPosition : vertices[c].Position;
            Vector3 newNormal = Vector3.Cross(xb - xa, xc - xa);
            if (newNormal.sqrMagnitude < MinimumTriangleArea)
                return false;
            newNormal.Normalize();
            if (Vector3.Dot(newNormal, tri.RestNormal) < NormalAlignmentThreshold)
                return false;
        }

        return true;
    }

    private void MergeVertices(int targetIndex, int sourceIndex, Vector3 newPosition)
    {
        var target = vertices[targetIndex];
        var source = vertices[sourceIndex];
        target.Position = newPosition;
        Vector3 newRest = (target.RestPosition * target.AggregateWeight + source.RestPosition * source.AggregateWeight) /
                          Mathf.Max(1, target.AggregateWeight + source.AggregateWeight);
        target.RestPosition = newRest;
        target.AggregateWeight += source.AggregateWeight;
        target.OriginalPositionSum += source.OriginalPositionSum;
        target.OriginalPositionSquaredSum += source.OriginalPositionSquaredSum;
        target.Locked |= source.Locked;
        target.Candidate |= source.Candidate;
        target.Quadric = target.Quadric + source.Quadric;
        target.SourceIndices.AddRange(source.SourceIndices);
        target.NormalSum += source.NormalSum;
        target.TangentSum += source.TangentSum;
        target.ColorSum += source.ColorSum;

        for (int channel = 0; channel < target.UVSum.Length; channel++)
        {
            if (source.UVSampleCount[channel] > 0)
            {
                target.UVSum[channel] += source.UVSum[channel];
                target.UVSampleCount[channel] += source.UVSampleCount[channel];
            }
        }

        if (source.BoneWeights != null)
        {
            if (target.BoneWeights == null)
                target.BoneWeights = new Dictionary<int, float>(source.BoneWeights.Count);
            foreach (var kv in source.BoneWeights)
            {
                if (target.BoneWeights.TryGetValue(kv.Key, out float value))
                    target.BoneWeights[kv.Key] = value + kv.Value;
                else
                    target.BoneWeights.Add(kv.Key, kv.Value);
            }
        }

        var neighborsToUpdate = new HashSet<int>(source.Neighbors);
        foreach (int neighbor in neighborsToUpdate)
        {
            if (neighbor == targetIndex)
                continue;
            vertices[neighbor].Neighbors.Remove(sourceIndex);
            vertices[neighbor].Neighbors.Add(targetIndex);
            target.Neighbors.Add(neighbor);
        }
        target.Neighbors.Remove(sourceIndex);
        target.Neighbors.Remove(targetIndex);

        var triangleSet = new HashSet<int>(source.IncidentTriangles);
        foreach (int triIndex in triangleSet)
        {
            var tri = triangles[triIndex];
            if (tri.Removed)
                continue;
            bool changed = false;
            if (tri.A == sourceIndex)
            {
                tri.A = targetIndex;
                changed = true;
            }
            if (tri.B == sourceIndex)
            {
                tri.B = targetIndex;
                changed = true;
            }
            if (tri.C == sourceIndex)
            {
                tri.C = targetIndex;
                changed = true;
            }

            bool removed = false;
            if (tri.A == tri.B || tri.B == tri.C || tri.C == tri.A)
            {
                tri.Removed = true;
                removed = true;
            }
            else
            {
                Vector3 ra = vertices[tri.A].RestPosition;
                Vector3 rb = vertices[tri.B].RestPosition;
                Vector3 rc = vertices[tri.C].RestPosition;
                Vector3 restNormal = Vector3.Cross(rb - ra, rc - ra);
                if (restNormal.sqrMagnitude < MinimumTriangleArea)
                {
                    tri.Removed = true;
                    removed = true;
                }
                else
                {
                    restNormal.Normalize();
                    tri.RestNormal = restNormal;
                }
            }

            triangles[triIndex] = tri;
            if (!removed && changed)
                target.IncidentTriangles.Add(triIndex);
        }

        var refreshedTriangles = new HashSet<int>(target.IncidentTriangles);
        foreach (int triIndex in refreshedTriangles)
        {
            var tri = triangles[triIndex];
            if (tri.Removed)
                continue;
            Vector3 ra = vertices[tri.A].RestPosition;
            Vector3 rb = vertices[tri.B].RestPosition;
            Vector3 rc = vertices[tri.C].RestPosition;
            Vector3 restNormal = Vector3.Cross(rb - ra, rc - ra);
            if (restNormal.sqrMagnitude < MinimumTriangleArea)
            {
                tri.Removed = true;
            }
            else
            {
                restNormal.Normalize();
                tri.RestNormal = restNormal;
            }
            triangles[triIndex] = tri;
        }

        source.Neighbors.Clear();
        source.IncidentTriangles.Clear();
        source.Removed = true;
        activeVertexCount--;
        if (source.Candidate)
            candidateVertexCount--;

        target.Revision++;

        foreach (int neighbor in target.Neighbors)
        {
            if (neighbor == targetIndex)
                continue;
            vertices[neighbor].Revision++;
            if (TryBuildEdgeCandidate(Math.Min(targetIndex, neighbor), Math.Max(targetIndex, neighbor), out var candidate))
                edgeQueue.Enqueue(candidate);
        }
    }

    private int NextTieBreakerValue()
    {
        if (random != null)
            return random.Next();
        return unchecked(candidateTieBreakerCounter++);
    }

    private void RunArapRelaxation()
    {
        Dictionary<int, float>[] weights = BuildCotangentWeights();
        if (weights == null)
            return;

        Matrix3x3[] rotations = new Matrix3x3[vertices.Length];
        for (int i = 0; i < rotations.Length; i++)
            rotations[i] = Matrix3x3.Identity;

        for (int iteration = 0; iteration < MaxArapIterations; iteration++)
        {
            // Local step
            for (int i = 0; i < vertices.Length; i++)
            {
                var v = vertices[i];
                if (v == null || v.Removed)
                    continue;
                if (weights[i] == null || weights[i].Count == 0)
                {
                    rotations[i] = Matrix3x3.Identity;
                    continue;
                }

                Matrix3x3 covariance = Matrix3x3.Zero;
                foreach (var kv in weights[i])
                {
                    int j = kv.Key;
                    float w = kv.Value;
                    Vector3 pij = vertices[j].RestPosition - v.RestPosition;
                    Vector3 xij = vertices[j].Position - v.Position;
                    covariance += Matrix3x3.OuterProduct(xij, pij) * w;
                }

                rotations[i] = ComputePolarRotation(covariance);
            }

            // Global step
            Vector3[] rhs = new Vector3[vertices.Length];
            for (int i = 0; i < vertices.Length; i++)
            {
                var v = vertices[i];
                if (v == null || v.Removed || weights[i] == null)
                    continue;

                Vector3 bi = Vector3.zero;
                foreach (var kv in weights[i])
                {
                    int j = kv.Key;
                    float w = kv.Value;
                    Vector3 pij = v.RestPosition - vertices[j].RestPosition;
                    Vector3 term = rotations[i].Multiply(pij) + rotations[j].Multiply(pij);
                    bi += 0.5f * w * term;
                }
                rhs[i] = bi;
            }

            SolveGlobalStep(weights, rhs);
        }
    }

    private float EvaluateEdgeCost(SymmetricMatrix quadric, Vector3 position, VertexData va, VertexData vb)
    {
        double quadricCost = quadric.Evaluate(position);
        int combinedWeight = Mathf.Max(va.AggregateWeight + vb.AggregateWeight, 1);
        Vector3 originalSum = va.OriginalPositionSum + vb.OriginalPositionSum;
        float originalSquaredSum = va.OriginalPositionSquaredSum + vb.OriginalPositionSquaredSum;
        float penalty = 0f;
        float positionMagnitude = position.sqrMagnitude;
        float sumSquaredDistance = combinedWeight * positionMagnitude - 2f * Vector3.Dot(position, originalSum) + originalSquaredSum;
        if (combinedWeight > 0)
            penalty = sumSquaredDistance / combinedWeight;

        return (float)quadricCost + OriginalPositionPenaltyWeight * penalty;
    }

    private Dictionary<int, float>[] BuildCotangentWeights()
    {
        if (triangles.Count == 0)
            return null;
        var weights = new Dictionary<int, float>[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            if (vertices[i] != null && !vertices[i].Removed)
                weights[i] = new Dictionary<int, float>();
        }

        foreach (var tri in triangles)
        {
            if (tri.Removed)
                continue;
            var va = vertices[tri.A];
            var vb = vertices[tri.B];
            var vc = vertices[tri.C];
            if (va == null || vb == null || vc == null || va.Removed || vb.Removed || vc.Removed)
                continue;

            Vector3 pa = va.RestPosition;
            Vector3 pb = vb.RestPosition;
            Vector3 pc = vc.RestPosition;

            float cotA = Cotangent(pb - pa, pc - pa);
            float cotB = Cotangent(pa - pb, pc - pb);
            float cotC = Cotangent(pa - pc, pb - pc);

            AddCotangentWeight(weights, tri.B, tri.C, cotA);
            AddCotangentWeight(weights, tri.C, tri.A, cotB);
            AddCotangentWeight(weights, tri.A, tri.B, cotC);
        }

        return weights;
    }

    private void AddCotangentWeight(Dictionary<int, float>[] weights, int i, int j, float cot)
    {
        if (float.IsNaN(cot) || float.IsInfinity(cot))
            return;
        float magnitude = Mathf.Abs(cot);
        if (magnitude < CotangentEpsilon)
            cot = (cot >= 0f ? 1f : -1f) * CotangentEpsilon;
        if (weights[i] == null)
            weights[i] = new Dictionary<int, float>();
        if (weights[j] == null)
            weights[j] = new Dictionary<int, float>();
        if (weights[i].TryGetValue(j, out float existing))
            weights[i][j] = existing + 0.5f * cot;
        else
            weights[i][j] = 0.5f * cot;
        if (weights[j].TryGetValue(i, out float back))
            weights[j][i] = back + 0.5f * cot;
        else
            weights[j][i] = 0.5f * cot;
    }

    private float Cotangent(Vector3 u, Vector3 v)
    {
        float denom = Vector3.Cross(u, v).magnitude;
        if (denom < CotangentEpsilon)
            denom = CotangentEpsilon;
        float dot = Vector3.Dot(u, v);
        return dot / denom;
    }

    private Matrix3x3 ComputePolarRotation(Matrix3x3 covariance)
    {
        Matrix3x3 rotation = covariance;
        if (rotation.FrobeniusNorm() < 1e-8f)
            return Matrix3x3.Identity;
        for (int i = 0; i < PolarIterations; i++)
        {
            Matrix3x3 inverse = rotation.Inverse();
            if (inverse.FrobeniusNorm() < 1e-8f)
            {
                rotation = Matrix3x3.Identity;
                break;
            }
            Matrix3x3 invTranspose = inverse.Transpose();
            Matrix3x3 next = (rotation + invTranspose) * 0.5f;
            if ((next - rotation).FrobeniusNorm() < PolarTolerance)
            {
                rotation = next;
                break;
            }
            rotation = next;
        }
        if (rotation.Determinant() < 0f)
        {
            rotation.m20 = -rotation.m20;
            rotation.m21 = -rotation.m21;
            rotation.m22 = -rotation.m22;
        }
        rotation = Orthonormalize(rotation);
        return rotation;
    }

    private Matrix3x3 Orthonormalize(Matrix3x3 m)
    {
        Vector3 c0 = new Vector3(m.m00, m.m10, m.m20);
        Vector3 c1 = new Vector3(m.m01, m.m11, m.m21);
        Vector3 c2 = new Vector3(m.m02, m.m12, m.m22);

        c0 = SafeNormalize(c0);
        c1 = SafeNormalize(c1 - Vector3.Dot(c1, c0) * c0);
        if (c1.sqrMagnitude < 1e-12f)
            c1 = BuildOrthogonalVector(c0);
        c1 = SafeNormalize(c1);
        c2 = Vector3.Cross(c0, c1);
        if (c2.sqrMagnitude < 1e-12f)
            c2 = BuildOrthogonalVector(c0);
        c2 = SafeNormalize(c2);
        c1 = Vector3.Cross(c2, c0);
        c1 = SafeNormalize(c1);

        return new Matrix3x3
        {
            m00 = c0.x, m10 = c0.y, m20 = c0.z,
            m01 = c1.x, m11 = c1.y, m21 = c1.z,
            m02 = c2.x, m12 = c2.y, m22 = c2.z
        };
    }

    private Vector3 BuildOrthogonalVector(Vector3 v)
    {
        if (Mathf.Abs(v.x) < Mathf.Abs(v.y) && Mathf.Abs(v.x) < Mathf.Abs(v.z))
            return new Vector3(0f, -v.z, v.y);
        if (Mathf.Abs(v.y) < Mathf.Abs(v.z))
            return new Vector3(-v.z, 0f, v.x);
        return new Vector3(-v.y, v.x, 0f);
    }

    private Vector3 SafeNormalize(Vector3 v)
    {
        float mag = v.magnitude;
        if (mag < 1e-12f)
            return Vector3.zero;
        return v / mag;
    }

    private static bool IsValidVector(Vector3 v)
    {
        return !(float.IsNaN(v.x) || float.IsNaN(v.y) || float.IsNaN(v.z) ||
                 float.IsInfinity(v.x) || float.IsInfinity(v.y) || float.IsInfinity(v.z));
    }

    private void SolveGlobalStep(Dictionary<int, float>[] weights, Vector3[] rhs)
    {
        List<int> freeVertices = new List<int>();
        int[] indexMap = new int[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            var v = vertices[i];
            if (v == null || v.Removed)
            {
                indexMap[i] = -1;
                continue;
            }
            if (v.Locked)
            {
                indexMap[i] = -1;
                continue;
            }
            indexMap[i] = freeVertices.Count;
            freeVertices.Add(i);
        }

        int n = freeVertices.Count;
        if (n == 0)
            return;

        var rows = new LinearSystemRow[n];
        float[] rhsX = new float[n];
        float[] rhsY = new float[n];
        float[] rhsZ = new float[n];
        float[] initialX = new float[n];
        float[] initialY = new float[n];
        float[] initialZ = new float[n];

        for (int idx = 0; idx < n; idx++)
        {
            int vi = freeVertices[idx];
            var v = vertices[vi];
            var row = new LinearSystemRow();
            rows[idx] = row;
            float diag = 0f;
            if (weights[vi] != null)
            {
                foreach (var kv in weights[vi])
                {
                    int neighbor = kv.Key;
                    float w = kv.Value;
                    diag += w;
                    int neighborIndex = indexMap[neighbor];
                    if (neighborIndex >= 0)
                    {
                        row.Neighbors.Add(neighborIndex);
                        row.Weights.Add(-w);
                    }
                    else
                    {
                        var lockedVertex = vertices[neighbor];
                        rhsX[idx] += w * lockedVertex.Position.x;
                        rhsY[idx] += w * lockedVertex.Position.y;
                        rhsZ[idx] += w * lockedVertex.Position.z;
                    }
                }
            }
            row.Diagonal = Mathf.Max(diag, 1e-8f);
            rhsX[idx] += rhs[vi].x;
            rhsY[idx] += rhs[vi].y;
            rhsZ[idx] += rhs[vi].z;
            initialX[idx] = v.Position.x;
            initialY[idx] = v.Position.y;
            initialZ[idx] = v.Position.z;
        }

        float[] solutionX = (float[])initialX.Clone();
        float[] solutionY = (float[])initialY.Clone();
        float[] solutionZ = (float[])initialZ.Clone();

        ConjugateGradient(rows, rhsX, solutionX);
        ConjugateGradient(rows, rhsY, solutionY);
        ConjugateGradient(rows, rhsZ, solutionZ);

        for (int idx = 0; idx < n; idx++)
        {
            int vi = freeVertices[idx];
            Vector3 newPos = new Vector3(solutionX[idx], solutionY[idx], solutionZ[idx]);
            if (!IsValidVector(newPos))
                newPos = vertices[vi].RestPosition;
            vertices[vi].Position = newPos;
        }
    }

    private void ConjugateGradient(LinearSystemRow[] rows, float[] rhs, float[] solution)
    {
        int n = rows.Length;
        float[] residual = new float[n];
        float[] direction = new float[n];
        float[] temp = new float[n];

        Multiply(rows, solution, temp);
        for (int i = 0; i < n; i++)
        {
            residual[i] = rhs[i] - temp[i];
            direction[i] = residual[i];
        }

        float deltaNew = Dot(residual, residual);
        float delta0 = deltaNew;
        int iter = 0;
        while (iter < MaxConjugateGradientIterations && deltaNew > ConjugateGradientTolerance * ConjugateGradientTolerance * delta0)
        {
            Multiply(rows, direction, temp);
            float alpha = deltaNew / Mathf.Max(Dot(direction, temp), 1e-12f);
            for (int i = 0; i < n; i++)
            {
                solution[i] += alpha * direction[i];
                residual[i] -= alpha * temp[i];
            }
            float deltaOld = deltaNew;
            deltaNew = Dot(residual, residual);
            float beta = deltaNew / Mathf.Max(deltaOld, 1e-12f);
            for (int i = 0; i < n; i++)
                direction[i] = residual[i] + beta * direction[i];
            iter++;
        }
    }

    private void Multiply(LinearSystemRow[] rows, float[] vector, float[] result)
    {
        int n = rows.Length;
        for (int i = 0; i < n; i++)
        {
            float sum = rows[i].Diagonal * vector[i];
            var neighbors = rows[i].Neighbors;
            var weights = rows[i].Weights;
            for (int k = 0; k < neighbors.Count; k++)
                sum += weights[k] * vector[neighbors[k]];
            result[i] = sum;
        }
    }

    private float Dot(float[] a, float[] b)
    {
        float sum = 0f;
        for (int i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
    }

    private Mesh BuildMesh(out bool removedAnyTriangle)
    {
        removedAnyTriangle = false;
        var mesh = new Mesh { name = sourceMesh.name };

        int vertexCount = 0;
        int[] map = new int[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            if (vertices[i] != null && !vertices[i].Removed)
            {
                map[i] = vertexCount++;
            }
            else
            {
                map[i] = -1;
            }
        }

        var newVertices = new Vector3[vertexCount];
        Vector3[] newNormals = (sourceNormals != null && sourceNormals.Length == sourceVertices.Length) ? new Vector3[vertexCount] : null;
        Vector4[] newTangents = (sourceTangents != null && sourceTangents.Length == sourceVertices.Length) ? new Vector4[vertexCount] : null;
        Color[] newColors = (sourceColors != null && sourceColors.Length == sourceVertices.Length) ? new Color[vertexCount] : null;
        BoneWeight[] newBoneWeights = (sourceBoneWeights != null && sourceBoneWeights.Length == sourceVertices.Length) ? new BoneWeight[vertexCount] : null;
        var uvChannels = new List<Vector2>[sourceUVs.Length];
        for (int channel = 0; channel < sourceUVs.Length; channel++)
        {
            if (sourceUVs[channel] != null && sourceUVs[channel].Length == sourceVertices.Length)
                uvChannels[channel] = new List<Vector2>(vertexCount);
        }

        var vertexSources = new List<int>[vertexCount];

        for (int i = 0; i < vertices.Length; i++)
        {
            var v = vertices[i];
            if (v == null || v.Removed)
                continue;
            int dst = map[i];
            Vector3 finalPosition = v.Position;
            if (!IsValidVector(finalPosition))
                finalPosition = v.RestPosition;
            if (!IsValidVector(finalPosition))
                finalPosition = Vector3.zero;
            newVertices[dst] = finalPosition;
            if (newNormals != null)
            {
                Vector3 normal = v.NormalSum;
                if (normal.sqrMagnitude > 0f)
                    normal.Normalize();
                else
                    normal = Vector3.up;
                newNormals[dst] = normal;
            }
            if (newTangents != null)
            {
                Vector4 tangent = v.TangentSum / Mathf.Max(1, v.AggregateWeight);
                if (tangent == Vector4.zero)
                    tangent = new Vector4(1f, 0f, 0f, 1f);
                newTangents[dst] = tangent;
            }
            if (newColors != null)
            {
                newColors[dst] = v.ColorSum / Mathf.Max(1, v.AggregateWeight);
            }
            if (newBoneWeights != null)
            {
                newBoneWeights[dst] = BuildBoneWeight(v.BoneWeights);
            }
            for (int channel = 0; channel < uvChannels.Length; channel++)
            {
                if (uvChannels[channel] == null)
                    continue;
                if (v.UVSampleCount[channel] > 0)
                    uvChannels[channel].Add(v.UVSum[channel] / Mathf.Max(1, v.UVSampleCount[channel]));
                else
                    uvChannels[channel].Add(Vector2.zero);
            }
            vertexSources[dst] = v.SourceIndices;
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
            mesh.bindposes = (Matrix4x4[])sourceMesh.bindposes.Clone();
        for (int channel = 0; channel < uvChannels.Length; channel++)
        {
            if (uvChannels[channel] != null)
                mesh.SetUVs(channel, uvChannels[channel]);
        }

        CopyBlendShapesToMesh(mesh, vertexSources);

        mesh.subMeshCount = submeshTriangleIndices.Length;
        bool anyTriangles = false;
        for (int sub = 0; sub < submeshTriangleIndices.Length; sub++)
        {
            var list = submeshTriangleIndices[sub];
            List<int> indices = new List<int>();
            foreach (int triIndex in list)
            {
                var tri = triangles[triIndex];
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
        Bounds combinedBounds = mesh.bounds;
        Bounds sourceBounds = sourceMesh.bounds;
        if (sourceBounds.size.sqrMagnitude > 0f)
        {
            combinedBounds.Encapsulate(sourceBounds.min);
            combinedBounds.Encapsulate(sourceBounds.max);
        }
        mesh.bounds = combinedBounds;
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

    private static void AddBoneWeight(Dictionary<int, float> dict, int boneIndex, float weight)
    {
        if (weight <= 0f)
            return;
        if (dict.TryGetValue(boneIndex, out float value))
            dict[boneIndex] = value + weight;
        else
            dict.Add(boneIndex, weight);
    }

    private static BoneWeight BuildBoneWeight(Dictionary<int, float> weights)
    {
        if (weights == null || weights.Count == 0)
            return new BoneWeight();
        var ordered = new List<KeyValuePair<int, float>>(weights);
        ordered.Sort((a, b) => b.Value.CompareTo(a.Value));
        BoneWeight bw = new BoneWeight();
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
                    bw.boneIndex0 = ordered[i].Key;
                    bw.weight0 = value;
                    break;
                case 1:
                    bw.boneIndex1 = ordered[i].Key;
                    bw.weight1 = value;
                    break;
                case 2:
                    bw.boneIndex2 = ordered[i].Key;
                    bw.weight2 = value;
                    break;
                case 3:
                    bw.boneIndex3 = ordered[i].Key;
                    bw.weight3 = value;
                    break;
            }
        }
        return bw;
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
