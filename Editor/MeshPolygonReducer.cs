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

        int[] triangles = mesh.triangles;
        int triangleCount = triangles.Length / 3;
        if (triangleCount == 0)
            return null;

        List<int> candidateTriangleIndices = new List<int>();
        bool useMask = vertexMask != null && vertexMask.Length == mesh.vertexCount;
        for (int i = 0; i < triangles.Length; i += 3)
        {
            if (useMask)
            {
                int v0 = triangles[i];
                int v1 = triangles[i + 1];
                int v2 = triangles[i + 2];
                if (!(vertexMask[v0] && vertexMask[v1] && vertexMask[v2]))
                    continue;
            }
            candidateTriangleIndices.Add(i);
        }

        if (candidateTriangleIndices.Count == 0)
            return null;

        int targetRemoval = Mathf.Clamp(Mathf.RoundToInt(candidateTriangleIndices.Count * reductionRatio), 0, candidateTriangleIndices.Count);
        if (targetRemoval == 0)
            return null;

        System.Random random;
        if (seed.HasValue)
            random = new System.Random(seed.Value);
        else
            random = new System.Random(mesh.GetInstanceID() ^ candidateTriangleIndices.Count ^ DateTime.Now.Millisecond);

        for (int i = candidateTriangleIndices.Count - 1; i > 0; i--)
        {
            int swapIndex = random.Next(i + 1);
            int temp = candidateTriangleIndices[i];
            candidateTriangleIndices[i] = candidateTriangleIndices[swapIndex];
            candidateTriangleIndices[swapIndex] = temp;
        }

        var removed = new HashSet<int>();
        for (int i = 0; i < targetRemoval; i++)
            removed.Add(candidateTriangleIndices[i]);

        List<int> newTriangles = new List<int>(triangles.Length - removed.Count * 3);
        for (int i = 0; i < triangles.Length; i += 3)
        {
            if (removed.Contains(i))
                continue;
            newTriangles.Add(triangles[i]);
            newTriangles.Add(triangles[i + 1]);
            newTriangles.Add(triangles[i + 2]);
        }

        if (newTriangles.Count == triangles.Length)
            return null;

        var newMesh = UnityEngine.Object.Instantiate(mesh);
        newMesh.triangles = newTriangles.ToArray();
        newMesh.RecalculateBounds();
        if (mesh.normals != null && mesh.normals.Length > 0)
            newMesh.RecalculateNormals();
        if (mesh.tangents != null && mesh.tangents.Length > 0)
            newMesh.RecalculateTangents();
        return newMesh;
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
