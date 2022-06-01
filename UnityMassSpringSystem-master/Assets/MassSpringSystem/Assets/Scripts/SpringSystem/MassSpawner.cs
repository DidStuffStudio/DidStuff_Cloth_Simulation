using UnityEngine;
using System.Collections;
using Unity.Mathematics;

public class MassSpawner : MonoBehaviour
{
    private Vector3[] positions;
    private Mesh mesh;
    private Vector3[] vertices;
    [SerializeField] private Material clothMaterial;

    private void FixedUpdate() => mesh.SetVertices(positions);

    public void UpdatePositions (Vector3[] p)
    {
        positions = p;
    }

    public void GenerateClothMesh(Vector3[] pos, int x, int y)
    {

        var cloth = new QuadPlaneDef(x, y, new Vector3(3.0f, 1.0f, 4.0f));
        cloth.UVScale = Vector2.one * -1;
        GameObject go = MakeQuadPlane.Create(cloth);
        go.transform.position = new Vector3(0, 0, 0);
        go.GetComponent<MeshRenderer>().material = clothMaterial;
        go.name = "Cloth";
        mesh = go.GetComponent<MeshFilter>().mesh;
        mesh.SetVertices(pos);
    }
}
