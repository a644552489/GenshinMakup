using UnityEditor;
using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Collections;
using UnityEditor.Formats.Fbx.Exporter;


public enum WRITETYPE
{
    VertexColor = 0,
    Tangent = 1,
    // Texter=2,
}
public class SmoothNormalTools : EditorWindow
{

    public WRITETYPE wt;
    // public bool customMesh;

    [MenuItem("Tools/ƽ�����߹���")]
    public static void ShowWindow()
    {
        EditorWindow.GetWindow(typeof(SmoothNormalTools));//��ʾ���д���ʵ�������û�У��봴��һ����
    }


    void OnGUI()
    {

        GUILayout.Space(5);
        GUILayout.Label("1������Scene��ѡ����Ҫƽ�����ߵ�����", EditorStyles.boldLabel);
        // mesh = (MeshFilter)EditorGUILayout.ObjectField(mesh,typeof(MeshFilter),true);
        GUILayout.Space(10);
        GUILayout.Label("2����ѡ����Ҫд��ƽ���������ռ䷨�����ݵ�Ŀ��", EditorStyles.boldLabel);
        wt = (WRITETYPE)EditorGUILayout.EnumPopup("д��Ŀ��", wt);
        GUILayout.Space(10);



        switch (wt)
        {
            case WRITETYPE.Tangent://ִ��д�뵽 ��������
                GUILayout.Label("  �����ƽ����ķ���д�뵽����������", EditorStyles.boldLabel);
                break;
            case WRITETYPE.VertexColor:// д�뵽����ɫ
                GUILayout.Label("  �����ƽ����ķ���д�뵽����ɫ��RGB�У�A���ֲ���", EditorStyles.boldLabel);
                break;
                // case WRITETYPE.Texter://д�뵽�Զ�����ͼ
                //             GUILayout.Label ("  д����ͼ��������ã�", EditorStyles.boldLabel);
                // break;
        }
        if (GUILayout.Button("3��ƽ������(Ԥ��Ч����"))
        {//ִ��ƽ��
            SmoothNormalPrev(wt);
        }

        GUILayout.Label("֮����ܻᱨNull Reference����");
        GUILayout.Label("��Ҫ����Mesh����MeshFilter�и��ǣ������������ñ���");
        GUILayout.Space(10);
        GUILayout.Label("  �Ὣmesh���浽Assets/SmoothNormalTools/��", EditorStyles.boldLabel);
        GUILayout.Space(5);
        // customMesh = EditorGUILayout.BeginToggleGroup ("Optional Settings", customMesh);
        // EditorGUILayout.EndToggleGroup ();
        if (GUILayout.Button("4������Mesh"))
        {
            selectMesh();
        }

    }
    public void SmoothNormalPrev(WRITETYPE wt)//Meshѡ���� �޸Ĳ�Ԥ��
    {


        if (Selection.activeGameObject == null)
        {//����Ƿ��ȡ������
            Debug.LogError("��ѡ������");
            return;
        }
        MeshFilter[] meshFilters = Selection.activeGameObject.GetComponentsInChildren<MeshFilter>();
        SkinnedMeshRenderer[] skinMeshRenders = Selection.activeGameObject.GetComponentsInChildren<SkinnedMeshRenderer>();
        foreach (var meshFilter in meshFilters)//��������Mesh ����ƽ�����߷���
        {
            Mesh mesh = meshFilter.sharedMesh;
            Vector3[] averageNormals = AverageNormal(mesh);
            write2mesh(mesh, averageNormals);
        }
        foreach (var skinMeshRender in skinMeshRenders)
        {
            Mesh mesh = skinMeshRender.sharedMesh;
            Vector3[] averageNormals = AverageNormal(mesh);
            write2mesh(mesh, averageNormals);
        }
    }

    public Vector3[] AverageNormal(Mesh mesh)
    {

        var averageNormalHash = new Dictionary<Vector3, Vector3>();
        for (var j = 0; j < mesh.vertexCount; j++)
        {
            if (!averageNormalHash.ContainsKey(mesh.vertices[j]))
            {
                averageNormalHash.Add(mesh.vertices[j], mesh.normals[j]);
            }
            else
            {
                averageNormalHash[mesh.vertices[j]] =
                    (averageNormalHash[mesh.vertices[j]] + mesh.normals[j]).normalized;
            }
        }

        var averageNormals = new Vector3[mesh.vertexCount];
        for (var j = 0; j < mesh.vertexCount; j++)
        {
            averageNormals[j] = averageNormalHash[mesh.vertices[j]];
            // averageNormals[j] = averageNormals[j].normalized;
        }

        ArrayList ObjToTangMatrix = new ArrayList();
        for (int i = 0; i < mesh.vertexCount; i++)
        {
            Vector3[] OtoTMat = new Vector3[3];
            OtoTMat[0] = new Vector3(mesh.tangents[i].x, mesh.tangents[i].y, mesh.tangents[i].z);
            OtoTMat[1] = Vector3.Cross(mesh.normals[i], OtoTMat[0]);
            OtoTMat[1] = new Vector3(OtoTMat[1].x * mesh.tangents[i].w, OtoTMat[1].y * mesh.tangents[i].w, OtoTMat[1].z * mesh.tangents[i].w);
            OtoTMat[2] = mesh.normals[i];
            ObjToTangMatrix.Add(OtoTMat);
        }

        for (int i = 0; i < averageNormals.Length; i++)
        {
            Vector3 tNormal = Vector3.zero;
            tNormal.x = Vector3.Dot(((Vector3[])ObjToTangMatrix[i])[0], averageNormals[i]);
            tNormal.y = Vector3.Dot(((Vector3[])ObjToTangMatrix[i])[1], averageNormals[i]);
            tNormal.z = Vector3.Dot(((Vector3[])ObjToTangMatrix[i])[2], averageNormals[i]);
            averageNormals[i] = tNormal;

        }



        return averageNormals;

    }

    public void write2mesh(Mesh mesh, Vector3[] averageNormals)
    {
        switch (wt)
        {
            case WRITETYPE.Tangent://ִ��д�뵽 ��������
                var tangents = new Vector4[mesh.vertexCount];
                for (var j = 0; j < mesh.vertexCount; j++)
                {
                    tangents[j] = new Vector4(averageNormals[j].x, averageNormals[j].y, averageNormals[j].z, 0);
                }
                mesh.tangents = tangents;
                break;
            case WRITETYPE.VertexColor:// д�뵽����ɫ
                Color[] _colors = new Color[mesh.vertexCount];
                Color[] _colors2 = new Color[mesh.vertexCount];
                _colors2 = mesh.colors;
                for (var j = 0; j < mesh.vertexCount; j++)
                {
                    _colors[j] = new Vector4(averageNormals[j].x*0.5f+0.5f, averageNormals[j].y*0.5f+0.5f, averageNormals[j].z*0.5f+0.5f, _colors2[j].a);
                }
                mesh.colors = _colors;
                break;
        }
    }


    public void selectMesh()
    {

        if (Selection.activeGameObject == null)
        {//����Ƿ��ȡ������
            Debug.LogError("��ѡ������");
            return;
        }
        MeshFilter[] meshFilters = Selection.activeGameObject.GetComponentsInChildren<MeshFilter>();
        SkinnedMeshRenderer[] skinMeshRenders = Selection.activeGameObject.GetComponentsInChildren<SkinnedMeshRenderer>();
        foreach (var meshFilter in meshFilters)//��������Mesh ����ƽ�����߷���
        {
            Mesh mesh = meshFilter.sharedMesh;
            Vector3[] averageNormals = AverageNormal(mesh);
            exportMesh(mesh, averageNormals);

        }
        foreach (var skinMeshRender in skinMeshRenders)
        {
            Mesh mesh = skinMeshRender.sharedMesh;
            Vector3[] averageNormals = AverageNormal(mesh);
            exportMesh(mesh, averageNormals);
        }
        ModelExporter.ExportObject("Assets/SmoothNormalTools/" +Selection.activeGameObject.name +".fbx" ,Selection.activeGameObject);
    }



    public void Copy(Mesh dest, Mesh src)
    {
        dest.Clear();
        dest.vertices = src.vertices;

        List<Vector4> uvs = new List<Vector4>();

        src.GetUVs(0, uvs); dest.SetUVs(0, uvs);
        src.GetUVs(1, uvs); dest.SetUVs(1, uvs);
        src.GetUVs(2, uvs); dest.SetUVs(2, uvs);
        src.GetUVs(3, uvs); dest.SetUVs(3, uvs);

        dest.normals = src.normals;
        dest.tangents = src.tangents;
        dest.boneWeights = src.boneWeights;
        dest.colors = src.colors;
        dest.colors32 = src.colors32;
        dest.bindposes = src.bindposes;

        dest.subMeshCount = src.subMeshCount;

        for (int i = 0; i < src.subMeshCount; i++)
            dest.SetIndices(src.GetIndices(i), src.GetTopology(i), i);

        dest.name = src.name;
    }
    public void exportMesh(Mesh mesh, Vector3[] averageNormals)
    {
        Mesh mesh2 = new Mesh();
        Copy(mesh2, mesh);
        switch (wt)
        {
            case WRITETYPE.Tangent://ִ��д�뵽 ��������
                Debug.Log("д�뵽������");
                var tangents = new Vector4[mesh2.vertexCount];
                for (var j = 0; j < mesh2.vertexCount; j++)
                {
                    tangents[j] = new Vector4(averageNormals[j].x, averageNormals[j].y, averageNormals[j].z, 0);
                }
                mesh2.tangents = tangents;
                break;
            case WRITETYPE.VertexColor:// д�뵽����ɫ
                Debug.Log("д�뵽����ɫ��");
                Color[] _colors = new Color[mesh2.vertexCount];
                Color[] _colors2 = new Color[mesh2.vertexCount];
                _colors2 = mesh2.colors;
                for (var j = 0; j < mesh2.vertexCount; j++)
                {
                    _colors[j] = new Vector4(averageNormals[j].x* 0.5f +0.5f, averageNormals[j].y * 0.5f + 0.5f, averageNormals[j].z * 0.5f + 0.5f, _colors2[j].a);
                }
                mesh2.colors = _colors;
                break;
        }

        //�����ļ���·��
        string DeletePath = Application.dataPath + "/SmoothNormalTools";
        Debug.Log(DeletePath);
        //�ж��ļ���·���Ƿ����
        if (!Directory.Exists(DeletePath))
        {  //����
            Directory.CreateDirectory(DeletePath);
        }
        //ˢ��
        AssetDatabase.Refresh();


        mesh2.name = mesh2.name + "_SMNormal";
        Debug.Log(mesh2.vertexCount);

      //  AssetDatabase.CreateAsset(mesh2, "Assets/SmoothNormalTools/" + mesh2.name + ".asset");

    }
}