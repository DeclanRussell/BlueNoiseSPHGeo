#include "Model.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

Model::Model(std::string _path):m_modelLoaded(false){
#ifdef LINUX
    glewExperimental = GL_TRUE;
    GLenum error = glewInit();
    if(error != GLEW_OK){
        std::cerr<<"GLEW IS NOT OK!!!"<<std::endl;
    }
#endif
    loadModel(_path);
}
Model::Model():m_modelLoaded(false){
#ifdef LINUX
    glewExperimental = GL_TRUE;
    GLenum error = glewInit();
    if(error != GLEW_OK){
        std::cerr<<"GLEW IS NOT OK!!!"<<std::endl;
    }
#endif
}

Model::~Model(){
   clearModel();
}

void Model::loadCube(){
    // Clear our buffers if a model is already loaded
    clearModel();
    // create and add data to the vbo
    GLfloat bufferData[] = {

      //  X     Y     Z       U     V        Normals
      // bottom
      -1.0f,-1.0f,-1.0f,   0.0f, 0.0f,       0.0, -1.0, 0.0,
      1.0f,-1.0f,-1.0f,   1.0f, 0.0f,        0.0, -1.0, 0.0,
      -1.0f,-1.0f, 1.0f,   0.0f, 1.0f,       0.0, -1.0, 0.0,
      1.0f,-1.0f,-1.0f,   1.0f, 0.0f,        0.0, -1.0, 0.0,
      1.0f,-1.0f, 1.0f,   1.0f, 1.0f,        0.0, -1.0, 0.0,
      -1.0f,-1.0f, 1.0f,   0.0f, 1.0f,       0.0, -1.0, 0.0,

      // top
      -1.0f, 1.0f,-1.0f,   1.0f, 0.0f,       0.0, 1.0, 0.0,
      -1.0f, 1.0f, 1.0f,   1.0f, 1.0f,       0.0, 1.0, 0.0,
      1.0f, 1.0f,-1.0f,   0.0f, 0.0f,        0.0, 1.0, 0.0,
      1.0f, 1.0f,-1.0f,   0.0f, 0.0f,        0.0, 1.0, 0.0,
      -1.0f, 1.0f, 1.0f,   1.0f, 1.0f,       0.0, 1.0, 0.0,
      1.0f, 1.0f, 1.0f,   0.0f, 1.0f,        0.0, 1.0, 0.0,

      // front
      -1.0f,-1.0f, 1.0f,   0.0f, 0.0f,       0.0, 0.0, 1.0,
      1.0f,-1.0f, 1.0f,   1.0f, 0.0f,        0.0, 0.0, 1.0,
      -1.0f, 1.0f, 1.0f,   0.0f, 1.0f,       0.0, 0.0, 1.0,
      1.0f,-1.0f, 1.0f,   1.0f, 0.0f,        0.0, 0.0, 1.0,
      1.0f, 1.0f, 1.0f,   1.0f, 1.0f,        0.0, 0.0, 1.0,
      -1.0f, 1.0f, 1.0f,   0.0f, 1.0f,       0.0, 0.0, 1.0,

      // back
      -1.0f,-1.0f,-1.0f,   1.0f, 0.0f,       0.0, 0.0, -1.0,
      -1.0f, 1.0f,-1.0f,   1.0f, 1.0f,       0.0, 0.0, -1.0,
      1.0f,-1.0f,-1.0f,   0.0f, 0.0f,        0.0, 0.0, -1.0,
      1.0f,-1.0f,-1.0f,   0.0f, 0.0f,        0.0, 0.0, -1.0,
      -1.0f, 1.0f,-1.0f,   1.0f, 1.0f,       0.0, 0.0, -1.0,
      1.0f, 1.0f,-1.0f,   0.0f, 1.0f,        0.0, 0.0, -1.0,

      // left
      -1.0f,-1.0f, 1.0f,   1.0f, 0.0f,       -1.0, 0.0, 0.0,
      -1.0f, 1.0f,-1.0f,   0.0f, 1.0f,       -1.0, 0.0, 0.0,
      -1.0f,-1.0f,-1.0f,   0.0f, 0.0f,       -1.0, 0.0, 0.0,
      -1.0f,-1.0f, 1.0f,   1.0f, 0.0f,       -1.0, 0.0, 0.0,
      -1.0f, 1.0f, 1.0f,   1.0f, 1.0f,       -1.0, 0.0, 0.0,
      -1.0f, 1.0f,-1.0f,   0.0f, 1.0f,       -1.0, 0.0, 0.0,

      // right
      1.0f,-1.0f, 1.0f,   0.0f, 0.0f,        1.0, 0.0, 0.0,
      1.0f,-1.0f,-1.0f,   1.0f, 0.0f,        1.0, 0.0, 0.0,
      1.0f, 1.0f,-1.0f,   1.0f, 1.0f,        1.0, 0.0, 0.0,
      1.0f,-1.0f, 1.0f,   0.0f, 0.0f,        1.0, 0.0, 0.0,
      1.0f, 1.0f,-1.0f,   1.0f, 1.0f,        1.0, 0.0, 0.0,
      1.0f, 1.0f, 1.0f,   0.0f, 1.0f,        1.0, 0.0, 0.0
    };

    m_numVerts = 6*3*2;
    m_vertexs.resize(m_numVerts);
    memcpy(&m_vertexs[0].x,&bufferData[0],sizeof(float)*m_numVerts);
    glGenBuffers(1, &m_vertexVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(bufferData), bufferData, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, NULL);

    // create a vao
    glGenVertexArrays(1,&m_VAO);
    glBindVertexArray(m_VAO);

    // find the position of the shader input "position"
    GLint positionLoc = 0;//m_program->getAttribLoc("position");
    GLint normalLoc = 1;// m_program->getAttribLoc("normals");
    GLint texCoordLoc = 2;//m_program->getAttribLoc("texCoord");


    // connect the data to the shader input
    glEnableVertexAttribArray(positionLoc);
    glEnableVertexAttribArray(texCoordLoc);
    glEnableVertexAttribArray(normalLoc);

    glBindBuffer(GL_ARRAY_BUFFER, m_vertexVBO);
    glVertexAttribPointer(positionLoc, 3, GL_FLOAT, GL_FALSE, 8*sizeof(GL_FLOAT), (GLvoid*)(0*sizeof(GL_FLOAT)));
    glVertexAttribPointer(texCoordLoc, 2, GL_FLOAT, GL_FALSE, 8*sizeof(GL_FLOAT), (GLvoid*)(3*sizeof(GL_FLOAT)));
    glVertexAttribPointer(normalLoc, 3, GL_FLOAT, GL_FALSE, 8*sizeof(GL_FLOAT), (GLvoid*)(5*sizeof(GL_FLOAT)));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    m_modelLoaded = true;

}
//----------------------------------------------------------------------------------------------------------------------
void Model::clearModel()
{
    if(m_modelLoaded)
    {
        glDeleteBuffers(1, &m_vertexVBO);
        glDeleteBuffers(1, &m_normalsVBO);
        glDeleteBuffers(1, &m_tangentsVBO);
        glDeleteBuffers(1, &m_textureVBO);
        glDeleteBuffers(1, &m_indiciesVBO);
        glDeleteVertexArrays(1, &m_VAO);

        m_numNormals = m_numVerts = m_numTangents = m_numTexCoords = m_numFaces = m_numIndicies = 0;

        m_vertexs.clear();
        m_normals.clear();
        m_tangents.clear();
        m_textureCoords.clear();
        m_indices.clear();

        m_modelLoaded = false;
    }
}
//----------------------------------------------------------------------------------------------------------------------
void Model::loadModel(std::string _path){
    // Clear model if one is already loaded
    clearModel();
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(_path.c_str(), aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace | aiProcess_Triangulate);

    aiMesh* mesh = scene->mMeshes[0];

    aiFace* faces;
    aiVector3D* vertices;
    aiVector3D* normals;
    aiVector3D* textureCoords;
    aiVector3D* tangents;

    m_numFaces = mesh->mNumFaces;

    m_numVerts = mesh->mNumVertices;
    if(mesh->HasNormals())m_numNormals = m_numVerts;
    if(mesh->HasTangentsAndBitangents())m_numTangents = m_numVerts;
    if(mesh->HasTextureCoords(0)) m_numTexCoords = m_numVerts;

    vertices = mesh->mVertices;
    normals = mesh->mNormals;
    textureCoords = mesh->mTextureCoords[0];
    tangents = mesh->mTangents;
    faces = mesh->mFaces;
    m_vertexs.resize(m_numVerts);
    m_numIndicies = 3; // We (I) only support triangle meshes (your polygons can suck it!)

    memcpy(&m_vertexs[0].x,&vertices[0].x,sizeof(aiVector3D)*m_numVerts);
    m_indices.resize(m_numFaces);
    for(unsigned int i=0;i<m_numFaces;i++)
    {
        m_indices[i].x = faces[i].mIndices[0];
        m_indices[i].y = faces[i].mIndices[1];
        m_indices[i].z = faces[i].mIndices[2];
    }

    if(m_numNormals)
    {
        m_normals.resize(m_numNormals);
        memcpy(&m_normals[0].x,&normals[0].x,sizeof(aiVector3D)*m_numNormals);
    }
    if(m_numTangents)
    {
        m_tangents.resize(m_numTangents);
        memcpy(&m_tangents[0].x,&tangents[0].x,sizeof(aiVector3D)*m_numTangents);
    }
    if(m_numTexCoords)
    {
        m_textureCoords.resize(m_numTexCoords);
        memcpy(&m_textureCoords[0].x,&textureCoords[0].x,sizeof(aiVector3D)*m_numTexCoords);
    }
    glGenBuffers(1, &m_vertexVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(aiVector3D)*mesh->mNumVertices, &vertices[0].x, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &m_normalsVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_normalsVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(aiVector3D)*mesh->mNumVertices, &normals[0].x, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &m_textureVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_textureVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(aiVector3D)*mesh->mNumVertices, &textureCoords[0].x, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &m_tangentsVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_tangentsVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(aiVector3D)*mesh->mNumVertices, &tangents[0].x, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &m_indiciesVBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indiciesVBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int)*3*m_indices.size(), &m_indices[0].x, GL_STATIC_DRAW);

    // create a vao
    glGenVertexArrays(1,&m_VAO);
    glBindVertexArray(m_VAO);

    // find the position of the shader input "position"
    GLint positionLoc = 0;//m_program->getAttribLoc("position");
    GLint normalLoc = 1;// m_program->getAttribLoc("normals");
    GLint texCoordLoc = 2;
    GLint tangLoc = 3;

    // connect the data to the shader input
    glEnableVertexAttribArray(positionLoc);
    glEnableVertexAttribArray(normalLoc);
    glEnableVertexAttribArray(texCoordLoc);
    glEnableVertexAttribArray(tangLoc);

    glBindBuffer(GL_ARRAY_BUFFER, m_vertexVBO);
    glVertexAttribPointer(positionLoc, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, m_normalsVBO);
    glVertexAttribPointer(normalLoc, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, m_textureVBO);
    glVertexAttribPointer(texCoordLoc, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, m_tangentsVBO);
    glVertexAttribPointer(tangLoc, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);

    m_modelLoaded = true;
}
