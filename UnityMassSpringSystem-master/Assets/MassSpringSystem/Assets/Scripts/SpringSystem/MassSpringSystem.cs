/**
 * Copyright 2017 Sean Soraghan
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
 * documentation files (the "Software"), to deal in the Software without restriction, including without 
 * limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of 
 * the Software, and to permit persons to whom the Software is furnished to do so, subject to the following 
 * conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or substantial 
 * portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
 * LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO 
 * EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE 
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

using UnityEngine;
using System.Collections;

//===========================================================================================
// Simple class that holds various property names and attributes for the 
// MassSpringSystem.compute shader.
//===========================================================================================

public static class SpringComputeShaderProperties
{
    public const string PositionBufferName       = "posBuffer";
    public const string VelocityBufferName       = "velBuffer";
    public const string SphereBufferName         = "sphereBuffer";
    public const string ExternalForcesBufferName = "externalForcesBuffer";
    public const string PropertiesBufferName     = "propertiesBuffer";
    public const string DeltaTimeBufferName      = "deltaTimeBuffer";
    public const int    NumProperties            = 5; // NOTE --> Change whenever we add a new field to the properties buffer.
    public const string VelKernel                = "CSMainVel";
    public const string ForcesKernel    = "CSMainForces";
    public const string ForcesBuffer    = "forcesBuffer";
}


//===========================================================================================
// Summary
//===========================================================================================
/**
 * This class is used to periodically run position and velocity buffers through compute
 * shader kernels that update them according to a mass spring model. It also provides access
 * to properties of the model that can be tweaked from the editor. This class is used to 
 * update these property values in the compute shader as they are changed externally.
 * 
 * The MassSpawner Spawner member variable is used to spawn and update game objects according
 * to the positions on the mass spring grid. Alternatively, points can be rendered using
 * the RenderShader Shader member variable. In order to do this, uncomment the OnPostRender
 * function and comment the Update function and the call to SpawnPrimitives in the 
 * Initialise function.
 * 
 * Various ComputeBuffer variables are used to read and write data to and from the compute 
 * shader (MassSpringComputeShader).
 * 
 * This class can also be used to translate touch and mouse input (from the UITouchHandler) 
 * into forces that are applied to the mass spring system (implemented by the 
 * MassSpringComputeShader). 
 */

public class MassSpringSystem : MonoBehaviour
{
    /** The compute shader that implements the mass spring model.
     */
    public ComputeShader MassSpringComputeShader;
    
    
    /** The mass of individual mass points in the mass spring model.
     *  Increasing this will make the mass points more resistive to
     *  the springs in the model, but will also reduce their velocity.
     */
    [Range(0.1f, 100.0f)] public float Mass            = 1.0f;  
   
    /** The level of damping in the system. Increasing this value
     *  will cause the system to return to a more 'stable' state quicker,
     *  and will reduce the propagation of forces throughout the grid.
     */ 
    [Range(0.1f,10f)] public float Damping         = 0.1f;
    
    /** The stiffness of the spings in the grid. Increasing this will
     *  cause mass points to 'rebound' with higher velocity, and will
     *  also decrease the time taken for the system to return to a
     *  'stable' state.
     */
    [Range(0.1f,2000f)] public float SpringStiffness = 10.0f;

    /** The lenght of the springs in the grid. This defines how far
     *  each mass unit is at a resting state.
     */
    [Range(0.1f, 10.0f)]  public float SpringLength    = 1.0f;
    

    [Range(0.0f, -15.0f)]  public float Gravity    = -4.0f;

    [Range(0.01f, 200.0f)]  public float windScale; // scale the noise (smaller or larger movements)
    [Range(0.01f, 15.0f)]  public float windStrength; // noise affect the position more

    [SerializeField] private GameObject sphere;

    /** The controller of the game object spawner object.
     */
    public MassSpawner Spawner;
    
    
    private ComputeBuffer debugBuffer;
    private ComputeBuffer propertiesBuffer;
    private ComputeBuffer deltaTimeBuffer;
    private ComputeBuffer positionBuffer;
    private ComputeBuffer velocityBuffer;
    private ComputeBuffer externalForcesBuffer;
    private ComputeBuffer sphereBuffer;
    private ComputeBuffer forcesBuffer;
    
    private const int gridUnitSideX       = 24;
    private const int gridUnitSideY       = 24;
    private const int numThreadsPerGroupX = 8;
    private const int numThreadsPerGroupY = 8;

    /** The resolution of our entire grid, according to the resolution and layout of the individual
     *  blocks processed in parallel by the compute shader.
     */
    private int GridResX;
    private int GridResY;

    /** The total number of mass points (vertices) in the grid.
     */ 
    private int VertCount;

    /** The two kernels in the compute shader for updating the positions and velocities, respectively. 
     */
    private int VelKernel;
    private int forcesKernel;
    

 private void Start ()
    {
        Initialise();
    }

    void Update ()
    {
        Dispatch();
        UpdateClothMesh();
    }
    

    private void OnDisable()
    {
        ReleaseBuffers();
    }

    /** Helper functions to get grid dimension properties in the world space.
     */
    private float GetWorldGridSideLengthX()
    {
        return GridResX * SpringLength;
    }

    private float GetWorldGridSideLengthY()
    {
        return GridResY * SpringLength;
    }

    //===========================================================================================
    // Construction / Destruction
    //===========================================================================================

    /** Initialise all of the compute buffers that will be used to update and read from the compute shader.
     *  Fill all of these buffers with data in order to construct a resting spring mass grid of the correct dimensions.
     *  Initialise the position and velocity kernel values using their name values from the SpringComputeShaderProperties static class.
     */
    public void CreateBuffers()
    {
        var sizeOfThreeFloat = sizeof(float) * 3;
        positionBuffer       = new ComputeBuffer (VertCount, sizeOfThreeFloat);
        velocityBuffer       = new ComputeBuffer (VertCount, sizeOfThreeFloat);
        externalForcesBuffer = new ComputeBuffer (VertCount, sizeOfThreeFloat);
        debugBuffer          = new ComputeBuffer (VertCount, sizeOfThreeFloat);
        propertiesBuffer     = new ComputeBuffer (SpringComputeShaderProperties.NumProperties, sizeof (float));
        deltaTimeBuffer      = new ComputeBuffer (1, sizeof (float));
        forcesBuffer = new ComputeBuffer(VertCount, sizeOfThreeFloat);
        sphereBuffer = new ComputeBuffer(4, sizeof(float)); // one float and one float3
        
        ResetBuffers();

        VelKernel = MassSpringComputeShader.FindKernel (SpringComputeShaderProperties.VelKernel);
        forcesKernel = MassSpringComputeShader.FindKernel (SpringComputeShaderProperties.ForcesKernel);
    }

    /** Fills all of the compute buffers with starting data to construct a resting spring mass grid of the correct dimensions
     *  according to GridResX and GridResY. For each vertex position we also calculate the positions of each of the neighbouring 
     *  vertices so that we can send this to the compute shader.
     */
    public void ResetBuffers()
    {
        Vector3[] positions  = new Vector3[VertCount];
        Vector3[] velocities = new Vector3[VertCount];
        Vector3[] extForces  = new Vector3[VertCount];
        Vector3[] forces = new Vector3[VertCount];
        
        for (float i = 0; i < VertCount; i+=1)
        {
            float x = ((i % GridResX - GridResX / 2.0f) / GridResX) * GetWorldGridSideLengthX();
            float y = ((i / GridResX - GridResY / 2.0f) / GridResY) * GetWorldGridSideLengthY();

            positions[(int)i]  = new Vector3 (x, y, 0.0f);
            velocities[(int)i] = new Vector3 (0.0f, 0.0f, 0.0f);
            extForces[(int)i]  = new Vector3 (0.0f, 0.0f, 0.0f);
        }

        positionBuffer.SetData       (positions);
        velocityBuffer.SetData       (velocities);
        debugBuffer.SetData          (positions);
        externalForcesBuffer.SetData (extForces);
        sphereBuffer.SetData         (new float[] {0.0f, 0.0f, 0.0f, 0.0f});
        forcesBuffer.SetData(forces);
    }

    public void ReleaseBuffers()
    {
        if (positionBuffer != null)
            positionBuffer.Release();
        if (velocityBuffer != null)
            velocityBuffer.Release();
        if (debugBuffer != null)
            debugBuffer.Release();
        if (propertiesBuffer != null)
            propertiesBuffer.Release();
        if (deltaTimeBuffer != null)
            deltaTimeBuffer.Release();
        if (sphereBuffer != null)
            sphereBuffer.Release();
        if (externalForcesBuffer != null)
            externalForcesBuffer.Release();
        if (forcesBuffer != null)
            forcesBuffer.Release();
    }


    /** Calculate our entire grid resolution and vertex count from the structure of the compute shader.
     *  Create our render material, and initialise and fill our compute buffers. Send the vertex neighbour 
     *  positions to the compute shader (we only need to do this once, whereas the position and velocities
     *  need to be sent continuously). Finally, we get the initial positions from the compute buffer and use
     *  them to spawn our game objects using the Spawner.
     */
    public void Initialise()
    { 
        GridResX  = gridUnitSideX * numThreadsPerGroupX;
        GridResY  = gridUnitSideY * numThreadsPerGroupY;
        VertCount = GridResX * GridResY;
        CreateBuffers();
        Vector3[] positions  = new Vector3[VertCount];
        positionBuffer.GetData (positions);
        Spawner.GenerateClothMesh(positions, GridResX, GridResY);
    }
    

    void SetGridPropertiesAndTime()
    {
        propertiesBuffer.SetData (new float[] {Mass, Damping, SpringStiffness, SpringLength, Gravity} );
        deltaTimeBuffer.SetData  (new float[] {Time.fixedDeltaTime});
    }


    void SetVelocityBuffers()
    {
        float sphereRadius = sphere.GetComponent<SphereCollider>().radius;
        Vector3 sphereOrigin = sphere.transform.position;
        sphereBuffer.SetData(new float[] { sphereRadius, sphereOrigin.x, sphereOrigin.y, sphereOrigin.z });
        MassSpringComputeShader.SetBuffer (VelKernel, SpringComputeShaderProperties.SphereBufferName, sphereBuffer);
        externalForcesBuffer.SetData(CalculateWindForces());
        MassSpringComputeShader.SetBuffer (VelKernel, SpringComputeShaderProperties.PropertiesBufferName,     propertiesBuffer);
        MassSpringComputeShader.SetBuffer (VelKernel, SpringComputeShaderProperties.DeltaTimeBufferName,      deltaTimeBuffer);
        MassSpringComputeShader.SetBuffer (VelKernel, SpringComputeShaderProperties.ExternalForcesBufferName, externalForcesBuffer);
        MassSpringComputeShader.SetBuffer (VelKernel, SpringComputeShaderProperties.VelocityBufferName,       velocityBuffer);
        MassSpringComputeShader.SetBuffer (VelKernel, SpringComputeShaderProperties.PositionBufferName,       positionBuffer);
        MassSpringComputeShader.SetBuffer (VelKernel, SpringComputeShaderProperties.ForcesBuffer,       forcesBuffer);

    }

    void SetForcesBuffer()
    {
        MassSpringComputeShader.SetBuffer (forcesKernel, SpringComputeShaderProperties.PropertiesBufferName,     propertiesBuffer);
        MassSpringComputeShader.SetBuffer (forcesKernel, SpringComputeShaderProperties.DeltaTimeBufferName,      deltaTimeBuffer);
        MassSpringComputeShader.SetBuffer (forcesKernel, SpringComputeShaderProperties.ExternalForcesBufferName, externalForcesBuffer);
        MassSpringComputeShader.SetBuffer (forcesKernel, SpringComputeShaderProperties.ForcesBuffer,       forcesBuffer);
        MassSpringComputeShader.SetBuffer (forcesKernel, SpringComputeShaderProperties.VelocityBufferName,       velocityBuffer);
        MassSpringComputeShader.SetBuffer (forcesKernel, SpringComputeShaderProperties.PositionBufferName,       positionBuffer);
    }
    
    void UpdateClothMesh()
    {
        Vector3[] positions = new Vector3[VertCount];
        positionBuffer.GetData (positions);

        if (Spawner != null)
            Spawner.UpdatePositions (positions);
    }

    private void Dispatch()
    {
        SetGridPropertiesAndTime();
        SetForcesBuffer();
        SetVelocityBuffers();
        MassSpringComputeShader.Dispatch  (forcesKernel, gridUnitSideX, gridUnitSideY, 1);
        MassSpringComputeShader.Dispatch  (VelKernel, gridUnitSideX, gridUnitSideY, 1);
    }
    

    private float[] CalculateWindForces() {
        var windValues = new float[VertCount];
        // double for loop through gridResX and gridResY
        for (int i = 0; i < GridResX; i++)
        {
            for (int j = 0; j < GridResY; j++)
            {
                // calculate wind and add it to the external forces buffer based on perlin noise
                float wind = Mathf.PerlinNoise (i*windScale, j*windScale) * windStrength;
                windValues[i + j*GridResX] = wind;
            }
        }
        return windValues;
    }

}
