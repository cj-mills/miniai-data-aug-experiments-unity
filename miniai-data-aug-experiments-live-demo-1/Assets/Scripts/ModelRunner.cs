using System.Linq;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.Barracuda;

public class ModelRunner : MonoBehaviour
{
    [Header("Model Assets")]
    [SerializeField] private NNModel model;
    [SerializeField] private string softmaxLayer = "softmaxLayer";
    [Tooltip("Target output layer index")]
    [SerializeField] private int outputLayerIndex = 0;
    [Tooltip("Option to order tensor data channels first (EXPERIMENTAL)")]
    [SerializeField] private bool useNCHW = true;
    [Tooltip("Execution backend for the model")]
    [SerializeField] private WorkerFactory.Type workerType = WorkerFactory.Type.Auto;

    [Header("Output Processing")]
    [Tooltip("Option to asynchronously download model output from GPU to CPU")]
    [SerializeField] private bool useAsyncGPUReadback = true;
    [Tooltip("JSON file with class labels")]
    [SerializeField] private TextAsset classLabels;

    private ModelBuilder modelBuilder;
    private class ClassLabels { public string[] classes; }
    private string[] classes;
    private IWorker engine;
    private Texture2D outputTextureCPU;

    /// <summary>
    /// Initialize necessary components during the start of the script.
    /// </summary>
    private void Start()
    {
        CheckAsyncGPUReadbackSupport();
        LoadAndPrepareModel();
        InitializeEngine();
        LoadClassLabels();
        CreateOutputTexture();
    }

    /// <summary>
    /// Check if the system supports async GPU readback and update the flag accordingly.
    /// </summary>
    private void CheckAsyncGPUReadbackSupport()
    {
        if (!SystemInfo.supportsAsyncGPUReadback)
        {
            useAsyncGPUReadback = false;
        }
    }

    /// <summary>
    /// Load the model and prepare it for execution by applying softmax to the output layer.
    /// </summary>
    private void LoadAndPrepareModel()
    {
        Model runtimeModel = ModelLoader.Load(model);
        string outputLayer = runtimeModel.outputs[outputLayerIndex];
        modelBuilder = new ModelBuilder(runtimeModel);
        modelBuilder.Softmax(softmaxLayer, outputLayer);
    }

    /// <summary>
    /// Initialize the inference engine and check if the model is using a Compute Shader backend.
    /// </summary>
    private void InitializeEngine()
    {
        engine = InitializeWorker(modelBuilder.model, workerType, useNCHW);

        // Check if the model is using a Compute Shader backend
        useAsyncGPUReadback = engine.Summary().Contains("Unity.Barracuda.ComputeVarsWithSharedModel") ? useAsyncGPUReadback : false;
    }

    /// <summary>
    /// Load the class labels from the provided JSON file.
    /// </summary>
    private void LoadClassLabels()
    {
        classes = JsonUtility.FromJson<ClassLabels>(classLabels.text).classes;
    }

    /// <summary>
    /// Create the output texture that will store the model output.
    /// </summary>
    private void CreateOutputTexture()
    {
        outputTextureCPU = new Texture2D(1, classes.Length, TextureFormat.RGBAFloat, false);
    }

    /// <summary>
    /// Initialize the worker for executing the model with the specified backend and channel order.
    /// </summary>
    /// <param name="model">The target model representation.</param>
    /// <param name="workerType">The target compute backend.</param>
    /// <param name="useNCHW">The channel order for the compute backend (default is true).</param>
    /// <returns>An initialized worker instance.</returns>
    private IWorker InitializeWorker(Model model, WorkerFactory.Type workerType, bool useNCHW = true)
    {
        workerType = WorkerFactory.ValidateType(workerType);

        if (useNCHW) ComputeInfo.channelsOrder = ComputeInfo.ChannelsOrder.NCHW;

        return WorkerFactory.CreateWorker(workerType, model);
    }


    /// <summary>
    /// Execute the model on the provided input texture and return the output array.
    /// </summary>
    /// <param name="inputTexture">The input texture for the model.</param>
    /// <returns>The output array of the model.</returns>
    public float[] ExecuteModel(RenderTexture inputTexture)
    {
        using (Tensor input = new Tensor(inputTexture, channels: 3))
        {
            engine.Execute(input);
            return ProcessOutput(engine);
        }
    }

    /// <summary>
    /// Process the output of the model execution.
    /// </summary>
    /// <param name="engine">The inference engine used to execute the model.</param>
    /// <returns>The output array of the model.</returns>
    private float[] ProcessOutput(IWorker engine)
    {
        float[] outputArray = new float[classes.Length];

        using (Tensor output = engine.PeekOutput(softmaxLayer))
        {
            if (useAsyncGPUReadback)
            {
                Tensor reshapedOutput = output.Reshape(new TensorShape(1, classes.Length, 1, 1));
                AsyncGPUReadback.Request(reshapedOutput.ToRenderTexture(), 0, TextureFormat.RGBAFloat, OnCompleteReadback);
                Color[] outputColors = outputTextureCPU.GetPixels();
                outputArray = outputColors.Select(color => color.r).Reverse().ToArray();
            }
            else
            {
                outputArray = output.data.Download(output.shape);
            }
        }

        return outputArray;
    }

    /// <summary>
    /// Get the class name corresponding to the provided class index.
    /// </summary>
    /// <param name="classIndex">The index of the class to retrieve.</param>
    /// <returns>The class name corresponding to the class index.</returns>
    public string GetClassName(int classIndex)
    {
        return classes[classIndex];
    }

    /// <summary>
    /// Callback method for handling the completion of async GPU readback.
    /// </summary>
    /// <param name="request">The async GPU readback request.</param>
    private void OnCompleteReadback(AsyncGPUReadbackRequest request)
    {
        if (request.hasError)
        {
            Debug.Log("GPU readback error detected.");
            return;
        }

        if (outputTextureCPU != null)
        {
            outputTextureCPU.LoadRawTextureData(request.GetData<uint>());
            outputTextureCPU.Apply();
        }
    }

    /// <summary>
    /// Clean up resources when the component is disabled.
    /// </summary>
    private void OnDisable()
    {
        engine.Dispose();
    }

}
