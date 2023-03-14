using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using UnityEngine.Rendering;
using System;
using UnityEngine.UI;
using System.Linq;

public class BarracudaImageClassifier : MonoBehaviour
{
    [Tooltip("Screen object in the scene")]
    public Transform screen;

    [Header("Data Processing")]
    [Tooltip("Target input dimensions for the model")]
    public int targetDim = 288;
    [Tooltip("Compute shader for GPU processing")]
    public ComputeShader processingShader;
    [Tooltip("Material with fragment shader for GPU processing")]
    public Material processingMaterial;

    [Header("Barracuda")]
    [Tooltip("Barracuda/ONNX model asset file")]
    public NNModel modelAsset;
    [Tooltip("Name of custom softmax output layer")]
    public string softmaxLayer = "softmaxLayer";
    [Tooltip("Target output layer index")]
    public int outputLayerIndex = 0;
    [Tooltip("Option to order tensor data channels first (EXPERIMENTAL)")]
    public bool useNCHW = true;
    [Tooltip("Execution backend for the model")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;

    [Header("Output Processing")]
    [Tooltip("Option to asynchronously download model output from GPU to CPU")]
    public bool useAsyncGPUReadback = true;
    [Tooltip("JSON file with class labels")]
    public TextAsset classLabels;
    [Tooltip("Minimum confidence score for keeping predictions")]
    [Range(0, 1f)]
    public float minConfidence = 0.5f;

    [Header("Debugging")]
    [Tooltip("Option to print debugging messages to the console")]
    public bool printDebugMessages = true;

    [Header("Webcam")]
    [Tooltip("Option to use webcam as input")]
    public bool useWebcam = false;
    [Tooltip("Requested webcam dimensions")]
    public Vector2Int webcamDims = new Vector2Int(1280, 720);
    [Tooltip("Requested webcam framerate")]
    [Range(0, 60)]
    public int webcamFPS = 60;

    [Header("GUI")]
    [Tooltip("Option to display predicted class")]
    public bool displayPredictedClass = true;
    [Tooltip("Option to display fps")]
    public bool displayFPS = true;
    [Tooltip("On-screen text color")]
    public Color textColor = Color.red;
    [Tooltip("Scale value for on-screen font size")]
    [Range(0, 99)]
    public int fontScale = 50;
    [Tooltip("Time in seconds between refreshing fps value")]
    [Range(0.01f, 1.0f)]
    public float fpsRefreshRate = 0.1f;
    [Tooltip("Toggle to use webcam as input source")]
    public Toggle useWebcamToggle;
    [Tooltip("Dropdown menu with available webcam devices")]
    public Dropdown webcamDropdown;

    // Webcam device list
    private WebCamDevice[] webcamDevices;
    // Live webcam video input
    private WebCamTexture webcamTexture;
    // Name of the current webcam device
    private string currentWebcam;

    // Dimensions of the test image
    private Vector2Int imageDims;
    // Texture of the test image
    private Texture imageTexture;
    // Dimensions of the current screen object
    private Vector2Int screenDims;
    // Texture for the model input
    private RenderTexture inputTexture;

    // Main model execution interface
    private IWorker engine;
    // Model input data storage
    private Tensor input;

    // Raw model output on GPU (when using useAsyncGPUReadback)
    private RenderTexture outputTextureGPU;
    // Raw model output on CPU (when using useAsyncGPUReadback)
    private Texture2D outputTextureCPU;

    // Class for reading class labels from JSON file
    class ClassLabels { public string[] classes; }
    // Ordered list of class names
    private string[] classes;
    // Predicted class index
    private int classIndex;

    // Current frame rate value
    private int fps = 0;
    // Timer for updating frame rate value
    private float fpsTimer = 0f;

    // Mean values for image normalization
    private float[] mean = { 0.4850f, 0.4560f, 0.4060f };
    // Standard deviation values for image normalization
    private float[] std = { 0.2290f, 0.2240f, 0.2250f };

    // GPU buffer for mean values
    private ComputeBuffer mean_buffer;
    // GPU buffer for standard deviation values
    private ComputeBuffer std_buffer;

    // Array for storing model output
    private float[] output_array;


    /// <summary>
    /// Initialize the selected webcam device
    /// </summary>
    /// <param name="deviceName">The name of the selected webcam device</param>
    private void InitializeWebcam(string deviceName)
    {
        // Stop playing any existing webcam texture
        if (webcamTexture && webcamTexture.isPlaying) webcamTexture.Stop();

        // Create new webcam texture with specified device name, dimensions, and FPS
        webcamTexture = new WebCamTexture(deviceName, webcamDims.x, webcamDims.y, webcamFPS);

        // Start playing the webcam texture
        webcamTexture.Play();

        // Update the useWebcam flag based on whether the webcam is playing
        useWebcam = webcamTexture.isPlaying;
        useWebcamToggle.SetIsOnWithoutNotify(useWebcam);

        // Log whether the webcam is playing or not
        Debug.Log(useWebcam ? "Webcam is playing" : "Webcam not playing, option disabled");
    }


    /// <summary>
    /// Resize and position an in-scene screen object
    /// </summary>
    private void InitializeScreen()
    {
        // Set the texture of the screen to either the webcam texture or the image texture
        screen.gameObject.GetComponent<MeshRenderer>().material.mainTexture = useWebcam ? webcamTexture : imageTexture;

        // Set the dimensions of the screen to either the dimensions of the webcam texture or the image texture
        screenDims = useWebcam ? new Vector2Int(webcamTexture.width, webcamTexture.height) : imageDims;

        // Set the y-rotation of the screen to 180 degrees if using the webcam, or 0 degrees if using the image texture
        float yRotation = useWebcam ? 180f : 0f;

        // Set the z-scale of the screen to -1 if using the webcam, or 1 if using the image texture
        float zScale = useWebcam ? -1f : 1f;

        // Apply the rotation and scale to the screen
        screen.rotation = Quaternion.Euler(0, yRotation, 0);
        screen.localScale = new Vector3(screenDims.x, screenDims.y, zScale);

        // Position the screen in the center of the screen with a z-value of 1
        screen.position = new Vector3(screenDims.x / 2, screenDims.y / 2, 1);
    }


    /// <summary>
    /// Initialize the GUI dropdown list
    /// </summary>
    private void InitializeDropdown()
    {
        // Create a list to store the names of all available webcams
        List<string> webcamNames = new List<string>();

        // Populate the list with the names of all available webcams
        foreach (WebCamDevice device in webcamDevices) webcamNames.Add(device.name);

        // Clear any existing options from the webcam dropdown
        webcamDropdown.ClearOptions();

        // Add the list of webcam names as options to the webcam dropdown
        webcamDropdown.AddOptions(webcamNames);

        // Set the selected value of the webcam dropdown to the index of the current webcam in the list of webcam names
        webcamDropdown.SetValueWithoutNotify(webcamNames.IndexOf(currentWebcam));
    }


    /// <summary>
    /// Resize and position the main camera based on an in-scene screen object
    /// </summary>
    /// <param name="screenDims">The dimensions of an in-scene screen object</param>
    private void InitializeCamera(Vector2Int screenDims, string cameraName = "Main Camera")
    {
        // Get a reference to the Main Camera GameObject
        GameObject camera = GameObject.Find(cameraName);
        // Adjust the camera position to account for updates to the screenDims
        camera.transform.position = new Vector3(screenDims.x / 2, screenDims.y / 2, -10f);
        // Render objects with no perspective (i.e. 2D)
        camera.GetComponent<Camera>().orthographic = true;
        // Adjust the camera size to account for updates to the screenDims
        camera.GetComponent<Camera>().orthographicSize = screenDims.y / 2;
    }


    /// <summary>
    /// Initialize an interface to execute the specified model using the specified backend
    /// </summary>
    /// <param name="model">The target model representation</param>
    /// <param name="workerType">The target compute backend</param>
    /// <param name="useNCHW">EXPERIMENTAL: The channel order for the compute backend</param>
    /// <returns></returns>
    private IWorker InitializeWorker(Model model, WorkerFactory.Type workerType, bool useNCHW = true)
    {
        // Validate the selected worker type
        workerType = WorkerFactory.ValidateType(workerType);

        // Set the channel order of the compute backend to channel-first
        if (useNCHW) ComputeInfo.channelsOrder = ComputeInfo.ChannelsOrder.NCHW;

        // Create a worker to execute the model using the selected backend
        return WorkerFactory.CreateWorker(workerType, model);
    }


    // Start is called before the first frame update
    void Start()
    {
        // Set the mean and standard deviation values in the processing material
        processingMaterial.SetFloatArray("mean", mean);
        processingMaterial.SetFloatArray("std", std);

        // If the system supports compute shaders, set up the mean and standard deviation values for use in the processing shader
        if (SystemInfo.supportsComputeShaders)
        {
            int kernelIndex = processingShader.FindKernel("NormalizeImage");

            mean_buffer = new ComputeBuffer(mean.Length, sizeof(float));
            mean_buffer.SetData(mean);
            std_buffer = new ComputeBuffer(std.Length, sizeof(float));
            std_buffer.SetData(std);

            processingShader.SetBuffer(kernelIndex, "mean", mean_buffer);
            processingShader.SetBuffer(kernelIndex, "std", std_buffer);
        }

        // Get the source image texture and dimensions
        imageTexture = screen.gameObject.GetComponent<MeshRenderer>().material.mainTexture;
        imageDims = new Vector2Int(imageTexture.width, imageTexture.height);

        // Get the list of available webcam devices
        webcamDevices = WebCamTexture.devices;
        currentWebcam = webcamDevices[0].name;
        useWebcam = webcamDevices.Length > 0 ? useWebcam : false;

        // Initialize the webcam if available
        if (useWebcam) InitializeWebcam(currentWebcam);

        // Initialize the screen and camera
        InitializeScreen();
        InitializeCamera(screenDims);

        // Load and modify the model, and initialize the engine for executing the model
        Model m_RunTimeModel = ModelLoader.Load(modelAsset);
        string outputLayer = m_RunTimeModel.outputs[outputLayerIndex];
        ModelBuilder modelBuilder = new ModelBuilder(m_RunTimeModel);
        modelBuilder.Softmax(softmaxLayer, outputLayer);
        engine = InitializeWorker(modelBuilder.model, workerType, useNCHW);

        // Load the class labels and initialize the output arrays
        classes = JsonUtility.FromJson<ClassLabels>(classLabels.text).classes;
        output_array = new float[classes.Length];

        // Initialize the GPU and CPU output textures
        outputTextureGPU = RenderTexture.GetTemporary(1, classes.Length, 24, RenderTextureFormat.ARGBHalf);
        outputTextureCPU = new Texture2D(1, classes.Length, TextureFormat.RGBAHalf, false);

        // Initialize the webcam dropdown list
        InitializeDropdown();
    }


    /// <summary>
    /// Process the provided image using the specified function on the GPU
    /// </summary>
    /// <param name="image">The target image RenderTexture</param>
    /// <param name="computeShader">The target ComputerShader</param>
    /// <param name="functionName">The target ComputeShader function</param>
    /// <returns></returns>
    private void ProcessImageGPU(RenderTexture image, ComputeShader computeShader, string functionName)
    {
        // Set the number of threads to be used on the GPU
        int numthreads = 8;

        // Find the index of the specified function in the compute shader
        int kernelHandle = computeShader.FindKernel(functionName);

        // Create a temporary HDR render texture
        RenderTexture result = RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);

        // Enable random write access to the render texture
        result.enableRandomWrite = true;
        result.Create();

        // Set the Result and InputImage variables in the compute shader to the appropriate textures
        computeShader.SetTexture(kernelHandle, "Result", result);
        computeShader.SetTexture(kernelHandle, "InputImage", image);

        // Dispatch the compute shader
        computeShader.Dispatch(kernelHandle, result.width / numthreads, result.height / numthreads, 1);

        // Copy the result of the compute shader into the source render texture
        Graphics.Blit(result, image);

        // Release the temporary render texture
        RenderTexture.ReleaseTemporary(result);
    }


    /// <summary>
    /// Scale the source image resolution to the target input dimensions
    /// while maintaing the source aspect ratio.
    /// </summary>
    /// <param name="imageDims"></param>
    /// <param name="targetDims"></param>
    /// <returns></returns>
    private Vector2Int CalculateInputDims(Vector2Int imageDims, int targetDim)
    {
        // Ensure the target dimension is at least 64px
        targetDim = Mathf.Max(targetDim, 64);

        Vector2Int inputDims = new Vector2Int();

        // Calculate the input dimensions to maintain aspect ratio while meeting the target dimension
        if (imageDims.x >= imageDims.y)
        {
            inputDims[0] = (int)(imageDims.x / ((float)imageDims.y / (float)targetDim));
            inputDims[1] = targetDim;
        }
        else
        {
            inputDims[0] = targetDim;
            inputDims[1] = (int)(imageDims.y / ((float)imageDims.x / (float)targetDim));
        }

        return inputDims;
    }


    /// <summary>
    /// Called once AsyncGPUReadback has been completed
    /// </summary>
    /// <param name="request"></param>
    private void OnCompleteReadback(AsyncGPUReadbackRequest request)
    {
        // Check if the request has an error
        if (request.hasError)
        {
            Debug.Log("GPU readback error detected.");
            return;
        }

        // Make sure the Texture2D is not null
        if (outputTextureCPU != null)
        {
            // Fill the Texture2D with raw data from the request
            outputTextureCPU.LoadRawTextureData(request.GetData<uint>());
            // Apply changes to the Texture2D
            outputTextureCPU.Apply();
        }
    }


    /// <summary>
    /// Process the raw model output to get the predicted class index
    /// </summary>
    /// <param name="engine">The interface for executing the model</param>
    /// <returns></returns>
    private float[] ProcessOutput(IWorker engine)
    {
        float[] output_array = new float[classes.Length];

        // Retrieve the raw model output
        Tensor output = engine.PeekOutput(softmaxLayer);

        if (useAsyncGPUReadback)
        {
            // Reshape the output tensor to a 1xNx1x1 shape
            output = output.Reshape(new TensorShape(1, classes.Length, 1, 1));

            // Copy the output tensor to a RenderTexture
            output.ToRenderTexture(outputTextureGPU);
            // Asynchronously download the RenderTexture data to the CPU
            AsyncGPUReadback.Request(outputTextureGPU, 0, TextureFormat.RGBAHalf, OnCompleteReadback);
            // Convert the Color array to a float array of predicted class scores
            Color[] output_colors = outputTextureCPU.GetPixels();
            output_array = output_colors.Select(color => color.r).Reverse().ToArray();
        }
        else
        {
            // Download the output tensor data to the CPU
            output_array = output.data.Download(output.shape);
        }

        // Clean up the Tensor and associated memory
        output.Dispose();

        return output_array;
    }


    // Update is called once per frame
    void Update()
    {
        // Check if there are any available webcam devices and set `useWebcam` accordingly
        useWebcam = webcamDevices.Length > 0 ? useWebcam : false;
        if (useWebcam)
        {
            // Initialize webcam if it is not already playing
            if (!webcamTexture || !webcamTexture.isPlaying) InitializeWebcam(currentWebcam);

            // Return if webcam is not initialized
            if (webcamTexture.width <= 16) return;

            // Resize and position screen and camera if dimensions don't match
            if (screenDims.x != webcamTexture.width)
            {
                InitializeScreen();
                InitializeCamera(screenDims);
            }
        }
        // If webcam is not being used and it's currently playing, stop it
        else if (webcamTexture && webcamTexture.isPlaying)
        {
            webcamTexture.Stop();

            InitializeScreen();
            InitializeCamera(screenDims);
        }
        // Set imageTexture to screen material main texture if not using webcam
        else
        {
            imageTexture = screen.gameObject.GetComponent<MeshRenderer>().material.mainTexture;
        }

        // Calculate the input dimensions
        Vector2Int inputDims = CalculateInputDims(screenDims, targetDim);
        if (printDebugMessages) Debug.Log($"Input Dims: {inputDims.x} x {inputDims.y}");

        // Create the input texture with the calculated input dimensions
        inputTexture = RenderTexture.GetTemporary(inputDims.x, inputDims.y, 24, RenderTextureFormat.ARGBHalf);
        if (printDebugMessages) Debug.Log($"Input Dims: {inputTexture.width}x{inputTexture.height}");

        // Copy the source texture into the input texture
        Graphics.Blit((useWebcam ? webcamTexture : imageTexture), inputTexture);

        // Check if the model is using a Compute Shader backend
        useAsyncGPUReadback = engine.Summary().Contains("Unity.Barracuda.ComputeVarsWithSharedModel") ? useAsyncGPUReadback : false;

        // Process the input texture using a Compute Shader or Graphics.Blit
        if (SystemInfo.supportsComputeShaders)
        {
            ProcessImageGPU(inputTexture, processingShader, "NormalizeImage");
            input = new Tensor(inputTexture, channels: 3);
        }
        else
        {
            // Create a temporary RenderTexture for preprocessing
            RenderTexture result = RenderTexture.GetTemporary(inputTexture.width, inputTexture.height, 24, RenderTextureFormat.ARGBHalf);
            RenderTexture.active = result;

            // Apply preprocessing steps
            Graphics.Blit(inputTexture, result, processingMaterial);

            // Initialize a Tensor using the result RenderTexture
            input = new Tensor(result, channels: 3);
            RenderTexture.ReleaseTemporary(result);
        }

        // Execute the model with the input Tensor
        engine.Execute(input);
        // Dispose Tensor and release associated memory.
        input.Dispose();

        // Release the input texture
        RenderTexture.ReleaseTemporary(inputTexture);
        // Get the model output
        output_array = ProcessOutput(engine);

        // Unload unused assets in web browser
        if (Application.platform == RuntimePlatform.WebGLPlayer) Resources.UnloadUnusedAssets();
    }


    /// <summary>
    /// This method is called when the value for the webcam toggle changes
    /// </summary>
    /// <param name="useWebcam"></param>
    public void UpdateWebcamToggle(bool useWebcam)
    {
        this.useWebcam = useWebcam;
    }


    /// <summary>
    /// Set the current webcam device to the one selected in the webcam dropdown
    /// </summary>
    public void UpdateWebcamDevice()
    {
        currentWebcam = webcamDevices[webcamDropdown.value].name;
        Debug.Log($"Selected Webcam: {currentWebcam}");
        
        // If the useWebcam flag is true, initialize the selected webcam
        if (useWebcam) InitializeWebcam(currentWebcam);

        // Initialize the screen and camera
        InitializeScreen();
        InitializeCamera(screenDims);
    }


    /// <summary>
    /// Update the minimum confidence score for keeping predictions
    /// </summary>
    /// <param name="slider"></param>
    public void UpdateConfidenceThreshold(Slider slider)
    {
        minConfidence = slider.value;
    }


    // Display GUI elements on the screen
    public void OnGUI()
    {
        // Set up the style for the GUI labels
        GUIStyle style = new GUIStyle
        {
            fontSize = (int)(Screen.width * (1f / (100f - fontScale)))
        };
        style.normal.textColor = textColor;

        // Define the rectangular positions for the two labels
        Rect slot1 = new Rect(10, 10, 500, 500);
        Rect slot2 = new Rect(10, style.fontSize * 1.5f, 500, 500);

        // Return if the output array is empty
        if (output_array.Length <= 0) return;

        // Find the index of the maximum value in the output array
        classIndex = Array.IndexOf(output_array, output_array.Max());

        // Calculate the label text based on the predicted class and its confidence
        string labelText = $"{classes[classIndex]} {(output_array[classIndex] * 100).ToString("0.##")}%";
        if (output_array[classIndex] < minConfidence) labelText = "None";

        // Determine the content for the first label
        bool validIndex = classIndex >= 0 && classIndex < classes.Length;
        string content = validIndex ? $"Predicted Class: {labelText}" : "Loading Model...";
        if (displayPredictedClass) GUI.Label(slot1, new GUIContent(content), style);

        // Calculate and display the FPS
        if (Time.unscaledTime > fpsTimer)
        {
            fps = (int)(1f / Time.unscaledDeltaTime);
            fpsTimer = Time.unscaledTime + fpsRefreshRate;
        }

        // Adjust screen position when not showing predicted class
        Rect fpsRect = displayPredictedClass ? slot2 : slot1;
        if (displayFPS) GUI.Label(fpsRect, new GUIContent($"FPS: {fps}"), style);
    }


    // Clean up resources when the object is disabled
    private void OnDisable()
    {
        // Release the mean and standard deviation buffers if compute shaders are supported
        if (SystemInfo.supportsComputeShaders)
        {
            mean_buffer.Release();
            std_buffer.Release();
        }

        // Release the temporary output texture
        RenderTexture.ReleaseTemporary(outputTextureGPU);

        // Dispose of the inference engine
        engine.Dispose();
    }
}
