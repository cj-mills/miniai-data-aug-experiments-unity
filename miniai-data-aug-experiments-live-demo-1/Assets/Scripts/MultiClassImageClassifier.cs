using System.Linq;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.Barracuda;

namespace BarracudaInferenceToolkit
{
    public class MultiClassImageClassifier : BarracudaModelRunner
    {
        [Header("Output Processing")]
        [SerializeField] private string softmaxLayer = "softmaxLayer";
        [Tooltip("Target output layer index")]
        [SerializeField] private int outputLayerIndex = 0;
        [Tooltip("Option to asynchronously download model output from GPU to CPU")]
        [SerializeField] private bool useAsyncGPUReadback = true;
        [Tooltip("JSON file with class labels")]
        [SerializeField] private TextAsset classLabels;

        // Helper class for deserializing class labels from the JSON file
        private class ClassLabels { public string[] classes; }

        private string[] classes;
        private Texture2D outputTextureCPU;

        /// <summary>
        /// Initialize necessary components during the start of the script.
        /// </summary>
        protected override void Start()
        {
            base.Start();
            CheckAsyncGPUReadbackSupport();
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
        protected override void LoadAndPrepareModel()
        {
            // Load and prepare the model with the base implementation
            base.LoadAndPrepareModel();

            // Get the output layer name
            string outputLayer = modelBuilder.model.outputs[outputLayerIndex];

            // Add the Softmax layer
            modelBuilder.Softmax(softmaxLayer, outputLayer);
        }

        /// <summary>
        /// Initialize the inference engine and check if the model is using a Compute Shader backend.
        /// </summary>
        protected override void InitializeEngine()
        {
            base.InitializeEngine();

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
            outputTextureCPU = new Texture2D(1, classes.Length, TextureFormat.ARGB32, false);
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
                base.ExecuteModel(input);
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
                    AsyncGPUReadback.Request(reshapedOutput.ToRenderTexture(), 0, TextureFormat.ARGB32, OnCompleteReadback);
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
    }
}