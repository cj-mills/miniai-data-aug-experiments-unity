using CJM.BarracudaInferenceToolkit;
using System;
using System.Linq;
using UnityEngine;

/// <summary>
/// The InferenceController class manages the process of running the inference on the input image
/// and updating the UI with the results.
/// </summary>
public class InferenceController : MonoBehaviour
{
    [Header("Components")]
    [SerializeField] private ImageProcessor imageProcessor;
    [SerializeField] private MultiClassImageClassifier modelRunner;
    [SerializeField] private InferenceUI inferenceUI;

    [Header("Settings")]
    [SerializeField] private MeshRenderer screenRenderer;
    [SerializeField] private bool printDebugMessages = false;

    private void Update()
    {
        if (!AreComponentsValid()) return;

        var imageTexture = screenRenderer.material.mainTexture;
        var screenDims = new Vector2Int(imageTexture.width, imageTexture.height);
        var inputDims = imageProcessor.CalculateInputDims(screenDims);

        var inputTexture = RenderTexture.GetTemporary(inputDims.x, inputDims.y, 24, RenderTextureFormat.ARGBHalf);
        Graphics.Blit(imageTexture, inputTexture);
        ProcessInputImage(inputTexture);
        var outputArray = modelRunner.ExecuteModel(inputTexture);
        UpdateUI(outputArray);
        RenderTexture.ReleaseTemporary(inputTexture);
    }

    /// <summary>
    /// Checks if all required components are assigned and valid.
    /// </summary>
    /// <returns>True if all components are valid, false otherwise.</returns>
    private bool AreComponentsValid()
    {
        if (imageProcessor == null || modelRunner == null || inferenceUI == null)
        {
            Debug.LogError("InferenceController requires ImageProcessor, ModelRunner, and InferenceUI components.");
            return false;
        }
        return true;
    }

    /// <summary>
    /// Processes the input image using the image processor.
    /// </summary>
    /// <param name="inputTexture">The input texture to be processed.</param>
    private void ProcessInputImage(RenderTexture inputTexture)
    {
        if (SystemInfo.supportsComputeShaders)
        {
            imageProcessor.ProcessImageComputeShader(inputTexture, "NormalizeImage");
        }
        else
        {
            imageProcessor.ProcessImageShader(inputTexture);
        }
    }

    /// <summary>
    /// Updates the UI with the results from the output array.
    /// </summary>
    /// <param name="outputArray">The output array from the model execution.</param>
    private void UpdateUI(float[] outputArray)
    {
        if (outputArray.Length <= 0) outputArray = new float[] { 0f };

        float confidenceScore = outputArray.Max();
        int classIndex = Array.IndexOf(outputArray, confidenceScore);
        bool modelLoaded = outputArray.Min() >= 0f && confidenceScore <= 1f;

        string className = modelRunner.GetClassName(classIndex);
        inferenceUI.UpdateUI(className, confidenceScore, modelLoaded);

        if (printDebugMessages) Debug.Log($"Output Array: {string.Join(", ", outputArray)}");
    }
}
