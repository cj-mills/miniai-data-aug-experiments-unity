using UnityEngine;
using UnityEngine.UI;
using TMPro;

/// <summary>
/// InferenceUI is responsible for displaying the predicted class and FPS on the screen.
/// </summary>
public class InferenceUI : MonoBehaviour
{
    [Header("UI Components")]
    [SerializeField] private Slider confidenceThresholdSlider;
    [SerializeField] private TextMeshProUGUI predictedClassText;
    [SerializeField] private TextMeshProUGUI fpsText;

    [Header("Settings")]
    [SerializeField, Tooltip("On-screen text color")]
    private Color textColor = Color.red;
    [SerializeField, Tooltip("Maximum font size for on-screen text"), Range(10, 99)]
    private int maxFontSize = 24;
    [SerializeField, Tooltip("Time in seconds between refreshing fps value"), Range(0.01f, 1.0f)]
    private float fpsRefreshRate = 0.1f;
    [SerializeField, Tooltip("Option to display fps")]
    private bool displayFPS = true;

    private float minConfidence;
    private string className;
    private float confidenceScore;
    private bool modelLoaded;
    private float fpsTimer;

    /// <summary>
    /// Initializes the UI components and sets the confidence threshold.
    /// </summary>
    private void Start()
    {
        confidenceThresholdSlider.onValueChanged.AddListener(UpdateConfidenceThreshold);
        minConfidence = confidenceThresholdSlider.value;

        // Set up Auto Size for predictedClassText and fpsText
        SetAutoSize(predictedClassText);
        SetAutoSize(fpsText);
    }

    /// <summary>
    /// Enables Auto Size and sets the minimum and maximum font size for the given TextMeshProUGUI object.
    /// </summary>
    /// <param name="textObject">The TextMeshProUGUI object to configure.</param>
    private void SetAutoSize(TextMeshProUGUI textObject)
    {
        textObject.enableAutoSizing = true;
        textObject.fontSizeMin = 0;
        textObject.fontSizeMax = maxFontSize;
    }

    /// <summary>
    /// Updates the UI with the provided class name, confidence score, and model load status.
    /// </summary>
    /// <param name="className">The predicted class name.</param>
    /// <param name="confidenceScore">The confidence score of the predicted class.</param>
    /// <param name="modelLoaded">Indicates whether the model is loaded.</param>
    public void UpdateUI(string className, float confidenceScore, bool modelLoaded)
    {
        this.className = className;
        this.confidenceScore = confidenceScore;
        this.modelLoaded = modelLoaded;

        UpdatePredictedClass();
    }

    /// <summary>
    /// Updates the FPS display if the displayFPS option is enabled.
    /// </summary>
    private void Update()
    {
        if (displayFPS)
        {
            UpdateFPS();
        }
    }

    /// <summary>
    /// Updates the displayed predicted class and its confidence score.
    /// </summary>
    private void UpdatePredictedClass()
    {
        string labelText = $"{className} {(confidenceScore * 100).ToString("0.##")}%";
        if (confidenceScore < minConfidence) labelText = "None";

        string content = modelLoaded ? $"Predicted Class: {labelText}" : "Loading Model...";
        predictedClassText.text = content;
        predictedClassText.color = textColor;
    }

    /// <summary>
    /// Updates the displayed FPS value.
    /// </summary>
    private void UpdateFPS()
    {
        if (Time.unscaledTime > fpsTimer)
        {
            int fps = (int)(1f / Time.unscaledDeltaTime);
            fpsText.text = $"FPS: {fps}";
            fpsText.color = textColor;

            fpsTimer = Time.unscaledTime + fpsRefreshRate;
        }
    }

    /// <summary>
    /// Updates the minimum confidence threshold for displaying the predicted class.
    /// </summary>
    /// <param name="value">The new minimum confidence threshold value.</param>
    private void UpdateConfidenceThreshold(float value)
    {
        minConfidence = value;
    }
}

