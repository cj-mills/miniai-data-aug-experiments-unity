using UnityEngine;
using UnityEngine.UI;

// Handle clicks on images in a Scroll View
public class ImageClickHandler : MonoBehaviour
{
    // The GameObject to set the material for
    public GameObject quadObject;

    // Set up the image click listeners
    void Start()
    {
        // Get all the images in the Scroll View
        Image[] images = transform.Find("Scroll View/Viewport/Content").GetComponentsInChildren<Image>();

        // Add a click listener to each image
        foreach (Image image in images)
        {
            Button button = image.GetComponent<Button>();
            if (button == null)
            {
                // Add a Button component to the image if it doesn't already have one
                button = image.gameObject.AddComponent<Button>();
            }

            // Set the click listener to call SetQuadMaterial with the clicked image
            button.onClick.AddListener(() => SetQuadMaterial(image));
        }
    }

    // Debug log the source of an image
    void PrintImageSource(Image image)
    {
        Debug.Log(image.name);
    }

    // Set the material's main texture to the clicked image's texture
    void SetQuadMaterial(Image image)
    {
        // Get the current texture name
        string texture_name = quadObject.GetComponent<Renderer>().material.mainTexture.name;

        // Return if the texture name is empty
        if (texture_name.Length == 0) return;

        // Set the material's main texture to the clicked image's texture
        quadObject.GetComponent<Renderer>().material.mainTexture = image.mainTexture;
    }
}

