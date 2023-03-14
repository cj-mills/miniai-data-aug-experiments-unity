Shader "Processing Shaders/NormalizeImage"
{
    Properties
    {
        _MainTex("Texture", 2D) = "white" {}
    }
    SubShader
    {
        // No culling or depth
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            // Uniform arrays to hold the mean and standard deviation values for each color channel (r, g, b)
            uniform float mean[3];
            uniform float std[3];

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            sampler2D _MainTex;

            // Fragment shader function
            float4 frag(v2f i) : SV_Target
            {
                // Sample the input image
                float4 col = tex2D(_MainTex, i.uv);
                // Normalize each color channel (r, g, b)
                col.r = (col.r - mean[0]) / std[0];
                col.g = (col.g - mean[1]) / std[1];
                col.b = (col.b - mean[2]) / std[2];
                // Return the normalized color values
                return col;
            }
            ENDCG
        }
    }
}
