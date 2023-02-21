Shader "yuanshen" 
{
    Properties 
    {
        [Header(Main Texture Setting)]
        [Space(5)]
        _BaseMap ("BaseMap", 2D) = "white" {}
        [HDR][MainColor]_BaseColor ("_BaseColor", Color) = (1, 1, 1, 1)
        _WorldLightInfluence ("World Light Influence", range(0.0, 1.0)) = 0.1
        [Space(30)]

        [Header(Shadow Setting)]
        [Space(5)]
        _LightMap ("LightMap", 2D) = "grey" {}
        _ShadowArea ("一级阴影面积", range(0.0, 1.0)) = 0.5
        _ShadowSmooth ("一级阴影平滑", range(0.0, 1.0)) = 0.05
        _DarkShadowArea ("二级阴影面积", range(0.0, 1.0)) = 0.5
        _DarkShadowSmooth ("二级阴影平滑", range(0.0, 1.0)) = 0.05
        [Toggle]_FixDarkShadow ("固定二级阴影颜色开关", float) = 1
        _ShadowMultColor ("阴影颜色", Color) = (0.5, 0.5, 0.5, 1.0)
        _DarkShadowMultColor ("暗阴影颜色", Color) = (0.5, 0.5, 0.5, 1.0)
        [Toggle]_IgnoreLightY ("固定灯光开关", float) = 0
        _FixLightY ("固定灯光y范围", range(-10.0, 10.0)) = 0.0
        [Space(30)]

        [Header(Shadow Setting)]
        [Space(5)]
        [Toggle(ENABLE_FACE_SHADOW_MAP)]_EnableFaceShadowMap ("Enable Face Shadow Map", float) = 0
        _FaceShadowMap ("Face Shadow Map", 2D) = "white" { }
        [Space(30)]

        [Header(Shadow Ramp)]
        [Space(5)]
        [Toggle(ENABLE_RAMP_SHADOW)] _EnableRampShadow ("Enable Ramp Shadow", float) = 1
        _RampMap ("Shadow Ramp Texture", 2D) = "white" { }
        _RampArea12 ("XY:1 ZW:2", Vector) = (-50, 1, -50, 4)
        _RampArea34 ("XY:3 ZW:4", Vector) = (-50, 0, -50, 2)
        _RampArea5 ("XY:5", Vector) = (-50, 3, -50, 0)
        _RampShadowRange ("Ramp Shadow Range", range(0.0, 1.0)) = 0.8
        //_Night ("_Night", range(-1.0, 1.0)) = 1
        _InNight ("In Night", Range(0,1)) = 1
        [Space(30)]

        [Header(Specular Setting)]
        [Space(5)]
        [Toggle]_EnableSpecular ("Enable Specular", Float) = 1
		_SpecularShift("_SpecularShift" , 2D) = "white"{}


        [HDR]_LightSpecColor ("Specular Color", color) = (0.8, 0.8, 0.8, 1)
        _Shininess ("Shininess", range(0.1, 20.0)) = 10.0
        _SpecMulti ("Multiple Factor", range(0.1, 1.0)) = 1
        [Space(30)]

        [Header(RimLight Setting)]
        [Space(5)]
        [Toggle]_EnableLambert ("Enable Lambert", float) = 1
        [Toggle]_EnableRim ("Enable Rim", float) = 1
        [HDR]_RimColor ("Rim Color", Color) = (1, 1, 1, 1)
        _RimSmooth ("Rim Smooth", Range(0.001, 1.0)) = 0.01
        _RimPow ("Rim Pow", Range(0.0, 10.0)) = 1.0
        [Space(5)]
        [Toggle]_EnableRimDS ("Enable Dark Side Rim", int) = 1
        [HDR]_DarkSideRimColor ("DarkSide Rim Color", Color) = (1, 1, 1, 1)
        _DarkSideRimSmooth ("DarkSide Rim Smooth", Range(0.001, 1.0)) = 0.01
        _DarkSideRimPow ("DarkSide Rim Pow", Range(0.0, 10.0)) = 1.0
        [Space(30)]

        [Header(Outline)]
        [Space(5)]
        _outlinecolor ("outline color", Color) = (0,0,0,1)
        _outlinewidth ("outline width", Range(0, 1)) = 0.01



    }

    SubShader
    {
        Tags
        {
            "RenderPipeline"="UniversalRenderPipeline"
            "RenderType"="Opaque"
        }



        HLSLINCLUDE
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

        CBUFFER_START(UnityPerMaterial) 

        float4 _BaseMap_ST;
        float4 _BaseColor;
        half _WorldLightInfluence;

        float4 _LightMap_ST;
        float4 _FaceShadowMap_ST;

        uniform float _ShadowArea; //一级阴影面积
        uniform float _DarkShadowArea; //二级阴影面积
        uniform float _FixDarkShadow; //是否固定二级阴影颜色
        uniform float _ShadowSmooth; //一级阴影平滑
        uniform float _DarkShadowSmooth; //二级阴影平滑

        uniform float4 _ShadowMultColor; //阴影颜色
        uniform float4 _DarkShadowMultColor; //暗阴影颜色

        uniform half4 _RampArea12;
        uniform half4 _RampArea34;
        uniform half2 _RampArea5;
        uniform float _RampShadowRange;
        uniform float _Night;

        float _EnableLambert;
        float _EnableRim;
        half4 _RimColor;
        float _RimSmooth;
        float _RimPow;
        float _EnableRimDS;
        half4 _DarkSideRimColor;
        float _DarkSideRimSmooth;
        float _DarkSideRimPow;

        float _EnableSpecular;
        float4 _LightSpecColor;
        float _Shininess;
        float _SpecMulti;
        uniform float _IgnoreLightY;
        uniform float _FixLightY;

        float _BloomFactor;
        half3 _EmissionColor;
        float _Emission;
        float _EmissionBloomFactor;

        uniform float4 _outlinecolor;
        uniform float _outlinewidth;
        half _InNight;




 

        CBUFFER_END
  

        TEXTURE2D(_BaseMap);
        SAMPLER(sampler_BaseMap);

        TEXTURE2D(_LightMap);
        SAMPLER(sampler_LightMap);

        TEXTURE2D(_FaceShadowMap);
        SAMPLER(sampler_FaceShadowMap);

        TEXTURE2D(_RampMap);
        SAMPLER(sampler_RampMap);

        TEXTURE2D(_CameraDepthTexture);
        SAMPLER(sampler_CameraDepthTexture);

	//	TEXTURE2D(_SpecularShift);
	//	 SAMPLER( sampler_SpecularShift);
	sampler2D _SpecularShift;
        float4 _SpecularShift_ST;
        ENDHLSL

        Pass
        {
            Name "FORWARD"
            Tags
            {
                "LightMode"="UniversalForward"
            }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS_CASCADE
            #pragma multi_compile _ _SHADOWS_SOFT//柔化阴影，得到软阴影

            #pragma shader_feature_local_fragment ENABLE_FACE_SHADOW_MAP
            #pragma shader_feature_local_fragment ENABLE_RAMP_SHADOW
        

            struct VertexInput //输入结构
            {
                float3 posOS : POSITION; // 顶点信息 Get✔
                half4 color: COLOR0;
                float2 uv0 : TEXCOORD1; // UV信息 Get✔
                float4 normalOS : NORMAL; // 法线信息 Get✔
                float4 tangentOS : TANGENT; // 切线信息 Get✔
            };

            struct VertexOutput //输出结构
            {
                float4 posCS : POSITION; // 屏幕顶点位置
                float4 color: COLOR0;
                float2 uv0 : TEXCOORD1; // UV0
                float3 posWS : TEXCOORD2; // 世界空间顶点位置
                float3 posVS: TEXCOORD3;
                float3 nDirWS : TEXCOORD4; // 世界空间法线方向
                float3 nDirVS :TEXCOORD5;
                float4 posNDC :TEXCOORD6;
				float3 tangentWS:TEXCOORD7;
				float3 bitangentWS :TEXCOORD8;
            };


 
                 float3 NPR_Base_RimLight(float NdotV,float NdotL,float3 baseColor)
                {
                    return (1 - smoothstep(_DarkSideRimSmooth,_DarkSideRimSmooth + 0.03,NdotV)) * _DarkSideRimPow * (1 - (NdotL * 0.5 + 0.5 )) * baseColor;
                }
                // float SobelSampleDepth(half2 uv , half2 offset , half DepthZ)
                // {
                //      half pixelCenter = DepthZ;
                //      half pixelLeft =               
                // }
        
            VertexOutput vert(VertexInput v) //顶点shader
            {
                VertexOutput o = (VertexOutput)0; // 新建输出结构
                o.color = v.color;
                VertexPositionInputs  vertexInput = GetVertexPositionInputs(v.posOS);

                o.nDirWS = TransformObjectToWorldNormal(v.normalOS);
				o.tangentWS = TransformObjectToWorldDir(v.tangentOS);
				o.bitangentWS = TransformObjectToWorldDir(cross(v.normalOS , v.tangentOS));
                o.posWS = TransformObjectToWorld(v.posOS);
                o.posVS = TransformWorldToView(o.posWS);
                o.posNDC = vertexInput.positionNDC;
                o.posCS = TransformWorldToHClip(o.posWS);
                o.nDirVS = TransformWorldToViewDir(o.nDirWS);
            

           
                 o.uv0 = v.uv0; // 传递UV
                o.uv0 = float2(o.uv0.x,  o.uv0.y);

                return o; // 返回输出结构
            }   

        float GetSSRimScale(float z)
            {
                float w = (1.0 / (PositivePow(z + saturate(UNITY_MATRIX_P._m00), 1.5) + 0.75)) * (_ScreenParams.y / 1080);
                w *= lerp(1, UNITY_MATRIX_P._m00, 0.60 * saturate(0.25 * z * z));
                return w < 0.01 ? 0: w;
            }

			//-----------------------------------------------------------------
			half3 ShiftedTangent(float3 t , float3 n , float shift)
			{
				return normalize(t + shift * n);
			}
			float StrandSpecular(float3 T, float3 V, float3 L, int exponent)
			{
				float3 H = normalize(L + V);
				float dotTH = dot(T, H);
				float sinTH = sqrt(1.0 - dotTH * dotTH);
				float dirAtten = smoothstep(-1, 0, dotTH);
				return dirAtten * pow(sinTH, exponent);
			}

			float HairSpecular(float3 t, float3 n, float3 l, float3 v, float2 uv)
			{
				float shiftTex = tex2D(_SpecularShift,uv  * _SpecularShift_ST.xy + _SpecularShift_ST.zw).r -0.5;
				float3 t1 = ShiftedTangent(t, n,  shiftTex);
				float specular =  StrandSpecular(t1, v, l, _Shininess);
				return specular;
			}
	//--------------------------------------------------------------------------------------------
            float4 frag(VertexOutput i) : COLOR //像素shader
            {
                float3 nDir = normalize(i.nDirWS); // 获取nDir
                Light mainLight = GetMainLight();
                float3 lDir = normalize(mainLight.direction);
                //由于面部阴影受光照角度影响极易产生难看的阴影，因此可以虑将光照固定成水平方向，再微调面部法线即可得到比较舒适的面部阴影。
             //   _FixLightY=0 ;//即可将光照方向固定至水平。
                float3 fixedlightDirWS = normalize(float3(lDir.x, _FixLightY, lDir.z));
                lDir = _IgnoreLightY ? fixedlightDirWS : lDir;
                // 准备点积结果
                float nDotl = dot(nDir, lDir);
                float lambert =  nDotl * 0.5f + 0.5f; // 截断负值

                //采样BaseMap和LightMap，确定最初的阴影颜色ShadowColor和DarkShadowColor 。
                half4 baseColor = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, i.uv0);
                half4 LightMapColor = SAMPLE_TEXTURE2D(_LightMap, sampler_LightMap, i.uv0);
                half3 viewDirWS = normalize(_WorldSpaceCameraPos.xyz - i.posWS.xyz);
       
                half4 FinalColor = 0;
                // FaceLightMap 
              
          


                //非Ramp阴影
                half3 ShadowColor = baseColor.rgb * _ShadowMultColor.rgb;
                half3 DarkShadowColor = baseColor.rgb * _DarkShadowMultColor.rgb;

                //经分析得知，LightMap的Alpha通道存储了5种信息，Alpha值大致对应Ramp贴图内的材质/颜色如下。
                // 0 hard/emission/specular/silk
                // 77 soft/common
                // 128 metal
                // 179 tights
                // 255 skin

                //Ramp阴影
                //#if ENABLE_RAMP_SHADOW

                float rampValue = lambert * (1.0 / _RampShadowRange - 0.003);

                   float halfLambert = smoothstep(0.0,0.5 , lambert)  * LightMapColor.b;
                half3 finalRampNight = SAMPLE_TEXTURE2D(_RampMap, sampler_RampMap, float2(halfLambert ,LightMapColor.a *0.45)); //采样下半
                half3 finalRampLight = SAMPLE_TEXTURE2D(_RampMap, sampler_RampMap, float2(halfLambert, LightMapColor.a * 0.45 + 0.55)).rgb;//采样上半
                half3 finalRamp = lerp(finalRampLight , finalRampNight , _InNight);
            
           

                //得到最终的Ramp阴影，根据rampValue与BaseColor结合
                rampValue = step(_RampShadowRange, lambert);
                half3 RampShadowColor = lerp(finalRamp * baseColor.rgb, baseColor.rgb, rampValue);

                ShadowColor = RampShadowColor;
                DarkShadowColor = RampShadowColor;

                //#endif
        
                //计算一级阴影颜色ShallowShadowColor的算法
                //如果SFactor = 0,ShallowShadowColor为一级阴影色,否则为BaseColor。
                float SWeight = (LightMapColor.g * baseColor.r + lambert) * 0.5 + 1.125; //总
                float SFactor = floor(SWeight - _ShadowArea); //修正率
                half3 ShallowShadowColor = lerp(ShadowColor, baseColor, SFactor);

                //由于希望可选择是否固定二级阴影颜色DarkShadowColor，因此二级阴影颜色如下计算。
                //如果SFactor = 0,DarkShadowColor为二级阴影色,否则为一级阴影色。
                SFactor = floor(SWeight - _DarkShadowArea);
                float FixDarkShadow = lerp(ShallowShadowColor, ShadowColor, _FixDarkShadow); //固定二级阴影颜色
                DarkShadowColor = lerp(DarkShadowColor, FixDarkShadow, SFactor); //是否固定二级阴影颜色DarkShadowColor

                // 平滑阴影边缘
                half rampS = smoothstep(0, _ShadowSmooth, lambert - _ShadowArea);
                half rampDS = smoothstep(0, _DarkShadowSmooth, lambert - _DarkShadowArea);
                ShallowShadowColor = lerp(ShadowColor, baseColor, rampS);
                DarkShadowColor = lerp(DarkShadowColor, ShadowColor, rampDS);

                //所有准备完成，该计算最终的片元使用哪一级阴影的颜色了。
                //如果SFactor = 0,FinalColor为二级阴影，否则为一级阴影。
                SFactor = floor(LightMapColor.g * baseColor.r +0.9f );
             
                FinalColor.rgb = lerp(DarkShadowColor, ShallowShadowColor, SFactor);
		

        //Custom RimLight 
//===========================================================================================================
  
            float2 L_View = normalize(mul((float3x3)UNITY_MATRIX_V, lDir).xy);
            float2 N_View = normalize(mul((float3x3)UNITY_MATRIX_V, nDir).xy);
            float lDotN = saturate(dot(N_View, L_View) + _DarkSideRimSmooth  );
            i.posNDC.xyz/= i.posNDC.w;
              float depth =i.posNDC.z;
                float linearDepth = LinearEyeDepth(depth  , _ZBufferParams); //离相机越近越小
            float scale = lDotN * _RimPow *0.1  * GetSSRimScale(linearDepth);
   
            float2 ssUV1 = clamp( i.posNDC.xy + N_View * scale, 0, _ScreenParams.xy - 1);
            
            
            float depthDiff = LinearEyeDepth(SAMPLE_TEXTURE2D(_CameraDepthTexture , sampler_CameraDepthTexture, ssUV1 ).r, _ZBufferParams) - linearDepth;
            float intensity = smoothstep(0.24 *_RimSmooth  * linearDepth, 0.25 * linearDepth, depthDiff);
            float4 rimColor =intensity * lerp(_DarkSideRimColor,_RimColor , rampS);
        
          
            
           FinalColor = max(FinalColor , rimColor);
          
        
           // float3 ssColor = intensity * lerp(1, context.baseColor, _RimLightBlend)
          //  * lerp(_RimLightColor.rgb, context.pointLightColor, luminance * _RimLightBlendPoint);
            
          
          
        
      

//===========================================================================================================




                #if ENABLE_FACE_SHADOW_MAP

                //采样脸部遮罩贴图
             //  float Faceshadow =SAMPLE_TEXTURE2D(_FaceShadowMap , sampler_FaceShadowMap , i.uv0).r;
                float var_FaceShadow = LightMapColor.r;
                float revert_FaceShadow =SAMPLE_TEXTURE2D(_LightMap, sampler_LightMap, half2(1-i.uv0.x , i.uv0.y)).r;
			
                //上方向
                float3 Up = unity_ObjectToWorld._12_22_32;

                //角色朝向
                float3 Front = unity_ObjectToWorld._13_23_33;

                //角色右侧朝向
                float3 Right = cross(Up, Front);

                //阴影贴图左右正反切换的开关
                float switchShadow = dot(normalize(Right.xz), normalize(lDir.xz)) * 0.5 + 0.5 < 0.5;

                //阴影贴图左右正反切换
                float FaceShadow = lerp(1-var_FaceShadow.r,1- revert_FaceShadow, 1-switchShadow.r);

                //脸部阴影范围
                float FaceShadowRange = dot(normalize(Front.xz), normalize(lDir.xz));

                //使用阈值来计算阴影
                float lightAttenuation = 1-smoothstep(FaceShadowRange -_ShadowSmooth, FaceShadowRange, FaceShadow -_ShadowArea );

                half4 FaceColor = lerp(_DarkShadowMultColor, _ShadowMultColor, lightAttenuation);
             return FaceColor;



              FinalColor = FaceColor  * FinalColor ;
       

                #endif
		
				//kajia-ya 由于效果不理想 ,暂不使用
				//float Anitspec = HairSpecular(i.bitangentWS ,nDir ,lDir , viewDirWS ,i.uv0 );
			
			


                // Blinn-Phong
				
                half3 halfViewLightWS = normalize(viewDirWS + mainLight.direction.xyz);

                half spec = pow(saturate(dot(i.nDirWS, halfViewLightWS)), _Shininess);
			
			
			
                spec = step(1.0f - LightMapColor.b, spec);
			
                half4 specularColor = _EnableSpecular * _LightSpecColor * _SpecMulti * LightMapColor.r * spec;

                half4 SpecDiffuse;
                SpecDiffuse.rgb = specularColor.rgb + FinalColor.rgb;
                SpecDiffuse.rgb *= _BaseColor.rgb;
                SpecDiffuse.a = specularColor.a * _BloomFactor * 10;

                // Rim Light
                float lambertF = dot(lDir, nDir);
                float lambertD = max(0, -lambertF);
                lambertF = max(0, lambertF);
                float rim = 1 - saturate(dot(viewDirWS, i.nDirWS));

                //正边缘光
                // float rimDot = pow(rim, _RimPow);
                // //rimDot = _EnableLambert * lambertF * rimDot + (1 - _EnableLambert) * rimDot;
                // rimDot = lerp(rimDot, lambertF * rimDot, _EnableLambert);
                // float rimIntensity = smoothstep(0, _RimSmooth, rimDot);
                // half4 Rim = _EnableRim * pow(rimIntensity, 5) * _RimColor * baseColor;
                // Rim.a = _EnableRim * rimIntensity * _BloomFactor;

                // //反边缘光
                // rimDot = pow(rim, _DarkSideRimPow);
                // //rimDot = _EnableLambert * lambertD * rimDot + (1 - _EnableLambert) * rimDot;
                // rimDot = lerp(rimDot, lambertD * rimDot, _EnableLambert);;
                // rimIntensity = smoothstep(0, _DarkSideRimSmooth, rimDot);
                // half4 RimDS = _EnableRimDS * pow(rimIntensity, 5) * _DarkSideRimColor * baseColor;
                // RimDS.a = _EnableRimDS * rimIntensity * _BloomFactor;
      

                // Emission & Bloom
                half4 Emission;
                Emission.rgb = _Emission * DarkShadowColor.rgb * _EmissionColor.rgb - SpecDiffuse.rgb;
                Emission.a = _EmissionBloomFactor * baseColor.a;

                // half4 SpecRimEmission;
                // SpecRimEmission.rgb = pow(DarkShadowColor, 1) * _Emission;
                // SpecRimEmission.a = (SpecDiffuse.a + Rim.a + RimDS.a);

              
       
               
                FinalColor = SpecDiffuse  + Emission.a * Emission ;// + SpecRimEmission.a * SpecRimEmission;

                FinalColor = (_WorldLightInfluence * _MainLightColor * FinalColor + (1 - _WorldLightInfluence) * FinalColor);

                return FinalColor;
            }
            ENDHLSL
        }



        Pass
        {
       Name "DepthOnly"
            Tags { "LightMode"="DepthOnly" }
        
            ZWrite On
            ColorMask 0
        
            HLSLPROGRAM
            // Required to compile gles 2.0 with standard srp library
            #pragma prefer_hlslcc gles
            #pragma exclude_renderers d3d11_9x gles
            //#pragma target 4.5
        
            // Material Keywords
            #pragma shader_feature _ALPHATEST_ON
            #pragma shader_feature _SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A
        
            // GPU Instancing
            #pragma multi_compile_instancing
            #pragma multi_compile _ DOTS_INSTANCING_ON
                    
            #pragma vertex DepthOnlyVertex
            #pragma fragment DepthOnlyFragment


        struct Attributes
        {
            float3 positionOS: POSITION;
            half4 color: COLOR0;
            half3 normalOS: NORMAL;
            half4 tangentOS: TANGENT;
            float2 texcoord: TEXCOORD0;
        };

        struct Varyings
        {
            float4 positionCS: POSITION;
            float4 color: COLOR0;
            float4 uv: TEXCOORD0;
        
      
        

        };
     Varyings DepthOnlyVertex(Attributes input)
        {
            Varyings output = (Varyings)0;
            output.color = input.color;




           VertexPositionInputs vertexInput  = GetVertexPositionInputs(input.positionOS);
            output.positionCS = vertexInput.positionCS;
       


      
 

            output.uv.xy = TRANSFORM_TEX(input.texcoord, _BaseMap);


       
            return output;
        }


        half4 DepthOnlyFragment(Varyings input): SV_TARGET
        {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
            #if ENABLE_ALPHA_CLIPPING
                clip(SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, input.uv).b - _Cutoff);
            #endif


            return 0;
        }
            // Again, using this means we also need _BaseMap, _BaseColor and _Cutoff shader properties
            // Also including them in cbuffer, except _BaseMap as it's a texture.
        
            ENDHLSL

 
          
        }
        Pass
        {
            Name "Outline"
            Tags
            {
            }
            Cull Front

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            struct VertexInput
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                float2 uv0 : TEXCOORD0; // UV信息 Get✔
            };

            struct VertexOutput
            {
                float4 pos : SV_POSITION;
                float2 uv0 : TEXCOORD0; // UV0
            };

inline float4 CalculateOutlineVertexClipPosition(float4 vertex ,float3 normal)
{                                                                                                //y = near plane
    float4 nearUpperRight = mul(unity_CameraInvProjection ,float4(1,1,UNITY_NEAR_CLIP_VALUE, _ProjectionParams.y) );
    float aspect = abs(nearUpperRight.y / nearUpperRight.x);
    VertexPositionInputs VertexInputs = GetVertexPositionInputs(vertex);
    //修正像素比例
    float aspect1 = _ScreenParams.x / _ScreenParams.y;
    float4 o_vertex = VertexInputs.positionCS;
    float3 viewNormal = mul((float3x3)UNITY_MATRIX_IT_MV , normal);
    float3 clipNormal = mul((float3x3) UNITY_MATRIX_P,viewNormal);
    float2 projectedNormal = normalize(clipNormal.xy);
    //由于顶点从裁剪空间转换到屏幕空间需要做齐次除法/w 为了防止值被修改因此先给他乘一个w
    projectedNormal *= min(o_vertex.w  , 2);
    //当屏幕比例不为1：1时，对x进行修正
   projectedNormal.y *= aspect1;
    o_vertex.xy += _outlinewidth * projectedNormal.xy * saturate(1-abs(normalize(viewNormal).z)) * 0.01 ;
    return o_vertex;

}
// inline void CalcuateOutlineColor(inout float3 color)
// {
//     float3 newMapColor = color;
//     float maxChan = max(max(newMapColor.r , newMapColor.g) , newMapColor.b);
//     float3 lerpVals = newMapColor / maxChan;
//     float _powerFactor = 10;
//     lerpVals = pow(lerpVals , _powerFactor);
//     newMapColor.rgb = lerp(_saturationFactor * newMapColor.rgb , newMapColor.rgb , lerpVals);
//     color.rgb = _brightnessFactor * newMapColor.rgb * color.rgb;
  
// }
            VertexOutput vert(VertexInput v)
            {
                VertexOutput o = (VertexOutput)0;
                o.pos = CalculateOutlineVertexClipPosition(v.vertex , v.normal);
                o.uv0 = TRANSFORM_TEX(v.uv0 , _BaseMap);
                return o;
            }

            float4 frag(VertexOutput i) : COLOR
            {
                half4 baseColor = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, i.uv0);
                half4 FinalColor = _outlinecolor * baseColor;
                return FinalColor;
            }
            ENDHLSL
        }
        UsePass "Universal Render Pipeline/Lit/ShadowCaster"
    }
}