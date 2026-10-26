using System.Collections.Generic;
using System.Globalization;
using System.IO;
using GaussianSplatting.Runtime;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace GaussianSplatting.Editor.Utils
{
    // Internal degree-1 layout; also matches ExportSplatData in SplatUtilities.compute.
    public struct InputSplatData
    {
        public Vector3 pos;
        public Vector3 nor;
        public Vector3 dc0;
        public Vector3 sh1, sh2, sh3;
        public float opacity;
        public Vector3 scale;
        public Quaternion rot;
    }

    [BurstCompile]
    public class GaussianFileReader
    {
        // Returns splat count
        public static int ReadFileHeader(string filePath)
        {
            int vertexCount = 0;
            if (File.Exists(filePath))
            {
                if (isPLY(filePath))
                    PLYFileReader.ReadFileHeader(filePath, out vertexCount, out _, out _);
                else if (isSPZ(filePath))
                    SPZFileReader.ReadFileHeader(filePath, out vertexCount);
            }
            return vertexCount;
        }

        public static unsafe void ReadFile(string filePath, out NativeArray<InputSplatData> splats)
        {
            if (isPLY(filePath))
            {
                PLYFileReader.ReadFile(filePath, out var splatCount, out var splatStride, out var attributes, out var verticesRawData);
                using (verticesRawData)
                using (var attributeMap = new NativeArray<int>(CreatePLYAttributeMap(attributes, splatStride), Allocator.TempJob))
                {
                    splats = new NativeArray<InputSplatData>(splatCount, Allocator.Persistent);
                    try
                    {
                        var job = new ReadPLYDataJob
                        {
                            input = verticesRawData.Reinterpret<float>(1),
                            attributeMap = attributeMap,
                            inputStride = splatStride / sizeof(float),
                            output = splats
                        };
                        job.Schedule(splatCount, 4096).Complete();
                        LinearizeData(splats);
                    }
                    catch
                    {
                        splats.Dispose();
                        throw;
                    }
                }
                return;
            }
            if (isSPZ(filePath))
            {
                SPZFileReader.ReadFile(filePath, out splats);
                return;
            }
            throw new IOException($"File {filePath} is not a supported format");
        }

        static bool isPLY(string filePath) => filePath.EndsWith(".ply", true, CultureInfo.InvariantCulture);
        static bool isSPZ(string filePath) => filePath.EndsWith(".spz", true, CultureInfo.InvariantCulture);

        static int[] CreatePLYAttributeMap(List<string> attributes, int stride)
        {
            if (stride != attributes.Count * sizeof(float))
                throw new IOException("Gaussian PLY vertex properties must be 32-bit floats");

            int shPropertyCount = 0;
            foreach (var attribute in attributes)
            {
                if (attribute.StartsWith("f_rest_"))
                    ++shPropertyCount;
            }
            if (shPropertyCount != 0 && shPropertyCount != 9 && shPropertyCount != 24 && shPropertyCount != 45)
                throw new IOException($"Unsupported PLY SH property count {shPropertyCount}; expected 0, 9, 24 or 45");

            // Map by name so normals are optional and higher-degree inputs can be truncated.
            string[] properties =
            {
                "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
                null, null, null, null, null, null, null, null, null,
                "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"
            };
            int shCount = shPropertyCount / 3;
            for (int coefficient = 0; coefficient < GaussianSplatAsset.kSHCoefficientCount; ++coefficient)
            {
                for (int channel = 0; channel < 3; ++channel)
                {
                    if (shCount > 0)
                        properties[9 + coefficient * 3 + channel] = $"f_rest_{channel * shCount + coefficient}";
                }
            }

            var map = new int[properties.Length];
            for (int i = 0; i < properties.Length; ++i)
            {
                map[i] = properties[i] == null ? -1 : attributes.IndexOf(properties[i]);
                bool optional = (i >= 3 && i < 6) || properties[i] == null;
                if (map[i] < 0 && !optional)
                {
                    throw new IOException($"Gaussian PLY is missing property {properties[i]}");
                }
            }
            return map;
        }

        [BurstCompile]
        struct ReadPLYDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float> input;
            [ReadOnly] public NativeArray<int> attributeMap;
            public int inputStride;
            public NativeArray<InputSplatData> output;

            public unsafe void Execute(int index)
            {
                InputSplatData splat = default;
                float* values = (float*)&splat;
                for (int i = 0; i < attributeMap.Length; ++i)
                {
                    int source = attributeMap[i];
                    values[i] = source < 0 ? 0 : input[index * inputStride + source];
                }
                output[index] = splat;
            }
        }

        [BurstCompile]
        struct LinearizeDataJob : IJobParallelFor
        {
            public NativeArray<InputSplatData> splatData;
            public void Execute(int index)
            {
                var splat = splatData[index];

                // rot
                var q = splat.rot;
                var qq = GaussianUtils.NormalizeSwizzleRotation(new float4(q.x, q.y, q.z, q.w));
                qq = GaussianUtils.PackSmallest3Rotation(qq);
                splat.rot = new Quaternion(qq.x, qq.y, qq.z, qq.w);

                // scale
                splat.scale = GaussianUtils.LinearScale(splat.scale);

                // color
                splat.dc0 = GaussianUtils.SH0ToColor(splat.dc0);
                splat.opacity = GaussianUtils.Sigmoid(splat.opacity);

                splatData[index] = splat;
            }
        }

        static void LinearizeData(NativeArray<InputSplatData> splatData)
        {
            LinearizeDataJob job = new LinearizeDataJob();
            job.splatData = splatData;
            job.Schedule(splatData.Length, 4096).Complete();
        }
    }
}
