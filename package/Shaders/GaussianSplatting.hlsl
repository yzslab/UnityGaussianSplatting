// SPDX-License-Identifier: MIT
#ifndef GAUSSIAN_SPLATTING_HLSL
#define GAUSSIAN_SPLATTING_HLSL

float InvSquareCentered01(float x)
{
    x -= 0.5;
    x *= 0.5;
    x = sqrt(abs(x)) * sign(x);
    return x + 0.5;
}

float3 QuatRotateVector(float3 v, float4 r)
{
    float3 t = 2 * cross(r.xyz, v);
    return v + r.w * t + cross(r.xyz, t);
}

float4 QuatMul(float4 a, float4 b)
{
    return float4(a.wwww * b + (a.xyzx * b.wwwx + a.yzxy * b.zxyy) * float4(1,1,1,-1) - a.zxyz * b.yzxz);
}

float4 QuatInverse(float4 q)
{
    return rcp(dot(q, q)) * q * float4(-1,-1,-1,1);
}

float3x3 CalcMatrixFromRotationScale(float4 rot, float3 scale)
{
    float3x3 ms = float3x3(
        scale.x, 0, 0,
        0, scale.y, 0,
        0, 0, scale.z
    );
    float x = rot.x;
    float y = rot.y;
    float z = rot.z;
    float w = rot.w;
    float3x3 mr = float3x3(
        1-2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y),
          2*(x*y + w*z), 1-2*(x*x + z*z),   2*(y*z - w*x),
          2*(x*z - w*y),   2*(y*z + w*x), 1-2*(x*x + y*y)
    );
    return mul(mr, ms);
}

void CalcCovariance3D(float3x3 rotMat, out float3 sigma0, out float3 sigma1)
{
    float3x3 sig = mul(rotMat, transpose(rotMat));
    sigma0 = float3(sig._m00, sig._m01, sig._m02);
    sigma1 = float3(sig._m11, sig._m12, sig._m22);
}

// from "EWA Splatting" (Zwicker et al 2002) eq. 31
float3 CalcCovariance2D(float3 worldPos, float3 cov3d0, float3 cov3d1, float4x4 matrixV, float4x4 matrixP, float4 screenParams)
{
    float4x4 viewMatrix = matrixV;
    float3 viewPos = mul(viewMatrix, float4(worldPos, 1)).xyz;

    // this is needed in order for splats that are visible in view but clipped "quite a lot" to work
    float aspect = matrixP._m00 / matrixP._m11;
    float tanFovX = rcp(matrixP._m00);
    float tanFovY = rcp(matrixP._m11 * aspect);
    float limX = 1.3 * tanFovX;
    float limY = 1.3 * tanFovY;
    viewPos.x = clamp(viewPos.x / viewPos.z, -limX, limX) * viewPos.z;
    viewPos.y = clamp(viewPos.y / viewPos.z, -limY, limY) * viewPos.z;

    float focal = screenParams.x * matrixP._m00 / 2;

    float3x3 J = float3x3(
        focal / viewPos.z, 0, -(focal * viewPos.x) / (viewPos.z * viewPos.z),
        0, focal / viewPos.z, -(focal * viewPos.y) / (viewPos.z * viewPos.z),
        0, 0, 0
    );
    float3x3 W = (float3x3)viewMatrix;
    float3x3 T = mul(J, W);
    float3x3 V = float3x3(
        cov3d0.x, cov3d0.y, cov3d0.z,
        cov3d0.y, cov3d1.x, cov3d1.y,
        cov3d0.z, cov3d1.y, cov3d1.z
    );
    float3x3 cov = mul(T, mul(V, transpose(T)));

    // Low pass filter to make each splat at least 1px size.
    cov._m00 += 0.3;
    cov._m11 += 0.3;
    return float3(cov._m00, cov._m01, cov._m11);
}

float3 CalcConic(float3 cov2d)
{
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    return float3(cov2d.z, -cov2d.y, cov2d.x) * rcp(det);
}

float2 CalcScreenSpaceDelta(float2 svPositionXY, float2 centerXY, float4 projectionParams)
{
    float2 d = svPositionXY - centerXY;
    d.y *= projectionParams.x;
    return d;
}

float CalcPowerFromConic(float3 conic, float2 d)
{
    return -0.5 * (conic.x * d.x*d.x + conic.z * d.y*d.y) + conic.y * d.x*d.y;
}

// Morton interleaving 16x16 group i.e. by 4 bits of coordinates, based on this thread:
// https://twitter.com/rygorous/status/986715358852608000
// which is simplified version of https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
uint EncodeMorton2D_16x16(uint2 c)
{
    uint t = ((c.y & 0xF) << 8) | (c.x & 0xF); // ----EFGH----ABCD
    t = (t ^ (t << 2)) & 0x3333;               // --EF--GH--AB--CD
    t = (t ^ (t << 1)) & 0x5555;               // -E-F-G-H-A-B-C-D
    return (t | (t >> 7)) & 0xFF;              // --------EAFBGCHD
}
uint2 DecodeMorton2D_16x16(uint t)      // --------EAFBGCHD
{
    t = (t & 0xFF) | ((t & 0xFE) << 7); // -EAFBGCHEAFBGCHD
    t &= 0x5555;                        // -E-F-G-H-A-B-C-D
    t = (t ^ (t >> 1)) & 0x3333;        // --EF--GH--AB--CD
    t = (t ^ (t >> 2)) & 0x0f0f;        // ----EFGH----ABCD
    return uint2(t & 0xF, t >> 8);      // --------EFGHABCD
}


static const float SH_C1 = 0.4886025;

struct SplatSHData
{
    half3 col, sh1, sh2, sh3;
};

half3 ShadeSH(SplatSHData splat, half3 dir, int shOrder, bool onlySH)
{
    dir *= -1;

    half x = dir.x, y = dir.y, z = dir.z;

    // ambient band
    half3 res = splat.col; // col = sh0 * SH_C0 + 0.5 is already precomputed
    if (onlySH)
        res = 0.5;
    // 1st degree
    if (shOrder >= 1)
    {
        res += SH_C1 * (-splat.sh1 * y + splat.sh2 * z - splat.sh3 * x);
    }
    return max(res, 0);
}

static const uint kTexWidth = 2048;

uint3 SplatIndexToPixelIndex(uint idx)
{
    uint3 res;

    uint2 xy = DecodeMorton2D_16x16(idx);
    uint width = kTexWidth / 16;
    idx >>= 8;
    res.x = (idx % width) * 16 + xy.x;
    res.y = (idx / width) * 16 + xy.y;
    res.z = 0;
    return res;
}

struct SplatChunkInfo
{
    uint colR, colG, colB, colA;
    float2 posX, posY, posZ;
    uint sclX, sclY, sclZ;
    uint shR, shG, shB;
};

StructuredBuffer<SplatChunkInfo> _SplatChunks;
uint _SplatChunkCount;

static const uint kChunkSize = 256;

struct SplatData
{
    float3 pos;
    float4 rot;
    float3 scale;
    half opacity;
    SplatSHData sh;
};

// Decode quaternion from a "smallest 3" e.g. 10.10.10.2 format
float4 DecodeRotation(float4 pq)
{
    uint idx = (uint)round(pq.w * 3.0); // note: need to round or index might come out wrong in some formats (e.g. fp16.fp16.fp16.fp16)
    float4 q;
    q.xyz = pq.xyz * sqrt(2.0) - (1.0 / sqrt(2.0));
    q.w = sqrt(1.0 - saturate(dot(q.xyz, q.xyz)));
    if (idx == 0) q = q.wxyz;
    if (idx == 1) q = q.xwyz;
    if (idx == 2) q = q.xywz;
    return q;
}
float4 PackSmallest3Rotation(float4 q)
{
    // find biggest component
    float4 absQ = abs(q);
    int index = 0;
    float maxV = absQ.x;
    if (absQ.y > maxV)
    {
        index = 1;
        maxV = absQ.y;
    }
    if (absQ.z > maxV)
    {
        index = 2;
        maxV = absQ.z;
    }
    if (absQ.w > maxV)
    {
        index = 3;
        maxV = absQ.w;
    }

    if (index == 0) q = q.yzwx;
    if (index == 1) q = q.xzwy;
    if (index == 2) q = q.xywz;

    float3 three = q.xyz * (q.w >= 0 ? 1 : -1); // -1/sqrt2..+1/sqrt2 range
    three = (three * sqrt(2.0)) * 0.5 + 0.5; // 0..1 range
    return float4(three, index / 3.0);
}

half3 DecodePacked_6_5_5(uint enc)
{
    return half3(
        (enc & 63) / 63.0,
        ((enc >> 6) & 31) / 31.0,
        ((enc >> 11) & 31) / 31.0);
}

half3 DecodePacked_5_6_5(uint enc)
{
    return half3(
        (enc & 31) / 31.0,
        ((enc >> 5) & 63) / 63.0,
        ((enc >> 11) & 31) / 31.0);
}

half3 DecodePacked_11_10_11(uint enc)
{
    return half3(
        (enc & 2047) / 2047.0,
        ((enc >> 11) & 1023) / 1023.0,
        ((enc >> 21) & 2047) / 2047.0);
}

float3 DecodePacked_16_16_16(uint2 enc)
{
    return float3(
        (enc.x & 65535) / 65535.0,
        ((enc.x >> 16) & 65535) / 65535.0,
        (enc.y & 65535) / 65535.0);
}

float4 DecodePacked_10_10_10_2(uint enc)
{
    return float4(
        (enc & 1023) / 1023.0,
        ((enc >> 10) & 1023) / 1023.0,
        ((enc >> 20) & 1023) / 1023.0,
        ((enc >> 30) & 3) / 3.0);
}
uint EncodeQuatToNorm10(float4 v) // 32 bits: 10.10.10.2
{
    return (uint) (v.x * 1023.5f) | ((uint) (v.y * 1023.5f) << 10) | ((uint) (v.z * 1023.5f) << 20) | ((uint) (v.w * 3.5f) << 30);
}


#ifdef SHADER_STAGE_COMPUTE
#define SplatBufferDataType RWByteAddressBuffer
#else
#define SplatBufferDataType ByteAddressBuffer
#endif

SplatBufferDataType _SplatPos;
SplatBufferDataType _SplatOther;
SplatBufferDataType _SplatSH;
Texture2D _SplatColor;
uint _SplatFormat;

// Match GaussianSplatAsset.VectorFormat
#define VECTOR_FMT_32F 0
#define VECTOR_FMT_16 1
#define VECTOR_FMT_11 2
#define VECTOR_FMT_6 3

uint LoadUShort(SplatBufferDataType dataBuffer, uint addrU)
{
    uint addrA = addrU & ~0x3;
    uint val = dataBuffer.Load(addrA);
    if (addrU != addrA)
        val >>= 16;
    return val & 0xFFFF;
}

uint LoadUInt(SplatBufferDataType dataBuffer, uint addrU)
{
    uint addrA = addrU & ~0x3;
    uint val = dataBuffer.Load(addrA);
    if (addrU != addrA)
    {
        uint val1 = dataBuffer.Load(addrA + 4);
        val = (val >> 16) | ((val1 & 0xFFFF) << 16);
    }
    return val;
}

float3 LoadAndDecodeVector(SplatBufferDataType dataBuffer, uint addrU, uint fmt)
{
    uint addrA = addrU & ~0x3;

    uint val0 = dataBuffer.Load(addrA);

    float3 res = 0;
    if (fmt == VECTOR_FMT_32F)
    {
        uint val1 = dataBuffer.Load(addrA + 4);
        uint val2 = dataBuffer.Load(addrA + 8);
        if (addrU != addrA)
        {
            uint val3 = dataBuffer.Load(addrA + 12);
            val0 = (val0 >> 16) | ((val1 & 0xFFFF) << 16);
            val1 = (val1 >> 16) | ((val2 & 0xFFFF) << 16);
            val2 = (val2 >> 16) | ((val3 & 0xFFFF) << 16);
        }
        res = float3(asfloat(val0), asfloat(val1), asfloat(val2));
    }
    else if (fmt == VECTOR_FMT_16)
    {
        uint val1 = dataBuffer.Load(addrA + 4);
        if (addrU != addrA)
        {
            val0 = (val0 >> 16) | ((val1 & 0xFFFF) << 16);
            val1 >>= 16;
        }
        res = DecodePacked_16_16_16(uint2(val0, val1));
    }
    else if (fmt == VECTOR_FMT_11)
    {
        uint val1 = dataBuffer.Load(addrA + 4);
        if (addrU != addrA)
        {
            val0 = (val0 >> 16) | ((val1 & 0xFFFF) << 16);
        }
        res = DecodePacked_11_10_11(val0);
    }
    else if (fmt == VECTOR_FMT_6)
    {
        if (addrU != addrA)
            val0 >>= 16;
        res = DecodePacked_6_5_5(val0);
    }
    return res;
}

float3 LoadSplatPosValue(uint index)
{
    uint fmt = _SplatFormat & 0xFF;
    uint stride = 0;
    if (fmt == VECTOR_FMT_32F)
        stride = 12;
    else if (fmt == VECTOR_FMT_16)
        stride = 6;
    else if (fmt == VECTOR_FMT_11)
        stride = 4;
    else if (fmt == VECTOR_FMT_6)
        stride = 2;
    return LoadAndDecodeVector(_SplatPos, index * stride, fmt);
}

float3 LoadSplatPos(uint idx)
{
    float3 pos = LoadSplatPosValue(idx);
    uint chunkIdx = idx / kChunkSize;
    if (chunkIdx < _SplatChunkCount)
    {
        SplatChunkInfo chunk = _SplatChunks[chunkIdx];
        float3 posMin = float3(chunk.posX.x, chunk.posY.x, chunk.posZ.x);
        float3 posMax = float3(chunk.posX.y, chunk.posY.y, chunk.posZ.y);
        pos = lerp(posMin, posMax, pos);
    }
    return pos;
}

half4 LoadSplatColTex(uint3 coord)
{
    return _SplatColor.Load(coord);
}

SplatData LoadSplatData(uint idx)
{
    SplatData s = (SplatData)0;

    // figure out raw data offsets / locations
    uint3 coord = SplatIndexToPixelIndex(idx);

    uint scaleFmt = (_SplatFormat >> 8) & 0xFF;
    uint shFormat = (_SplatFormat >> 16) & 0xFF;

    uint otherStride = 4; // rotation is 10.10.10.2
    if (scaleFmt == VECTOR_FMT_32F)
        otherStride += 12;
    else if (scaleFmt == VECTOR_FMT_16)
        otherStride += 6;
    else if (scaleFmt == VECTOR_FMT_11)
        otherStride += 4;
    else if (scaleFmt == VECTOR_FMT_6)
        otherStride += 2;
    if (shFormat > VECTOR_FMT_6)
        otherStride += 2;
    uint otherAddr = idx * otherStride;

    // SH degree 1: three RGB coefficients. Match GaussianSplatAsset.SHTableItem*.
    uint shStride = 0;
    if (shFormat == VECTOR_FMT_32F)
        shStride = 48;
    else if (shFormat == VECTOR_FMT_16 || shFormat > VECTOR_FMT_6)
        shStride = 24;
    else if (shFormat == VECTOR_FMT_11)
        shStride = 12;
    else if (shFormat == VECTOR_FMT_6)
        shStride = 8;

    s.pos       = LoadSplatPosValue(idx);
    s.rot       = DecodeRotation(DecodePacked_10_10_10_2(LoadUInt(_SplatOther, otherAddr)));
    s.scale     = LoadAndDecodeVector(_SplatOther, otherAddr + 4, scaleFmt);
    half4 col   = LoadSplatColTex(coord);

    uint shIndex = idx;
    if (shFormat > VECTOR_FMT_6)
        shIndex = LoadUShort(_SplatOther, otherAddr + otherStride - 2);
    uint shOffset = shIndex * shStride;
    if (shFormat == VECTOR_FMT_32F)
    {
        s.sh.sh1 = asfloat(_SplatSH.Load3(shOffset));
        s.sh.sh2 = asfloat(_SplatSH.Load3(shOffset + 12));
        s.sh.sh3 = asfloat(_SplatSH.Load3(shOffset + 24));
    }
    else if (shFormat == VECTOR_FMT_16 || shFormat > VECTOR_FMT_6)
    {
        uint4 raw = _SplatSH.Load4(shOffset);
        uint last = _SplatSH.Load(shOffset + 16);
        s.sh.sh1 = half3(f16tof32(raw.x), f16tof32(raw.x >> 16), f16tof32(raw.y));
        s.sh.sh2 = half3(f16tof32(raw.y >> 16), f16tof32(raw.z), f16tof32(raw.z >> 16));
        s.sh.sh3 = half3(f16tof32(raw.w), f16tof32(raw.w >> 16), f16tof32(last));
    }
    else if (shFormat == VECTOR_FMT_11)
    {
        uint3 raw = _SplatSH.Load3(shOffset);
        s.sh.sh1 = DecodePacked_11_10_11(raw.x);
        s.sh.sh2 = DecodePacked_11_10_11(raw.y);
        s.sh.sh3 = DecodePacked_11_10_11(raw.z);
    }
    else if (shFormat == VECTOR_FMT_6)
    {
        uint2 raw = _SplatSH.Load2(shOffset);
        s.sh.sh1 = DecodePacked_5_6_5(raw.x);
        s.sh.sh2 = DecodePacked_5_6_5(raw.x >> 16);
        s.sh.sh3 = DecodePacked_5_6_5(raw.y);
    }

    // if raw data is chunk-relative, convert to final values by interpolating between chunk min/max
    uint chunkIdx = idx / kChunkSize;
    if (chunkIdx < _SplatChunkCount)
    {
        SplatChunkInfo chunk = _SplatChunks[chunkIdx];
        float3 posMin = float3(chunk.posX.x, chunk.posY.x, chunk.posZ.x);
        float3 posMax = float3(chunk.posX.y, chunk.posY.y, chunk.posZ.y);
        half3 sclMin = half3(f16tof32(chunk.sclX    ), f16tof32(chunk.sclY    ), f16tof32(chunk.sclZ    ));
        half3 sclMax = half3(f16tof32(chunk.sclX>>16), f16tof32(chunk.sclY>>16), f16tof32(chunk.sclZ>>16));
        half4 colMin = half4(f16tof32(chunk.colR    ), f16tof32(chunk.colG    ), f16tof32(chunk.colB    ), f16tof32(chunk.colA    ));
        half4 colMax = half4(f16tof32(chunk.colR>>16), f16tof32(chunk.colG>>16), f16tof32(chunk.colB>>16), f16tof32(chunk.colA>>16));
        half3 shMin = half3(f16tof32(chunk.shR    ), f16tof32(chunk.shG    ), f16tof32(chunk.shB    ));
        half3 shMax = half3(f16tof32(chunk.shR>>16), f16tof32(chunk.shG>>16), f16tof32(chunk.shB>>16));
        s.pos = lerp(posMin, posMax, s.pos);
        s.scale     = lerp(sclMin, sclMax, s.scale);
        s.scale *= s.scale;
        s.scale *= s.scale;
        s.scale *= s.scale;
        col   = lerp(colMin, colMax, col);
        col.a = InvSquareCentered01(col.a);

        if (shFormat == VECTOR_FMT_11 || shFormat == VECTOR_FMT_6)
        {
            s.sh.sh1    = lerp(shMin, shMax, s.sh.sh1 );
            s.sh.sh2    = lerp(shMin, shMax, s.sh.sh2 );
            s.sh.sh3    = lerp(shMin, shMax, s.sh.sh3 );
        }
    }
    s.opacity   = col.a;
    s.sh.col    = col.rgb;

    return s;
}

struct SplatViewData
{
    float4 pos;
    float2 axis1, axis2;
    uint2 color; // 4xFP16
};

// If we are rendering into backbuffer directly (e.g. HDR off, no postprocessing),
// the color target texture is a render target (so projection is upside down),
// but the depth buffer we get is not upside down. We want to flip
// our rendering upside down manually for this case.
//
// There does not seem to be a good way to detect this situation in Unity; work around it
// by setting _CameraTargetTexture global texture to BuiltinRenderTextureType.CameraTarget
// from the command buffer. When CameraTarget will be null (i.e. backbuffer), the _TexeSize
// property of the texture will get set to (1,1,1,1).
//
// One could hope someday Unity will fix all this upside-down thingy...
float4 _CameraTargetTexture_TexelSize;
void FlipProjectionIfBackbuffer(inout float4 vpos)
{
    if (_CameraTargetTexture_TexelSize.z == 1.0)
        vpos.y = -vpos.y;
}

#endif // GAUSSIAN_SPLATTING_HLSL
