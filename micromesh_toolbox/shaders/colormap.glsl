
// Adapted from https://www.shadertoy.com/view/WlfXRN
// License CC0 (public domain)
//   https://creativecommons.org/share-your-work/public-domain/cc0/

precision highp float;

#define COLORMAP_TEMPERATURE 0
#define COLORMAP_VIRIDIS 1
#define COLORMAP_PLASMA 2
#define COLORMAP_MAGMA 3
#define COLORMAP_INFERNO 4
#define COLORMAP_TURBO 5
#define COLORMAP_BATLOW 6

vec3 viridis(in float t)
{

  const vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
  const vec3 c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
  const vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
  const vec3 c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
  const vec3 c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105);
  const vec3 c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234);
  const vec3 c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832);

  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

vec3 plasma(float t)
{

  const vec3 c0 = vec3(0.05873234392399702, 0.02333670892565664, 0.5433401826748754);
  const vec3 c1 = vec3(2.176514634195958, 0.2383834171260182, 0.7539604599784036);
  const vec3 c2 = vec3(-2.689460476458034, -7.455851135738909, 3.110799939717086);
  const vec3 c3 = vec3(6.130348345893603, 42.3461881477227, -28.51885465332158);
  const vec3 c4 = vec3(-11.10743619062271, -82.66631109428045, 60.13984767418263);
  const vec3 c5 = vec3(10.02306557647065, 71.41361770095349, -54.07218655560067);
  const vec3 c6 = vec3(-3.658713842777788, -22.93153465461149, 18.19190778539828);

  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

vec3 magma(float t)
{

  const vec3 c0 = vec3(-0.002136485053939582, -0.000749655052795221, -0.005386127855323933);
  const vec3 c1 = vec3(0.2516605407371642, 0.6775232436837668, 2.494026599312351);
  const vec3 c2 = vec3(8.353717279216625, -3.577719514958484, 0.3144679030132573);
  const vec3 c3 = vec3(-27.66873308576866, 14.26473078096533, -13.64921318813922);
  const vec3 c4 = vec3(52.17613981234068, -27.94360607168351, 12.94416944238394);
  const vec3 c5 = vec3(-50.76852536473588, 29.04658282127291, 4.23415299384598);
  const vec3 c6 = vec3(18.65570506591883, -11.48977351997711, -5.601961508734096);

  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

vec3 inferno(float t)
{

  const vec3 c0 = vec3(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
  const vec3 c1 = vec3(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
  const vec3 c2 = vec3(11.60249308247187, -3.972853965665698, -15.9423941062914);
  const vec3 c3 = vec3(-41.70399613139459, 17.43639888205313, 44.35414519872813);
  const vec3 c4 = vec3(77.162935699427, -33.40235894210092, -81.80730925738993);
  const vec3 c5 = vec3(-71.31942824499214, 32.62606426397723, 73.20951985803202);
  const vec3 c6 = vec3(25.13112622477341, -12.24266895238567, -23.07032500287172);

  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

// Copyright 2019 Google LLC.
// SPDX-License-Identifier: Apache-2.0
// https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7
vec3 turbo(in float x)
{
  const vec4 kRedVec4   = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  const vec4 kGreenVec4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  const vec4 kBlueVec4  = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  const vec2 kRedVec2   = vec2(-152.94239396, 59.28637943);
  const vec2 kGreenVec2 = vec2(4.27729857, 2.82956604);
  const vec2 kBlueVec2  = vec2(-89.90310912, 27.34824973);

  x       = clamp(x, 0.0, 1.0);
  vec4 v4 = vec4(1.0, x, x * x, x * x * x);
  vec2 v2 = v4.zw * v4.z;
  return vec3(dot(v4, kRedVec4) + dot(v2, kRedVec2), dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
              dot(v4, kBlueVec4) + dot(v2, kBlueVec2));
}

// Approximates the batlow color ramp from the scientific color ramps package.
// Input will be clamped to [0, 1]; output is sRGB.
vec3 batlow(float t)
{
  t             = clamp(t, 0.0f, 1.0f);
  const vec3 c5 = vec3(10.741, -0.934, -16.125);
  const vec3 c4 = vec3(-28.888, 2.021, 34.529);
  const vec3 c3 = vec3(24.263, -0.335, -20.561);
  const vec3 c2 = vec3(-6.069, -1.511, 2.47);
  const vec3 c1 = vec3(0.928, 1.455, 0.327);
  const vec3 c0 = vec3(0.007, 0.103, 0.341);

  vec3 result = ((((c5 * t + c4) * t + c3) * t + c2) * t + c1) * t + c0;

  return min(result, vec3(1.0f));
}

// utility for temperature
float fade(float low, float high, float value)
{
  float mid   = (low + high) * 0.5;
  float range = (high - low) * 0.5;
  float x     = 1.0 - clamp(abs(mid - value) / range, 0.0, 1.0);
  return smoothstep(0.0, 1.0, x);
}

// Return a cold-hot color based on intensity [0-1]
vec3 temperature(float intensity)
{
  const vec3 blue   = vec3(0.0, 0.0, 1.0);
  const vec3 cyan   = vec3(0.0, 1.0, 1.0);
  const vec3 green  = vec3(0.0, 1.0, 0.0);
  const vec3 yellow = vec3(1.0, 1.0, 0.0);
  const vec3 red    = vec3(1.0, 0.0, 0.0);

  vec3 color = (fade(-0.25, 0.25, intensity) * blue    //
                + fade(0.0, 0.5, intensity) * cyan     //
                + fade(0.25, 0.75, intensity) * green  //
                + fade(0.5, 1.0, intensity) * yellow   //
                + smoothstep(0.75, 1.0, intensity) * red);
  return color;
}


vec3 colorMap(int map, float t)
{
  switch(map)
  {
    case COLORMAP_VIRIDIS:
      return viridis(t);
    case COLORMAP_PLASMA:
      return plasma(t);
    case COLORMAP_MAGMA:
      return magma(t);
    case COLORMAP_INFERNO:
      return inferno(t);
    case COLORMAP_TURBO:
      return turbo(t);
    case COLORMAP_BATLOW:
      return batlow(t);
    case COLORMAP_TEMPERATURE:
      return temperature(t);
  }
}
