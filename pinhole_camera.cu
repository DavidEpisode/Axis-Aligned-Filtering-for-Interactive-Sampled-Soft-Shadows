/* 
 * Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <optix_world.h>
#include <optixPathTracer.h>
#include "helpers.h"
#include "random.h"
#include "commonStructs.h"

#define M_PI 3.14159265358979323846


using namespace optix;

struct PerRayData_radiance
{
    float3 result;
    float3 brdf;
    bool hit;
    int obj_id;
    float t_hit;

    int sqrt_samples;
    float s1;
    float s2;
    bool hit_shadow;
    float3 normal;

    float  importance;
    int    depth;
    float3 coordinate;

    float3 shadow_attenuation;
    float weight;

};

struct PerRayData_shadow
{
    bool hit;
    float d_min;
    float d_max;
    float3 attenuation;
};

rtDeclareVariable(int, cnt , , );

// camera parameter
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );

rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );

// Ray parameter
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, );

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );

rtDeclareVariable(float,   sqrt_spp, , );
rtDeclareVariable(float3, bg_color, , );
rtDeclareVariable(float,  max_spp, , );

// buffer set
rtBuffer<uchar4, 2>              output_buffer;
rtBuffer<float2, 2>              slope_buffer;
//
rtBuffer<float3, 2>              brdf_buffer;
//
rtBuffer<float2, 2>              slope_temp_buffer;
rtBuffer<int,    2>              obj_id_buffer;
rtBuffer<float3, 2>              pixel_buffer;
rtBuffer<float,  2>              projection_buffer;
rtBuffer<float,  2>              spp_buffer;
rtBuffer<float,  2>              spp_adaptive_buffer;
rtBuffer<uint2,  2>              seed_buffer;
rtBuffer<int,    2>              hit_buffer;
rtBuffer<int,    2>              hit_shadow_buffer;
rtBuffer<int,    2>              hit_shadow_temp_buffer;
rtBuffer<float3, 2>              disparity_buffer;
rtBuffer<float3, 2>              coordinate_buffer;
rtBuffer<float3,  2>             pixel_middle_buffer;
rtBuffer<float,  2>              beta_buffer;
rtBuffer<ParallelogramLight>     lights;

rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(int,          use_filter, ,);

// light parameter
rtDeclareVariable(float, light_sigma, , );
rtDeclareVariable(float3, light_normal, , );

rtDeclareVariable(unsigned int,  radiance_ray_type, , );
rtDeclareVariable(unsigned int,  shadow_ray_type , , );

// image plane parameter
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(uint2, image_size, , );
rtDeclareVariable(uint2, patch_size, , );

//material parameter
rtDeclareVariable(float3,   Ka, , );
rtDeclareVariable(float3,   Ks, , );
rtDeclareVariable(float,    phong_exp, , );
rtDeclareVariable(float3,   Kd, , );
rtDeclareVariable(float3,   Kr, , );
rtDeclareVariable(float3,   ambient_light_color, , );
rtDeclareVariable(int,               max_depth, , );
rtDeclareVariable(int,      obj_id, , );

__device__ __inline__ float computeOmegaxf( float s2 ){
    float omegaxf = min( 1/(light_sigma * s2), 1/(projection_buffer[launch_index] * (1+s2)));
//    float omegaxf = min(2/(light_sigma * s2), 1/(projection_buffer[launch_index]*(1+s2)));

    return omegaxf;
}

__device__ __inline__ float computeAdaptiveSamplingRate(float s1, float s2, float omegaxf){
    float n1 = omegaxf * projection_buffer[launch_index] + 1 / (1+ s2);
    float n2 = 1/ light_sigma + fminf(s1 * omegaxf, s1 / ((1+s1)*projection_buffer[launch_index]));
    return n1 * n1 * n2 * n2;
//    float spp_t_1 = (1/(1+s2)+projection_buffer[launch_index]*omegaxf);
//    float spp_t_2 = (1+light_sigma * min(s1*omegaxf,1/projection_buffer[launch_index]
//                                                * s1/(1+s1)));
//    float spp = 4*spp_t_1*spp_t_1*spp_t_2*spp_t_2;
//    return spp;

}



__device__ __inline__ float3 heatmap( float val ){
    float fraction;
    if (val < 0.0f)
        fraction = -1.0f;
    else if (val > 1.0f)
        fraction = 1.0f;
    else
        fraction = 2.0f * val - 1.0f;

    if (fraction < -0.5f) //B
        return make_float3(0.0f, 2*(fraction+1.0f), 1.0f);
    else if (fraction < 0.0f) //G
        return make_float3(0.0f, 1.0f, 1.0f - 2.0f * (fraction + 0.5f));
    else if (fraction < 0.5f) //G
        return make_float3(2.0f * fraction, 1.0f, 0.0f);
    else //R
        return make_float3(1.0f, 1.0f - 2.0f*(fraction - 0.5f), 0.0f);
}

__device__ __inline__ float gaussian( float distance, float omega ){
    float beta = 1 / (3 * omega);
    float a = - distance / (2 * beta * beta);
    float index = exp( a );
    return index;
}

RT_PROGRAM void pinhole_camera_first()
{
  // set ray
  float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);

  // intialize buffer
  slope_buffer[launch_index] = make_float2( 0.0f, 1000.0f);
  slope_temp_buffer[launch_index] = make_float2( 0.0f, 1000.0f);
  float spp = sqrt_spp * sqrt_spp;
  spp_buffer[launch_index] = spp;
  hit_buffer[launch_index] = 1;
  hit_shadow_buffer[launch_index] = 50;
  hit_shadow_temp_buffer[launch_index] = 50;
  disparity_buffer[launch_index] = make_float3( 0, 0, 0 );
  coordinate_buffer[launch_index] = make_float3( 0.0, 0.0, 0.0 );
  beta_buffer[launch_index] = 50;

  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
  PerRayData_radiance prd;

  prd.obj_id = -1;
  prd.sqrt_samples = sqrt_spp;
  prd.s1 = 0;
  prd.s2 = 10000;
  prd.hit_shadow = false;
  prd.hit = false;
  prd.shadow_attenuation = make_float3( 0.0f );
  prd.weight = 0.0f;

  rtTrace(top_object, ray, prd);

  obj_id_buffer[launch_index] = prd.obj_id;
  coordinate_buffer[launch_index] = prd.coordinate;
  pixel_buffer[launch_index] = prd.result;
  slope_buffer[launch_index] = make_float2( prd.s1, prd.s2 );
  float pro_d = prd.t_hit * tan(30.0 * M_PI / 180.0) / (image_size.y / 2.0);
  projection_buffer[launch_index] = pro_d;

  brdf_buffer[launch_index] = prd.brdf;


    if (!prd.hit){
      spp_adaptive_buffer[launch_index] = 0;
      hit_buffer[launch_index] = 0;
      slope_buffer[launch_index] = make_float2( 0.7f, 0.3f);
      slope_temp_buffer[launch_index] = make_float2( 0.7f, 0.3f);;
      beta_buffer[launch_index] = 100;
      return;
  }
  hit_shadow_buffer[launch_index] = 0;
  spp_adaptive_buffer[launch_index] = spp_buffer[launch_index];
  disparity_buffer[launch_index] = make_float3( 0.4, 0.4, 0.9);
  beta_buffer[launch_index] = 200;
//  slope_buffer


  if (prd.hit_shadow){
      float Omegaxf = computeOmegaxf(prd.s2);
      slope_buffer[launch_index] = make_float2( prd.s1, prd.s2 );
      hit_shadow_buffer[launch_index] = 1;

  }
}

__device__ __inline__ float2 maxminslope( float2 cur_slope, int obj_id, int& hit_shadow, uint2 launch_index){
    float2 target_slope = cur_slope;
    int patch_x = patch_size.x;
    int patch_y = patch_size.y;
    for (int i = -patch_x; i<=patch_x;i++){
        for (int j = -patch_y; j<=patch_y; j++ ){
            if (launch_index.x + i>0 && launch_index.x +i < image_size.x && launch_index.y+j > 0 && launch_index.y+j < image_size.y){
                uint2 target_index = make_uint2( launch_index.x + i, launch_index.y + j );
                if (obj_id_buffer[target_index] == obj_id){
                    float2 temp_slope = slope_buffer[target_index];
                    hit_shadow |= hit_shadow_buffer[target_index];
                    target_slope.x = fmaxf( target_slope.x, temp_slope.x );
                    target_slope.y = fminf( target_slope.y, temp_slope.y );
                }
            }
        }
    }
    return target_slope;
}

RT_PROGRAM void optimals1s2(){
    if (obj_id_buffer[launch_index] == 1){

        float2 cur_slope = slope_buffer[launch_index];
        int obj_id = obj_id_buffer[launch_index];
        int hit_shadow = hit_shadow_buffer[launch_index];

        float2 target_slope = maxminslope(cur_slope, obj_id, hit_shadow, launch_index);
        if (hit_shadow_buffer[launch_index] != 1){
            hit_shadow_temp_buffer[launch_index] = hit_shadow;
            slope_temp_buffer[launch_index] = target_slope;
        }
        else{
            hit_shadow_temp_buffer[launch_index] = hit_shadow_buffer[launch_index];
            slope_temp_buffer[launch_index] = cur_slope;
        }
    }
}

RT_PROGRAM void pinhole_camera_second(){
    if (hit_buffer[launch_index] == 0 )
        return;

    if (obj_id_buffer[launch_index] != 1)
        return;

    if (hit_shadow_temp_buffer[launch_index] != 1)
        return;

    float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f;
    float3 ray_origin = eye;
    float3 ray_direction = normalize(d.x*U + d.y*V + W);

    float2 cur_slope = slope_temp_buffer[launch_index];
    float Omegaxf = computeOmegaxf(cur_slope.y);
    float n = computeAdaptiveSamplingRate( cur_slope.x, cur_slope.y, Omegaxf );
    beta_buffer[launch_index] = n;
//    if (n>70)
//        n = 71.0;
//    if (n<60)
//        n = 59.0;

    float cur_n = spp_adaptive_buffer[launch_index];
    spp_adaptive_buffer[launch_index] = min(n, max_spp);

    if (cur_n < n){
        disparity_buffer[launch_index] = make_float3(0.8, 0.2, 0.2);
        float spp_sqrt = ceilf(sqrt(n));
        int nn = int(spp_sqrt);
        PerRayData_radiance prd1;
        prd1.s1 = cur_slope.x;
        prd1.s2 = cur_slope.y;
        prd1.hit_shadow = false;
        prd1.sqrt_samples = nn;

        optix::Ray ray1(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
        rtTrace(top_object, ray1, prd1);
        if (!prd1.hit)
            return;
        pixel_buffer[launch_index] = prd1.result;
    }
}

__device__ __inline__ float gaussian_calculation(uint2 target_index, float3 coordinate, float Omegaxf){
    float3 target_coordinate = coordinate_buffer[target_index];
    float eudist = length( coordinate - target_coordinate );
    float eudist_veritical = dot( coordinate - target_coordinate, light_normal );
    float distance = eudist * eudist - eudist_veritical * eudist_veritical;
    float e = gaussian( distance, Omegaxf );
    return e;


}

RT_PROGRAM void gaussian_filter1(){
    if ( hit_shadow_temp_buffer[launch_index] != 1)
        return;
    float3 filter_pixel = make_float3( 0.0f );
    float3 coordinate = coordinate_buffer[launch_index];
    float Omegaxf = computeOmegaxf(slope_temp_buffer[launch_index].y);
    float exp = 0.0f;
    int x = patch_size.x + 5;
    int y = patch_size.y + 5;
    for (int i = -x; i < x; ++i){
        for (int j = -y; j < y; ++j){
            if (launch_index.x + i > 0 && launch_index.x +i < image_size.x && launch_index.y + j > 0 && launch_index.y + j < image_size.y){
                uint2 target_index = make_uint2( launch_index.x + i, launch_index.y + j);
                if (hit_shadow_temp_buffer[target_index] == 1){
                    float e = gaussian_calculation(target_index, coordinate, Omegaxf);
                    filter_pixel +=  make_float3( e * pixel_buffer[target_index].x, e * pixel_buffer[target_index].y, e * pixel_buffer[target_index].z );
                    exp += e;
                }
            }
        }
    }
    if (exp > 0.000001){
        float3 p = pixel_buffer[launch_index];
        pixel_middle_buffer[launch_index] = make_float3( filter_pixel.x / exp, filter_pixel.y / exp, filter_pixel.z / exp );
    }
    else
        pixel_middle_buffer[launch_index] = pixel_buffer[launch_index];

}


RT_PROGRAM void closest_hit_radiance()
{
    float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
    float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

    prd_radiance.t_hit = t_hit;
    prd_radiance.hit = true;
    float3 hit_point = ray.origin + t_hit * ray.direction;
    prd_radiance.coordinate = hit_point;
    prd_radiance.normal = ffnormal;
    unsigned int sample_number = prd_radiance.sqrt_samples * prd_radiance.sqrt_samples;
    uint2 seed = seed_buffer[launch_index];


    // ambient contribution
    float3 final_result= make_float3( 0.0f );

    // compute direct lighting
    unsigned int num_lights = lights.size();
    for(int ii = 0; ii < num_lights; ++ii) {
        float3 results = make_float3(0.0f);
        float3 color = Ka * ambient_light_color;

        ParallelogramLight light = lights[ii];
        float3 light_center = light.corner + 0.5 * light.v1 + 0.5 * light.v2;
        float Ldist = optix::length(light_center - hit_point);
        float3 L = optix::normalize(light_center - hit_point);
        float nDl = optix::dot(ffnormal, L);
        float3 H = optix::normalize(L - ray.direction);
        float nDh = fmaxf(dot(ffnormal, H), 0.0f);
        if (nDl > 0)
            color += Kd * nDl;
        if (nDh > 0)
            color += Ks * pow(nDh, phong_exp);
        prd_radiance.brdf = color;
        float to_light_dist = length(light_center - hit_point);


        // cast shadow ray
        float3 light_attenuation = make_float3(static_cast<float>( nDl > 0.0f ));


        if (nDl > 0.0f) {
            for (int i = 0; i < prd_radiance.sqrt_samples; ++i) {
                for (int j = 0; j < prd_radiance.sqrt_samples; ++j) {
                    float3 result = make_float3(0.0f);
                    PerRayData_shadow shadow_prd;
                    shadow_prd.attenuation = make_float3(1.0f);
                    shadow_prd.d_min = 100000;
                    shadow_prd.d_max = 0;
                    shadow_prd.hit = false;
                    float2 sample = make_float2(rnd(seed.x), rnd(seed.y)); //rnd(seed.x), rnd(seed.y)
                    sample.x = (sample.x + ((float) i)) / prd_radiance.sqrt_samples;
                    sample.y = (sample.y + ((float) j)) / prd_radiance.sqrt_samples;

                    float3 target = sample.x * light.v1 + sample.y * light.v2 + light.corner;
                    float dist = optix::length(light_center - target);
                    float3 sample_direction = optix::normalize(target - hit_point);
                    float factor = dist * dist / (2 * light_sigma * light_sigma);
                    float w = exp(-factor);
                    prd_radiance.weight += w;


                    if(dot(sample_direction, ffnormal) > 0.0f){
                        float sampleLdist = optix::length(target - hit_point);
                        optix::Ray shadow_ray = optix::make_Ray(hit_point, sample_direction, shadow_ray_type, scene_epsilon,
                                                                sampleLdist);

                        rtTrace(top_shadower, shadow_ray, shadow_prd);
                        light_attenuation = shadow_prd.attenuation;

                        if (shadow_prd.hit){
                            prd_radiance.hit_shadow = true;
                            float d_min = sampleLdist - shadow_prd.d_max;
                            float d_max = sampleLdist - shadow_prd.d_min;

                            float s1 = sampleLdist / d_min - 1.0f;
                            float s2 = sampleLdist / d_max - 1.0f;

                            prd_radiance.s1 = fmaxf( s1, prd_radiance.s1 );
                            prd_radiance.s2 = fminf( s2, prd_radiance.s2 );
                        }

                        if (fmaxf(light_attenuation) > 0.0f) {
                            float3 Lc = light.color * light_attenuation;
                            float snDl = optix::dot(ffnormal, sample_direction);

                            result += Kd * snDl * Lc;

                            float3
                            H = optix::normalize(sample_direction - ray.direction);
                            float snDh = optix::dot(ffnormal, H);
                            if (snDh > 0) {
                                float power = pow(snDh, phong_exp);
                                result += Ks * power * Lc;
                            }
                            results += result;
                            prd_radiance.shadow_attenuation += w * result;

                        }
                    }
                }
            }
//            if (prd_radiance.weight > 0.4)
//            results = prd_radiance.shadow_attenuation / prd_radiance.weight;
//            else
                results /= sample_number;
//            else
//            results = prd_radiance.shadow_attenuation / prd_radiance.weight;
//            float3 a = prd_radiance.shadow_attenuation / prd_radiance.weight;
//            if (prd_radiance.hit_shadow)
//                rtPrintf("(%f, %f, %f), (%f, %f, %f)\n", results.x, results.y, results.z, a.x, a.y, a.z);
//                rtPrintf("%f, %f, %f\n", prd_radiance.shadow_attenuation.x,prd_radiance.shadow_attenuation.y,prd_radiance.shadow_attenuation.z);
//                rtPrintf("%f,(%f,%f,%f)\n", prd_radiance.weight, a.x,a.y,a.z);
//            rtPrintf("%f\n", prd_radiance.weight);

            final_result += results;
        }

    }


    // pass the color back up the tree
    final_result += Ka * ambient_light_color;
    prd_radiance.result = final_result;
    prd_radiance.obj_id = obj_id;
    seed_buffer[launch_index] = seed;
//
}

RT_PROGRAM void any_hit_shadow()
{
    // this material is opaque, so it fully attenuates all shadow rays
    prd_shadow.attenuation = optix::make_float3(0.0f);
    prd_shadow.hit = true;

    prd_shadow.d_min = fminf( prd_shadow.d_min, t_hit );
    prd_shadow.d_max = fmaxf( prd_shadow.d_max, t_hit );

    rtTerminateRay();
}

//rtTextureSampler<float4, 2> Kd_map;
//rtDeclareVariable(float3, texcoord, attribute texcoord, );
//
//RT_PROGRAM void closest_hit_radiance_textured()
//{
//    float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
//    float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
//
//    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
//
//    const float3 Kd_val = make_float3( tex2D( Kd_map, texcoord.x, texcoord.y ) );
//    float3 color = Ka * ambient_light_color;
//    float3 hit_point = ray.origin + t_hit * ray.direction;
//    prd_radiance.t_hit = t_hit;
//    prd_radiance.world_loc = hit_point;
//    prd_radiance.hit = true;
//    prd_radiance.n = ffnormal;
//
//    uint2 seed = shadow_rng_seeds[launch_index];
//    AreaLight light = lights[0];
//    float3 lx = light.v2 - light.v1;
//    float3 ly = light.v3 - light.v1;
//    float3 lo = light.v1;
//    float3 colorAvg = make_float3(0,0,0);
//
//    float3 light_center = (0.5 * lx + 0.5 * ly) +lo;
//    float3 to_light = light_center - hit_point;
//    float dist_to_light = sqrt(to_light.x*to_light.x + to_light.y*to_light.y
//                               + to_light.z*to_light.z);
//
//    prd_radiance.dist_to_light = dist_to_light;
//    float3 L = normalize(to_light);
//    float nDl = fmaxf(dot( ffnormal, L ),0.0f);
//    float3 H = normalize(L - ray.direction);
//    float nDh = fmaxf(dot( ffnormal, H ),0.0f);
//    // white light
//    color += Kd_val * nDl;
//    if (nDh > 0)
//        color += Ks * pow(nDh, phong_exp);
//    prd_radiance.brdf = color;
//
//
//
//}

RT_PROGRAM void show()
{
//    // output buffer display
//    if (hit_shadow_buffer[launch_index] == 1){
//        output_buffer[launch_index] = make_color( pixel_middle_buffer[launch_index] );
//    }
//    else
//        output_buffer[launch_index] = make_color( pixel_buffer[launch_index]);
//    output_buffer[launch_index] = make_color( vis_buffer[launch_index] * pixel_buffer[launch_index] );
//
    if (hit_shadow_temp_buffer[launch_index] == 1 )
        output_buffer[launch_index] = make_color( pixel_middle_buffer[launch_index] );
    else
        output_buffer[launch_index] = make_color( pixel_buffer[launch_index] );

//    output_buffer[launch_index] = make_color( shadow_buffer[launch_index].x * brdf_buffer[launch_index] );

    // adaptive samping rate
//    output_buffer[launch_index] =  make_color( heatmap(spp_adaptive_buffer[launch_index] / 60.0) );
//    if (spp_adaptive_buffer[launch_index] > 70)
//        output_buffer[launch_index] = make_color( make_float3( 0.8, 0.2, 0.2));
//    else
//        output_buffer[launch_index] = make_color( make_float3(0.2, 0.2, 0.8));
    // spp buffer
//    output_buffer[launch_index] =  make_color( heatmap(spp_buffer[launch_index] / 60.0) );
    //obj id buffer
//    output_buffer[launch_index] = make_color( heatmap((float)(obj_id_buffer[launch_index])/5.0 ) );
    // s1
//    output_buffer[launch_index] = make_color( heatmap( (float)(slope_temp_buffer[launch_index].x) ) );
//    // s2
//    output_buffer[launch_index] = make_color( heatmap( (float)(slope_temp_buffer[launch_index].y) ) );
//    if (hit_shadow_buffer[launch_index] == 1)
//        rtPrintf("%f, %f\n", slope_buffer[launch_index].x, slope_buffer[launch_index].y);
    // hit buffer
//    output_buffer[launch_index] = make_color( make_float3( hit_buffer[launch_index]) );
    // hit shadow buffer
//    output_buffer[launch_index] = make_color( heatmap( hit_shadow_buffer[launch_index] / 5.0 ) );

    // disparity buffer
//    if (hit_shadow_temp_buffer[launch_index] == 1)
//    {
//        float3 pixel1 = pixel_middle_buffer[launch_index];
//        float3 pixel2 = pixel_buffer[launch_index];
//        if (pixel1.x != pixel2.x || pixel1.y != pixel2.y || pixel1.z != pixel2.z)
//            output_buffer[launch_index] = make_color( make_float3(0.8, 0.2, 0.2) );
//        else
//            output_buffer[launch_index] = make_color( bg_color );
//    }
//    else
//        output_buffer[launch_index] = make_color( make_float3(0.1, 0.1, 0.1) );
//    output_buffer[launch_index] = make_color( disparity_buffer[launch_index]);

//     beta buffer
//    if (hit_shadow_temp_buffer[launch_index] == 1){
////        if (spp_adaptive_buffer[launch_index] > 70)
////            output_buffer[launch_index] = make_color( make_float3(0.8, 0.1, 0.1) );
////        else if (spp_adaptive_buffer[launch_index] > 60)
////            output_buffer[launch_index] = make_color( make_float3(0.8, 0.4, 0.1) );
////        else if (spp_adaptive_buffer[launch_index] > 50)
////            output_buffer[launch_index] = make_color( make_float3(0.8, 0.8, 0.4) );
////        else if (spp_adaptive_buffer[launch_index] > 40)
////            output_buffer[launch_index] = make_color( make_float3(0.1, 0.8, 0.1) );
////        else if (spp_adaptive_buffer[launch_index] > 30)
////            output_buffer[launch_index] = make_color( make_float3(0.1, 0.8, 0.4) );
////        else
////            output_buffer[launch_index] = make_color( make_float3(0.1, 0.8, 0.8) );
//        output_buffer[launch_index] = make_color( heatmap(beta_buffer[launch_index]/60.0));
//    }
//    else if (hit_buffer[launch_index])
//        output_buffer[launch_index] = make_color(make_float3(0.0f));
//    else
//        output_buffer[launch_index] = make_color(bg_color);



}

RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  output_buffer[launch_index] = make_color(bad_color);
}


RT_PROGRAM void miss()
{
  prd_radiance.result = bg_color;
  prd_radiance.hit_shadow = false;
  prd_radiance.brdf = bg_color;

}