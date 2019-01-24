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

//-----------------------------------------------------------------------------
//
// optixMeshViewer: simple interactive mesh viewer 
//
//-----------------------------------------------------------------------------

#include<time.h>
#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#    include <GL/wglew.h>
#    include <GL/freeglut.h>
#  else
#    include <GL/glut.h>
#  endif
#endif

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>
#include "optixPathTracer.h"
#include <sutil.h>
#include "commonStructs.h"
#include <Arcball.h>
#include <OptiXMesh.h>
#include "random.h"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdint.h>

using namespace optix;

const char* const SAMPLE_NAME = "optixMeshViewer";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context        context;
uint32_t       width  = 1024u;
uint32_t       height = 768u;
bool           use_pbo = true;
int            rr_begin_depth = 1;
optix::Aabb    aabb;

// view state
int _use_filter = 1;


// Camera state
float3         camera_up;
float3         camera_lookat;
float3         camera_eye;
Matrix4x4      camera_rotate;
sutil::Arcball arcball;

// Mouse state
int2           mouse_prev_pos;
int            mouse_button;

// Material
std::vector<Material> materials;
enum{
    MESH_FLOOR = 0,
    MESH_GRID = 1
};
time_t t1, t2, t3, t4, t5;

//------------------------------------------------------------------------------
//
// Forward decls 
//
//------------------------------------------------------------------------------

struct UsageReportLogger;

Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext( int usage_report_level, UsageReportLogger* logger );
void loadMesh( const std::string& filename );


void setupCamera();
void setupLights();
void updateCamera();
void glutInitialize( int* argc, char** argv );
void glutRun();

void glutDisplay();
void glutKeyboardPress( unsigned char k, int x, int y );
void glutMousePress( int button, int state, int x, int y );
void glutMouseMotion( int x, int y);
void glutResize( int w, int h );


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer()
{
    return context[ "output_buffer" ]->getBuffer();
}


void destroyContext()
{
    if( context )
    {
        context->destroy();
        context = 0;
    }
}


struct UsageReportLogger
{
  void log( int lvl, const char* tag, const char* msg )
  {
    std::cout << "[" << lvl << "][" << std::left << std::setw( 12 ) << tag << "] " << msg;
  }
};

// Static callback
void usageReportCallback( int lvl, const char* tag, const char* msg, void* cbdata )
{
    // Route messages to a C++ object (the "logger"), as a real app might do.
    // We could have printed them directly in this simple case.

    UsageReportLogger* logger = reinterpret_cast<UsageReportLogger*>( cbdata );
    logger->log( lvl, tag, msg ); 
}

void registerExitHandler()
{
    // register shutdown handler
#ifdef _WIN32
    glutCloseFunc( destroyContext );  // this function is freeglut-only
#else
    atexit( destroyContext );
#endif
}

void createContext( int usage_report_level, UsageReportLogger* logger )
{
    // Set up context
    context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 5 );
    context->setStackSize( 20000 );
    if( usage_report_level > 0 )
    {
        context->setUsageReportCallback( usageReportCallback, usage_report_level, logger );
    }

    context->setPrintEnabled( 1 );
    context->setPrintBufferSize( 4096 );

    context["radiance_ray_type"]->setUint( 0u );
    context["shadow_ray_type"  ]->setUint( 1u );
    context["scene_epsilon"    ]->setFloat( 1.e-3f );
    context["ambient_light_color"]->setFloat( 0.3f, 0.33f, 0.28f );
    context["sqrt_spp"]->setFloat( 3.0 );
    context["image_size"]->setUint( make_uint2(width, height) );
    context["patch_size"]->setUint( make_uint2(20, 20));
    context["cnt"]->setInt(0);
    context["use_filter"]->setInt( _use_filter );
    context["max_spp"]->setFloat( 150.0 );

    // output buffer
    Buffer buffer = sutil::createOutputBuffer( context, RT_FORMAT_UNSIGNED_BYTE4, width, height, use_pbo );
    context["output_buffer"]->set( buffer );

    // pixel buffer
    Buffer pixel_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, width, height );
    context["pixel_buffer"]->set( pixel_buffer );

    // pixel middle buffer
    Buffer pixel_middle_buffer = context->createBuffer( context, RT_FORMAT_FLOAT3, width, height );
    context["pixel_middle_buffer"]->set( pixel_middle_buffer );

    // obj_id buffer
    Buffer obj_id_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
            RT_FORMAT_FLOAT, width, height );
    context["obj_id_buffer"]->set( obj_id_buffer );

    // slope buffer
    Buffer slope = context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
            RT_FORMAT_FLOAT2, width, height );
    context["slope_buffer"]->set( slope );

    // slope temp buffer
    Buffer slope_temp_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
            RT_FORMAT_FLOAT2, width, height );
    context["slope_temp_buffer"]->set( slope_temp_buffer );

    // spp buffer
    Buffer spp = context->createBuffer( RT_BUFFER_INPUT_OUTPUT,
            RT_FORMAT_FLOAT, width, height );
    context["spp_buffer"]->set( spp );

    // spp adaptive sampling buffer
    Buffer spp_adaptive_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, width, height );
    context["spp_adaptive_buffer"]->set(spp_adaptive_buffer);

    // projection buffer
    Buffer projection_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
            RT_FORMAT_FLOAT, width, height );
    context["projection_buffer"]->set( projection_buffer );

    // hit buffer
    Buffer hit_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_INT, width, height);
    context["hit_buffer"]->set( hit_buffer );

    // hit shadow buffer
    Buffer hit_shadow_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_INT, width, height);
    context["hit_shadow_buffer"]->set( hit_shadow_buffer );

    // hit shadow temp buffer
    Buffer hit_shadow_temp_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_INT, width, height);
    context["hit_shadow_temp_buffer"]->set( hit_shadow_temp_buffer );

    //disparity buffer
    Buffer disparity_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
            RT_FORMAT_FLOAT3, width, height);
    context["disparity_buffer"]->set( disparity_buffer );

    // coordinate buffer
    Buffer cooridnate_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
            RT_FORMAT_FLOAT3, width, height);
    context["coordinate_buffer"]->set( cooridnate_buffer );

    // normal buffer
    Buffer normal_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
            RT_FORMAT_FLOAT3, width, height);
    context["normal_buffer"]->set( normal_buffer );

    // beta buffer
    Buffer beta_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
            RT_FORMAT_FLOAT, width, height);
    context["beta_buffer"]->set(beta_buffer);


    // seed buffer
    Buffer seed_buffer = context->createBuffer(RT_BUFFER_INPUT,  RT_FORMAT_UNSIGNED_INT2,
            width, height);
    context["seed_buffer"]->set(seed_buffer);
    uint2* seeds = reinterpret_cast<uint2*>( seed_buffer->map() );

    for( unsigned int i = 0; i < width * height; ++i ){
//        unsigned int ui = i;
//        unsigned int x = lcg(ui);
//        unsigned int y = lcg(ui);
//        seeds[i].x = x;
//        seeds[i].y = y;
        seeds[i] = random2u();
    }
    seed_buffer->unmap();
    // below are adapted from arealight.cpp
    // brdf buffer
    Buffer brdf_buffer = context->createBuffer( context, RT_FORMAT_FLOAT3, width, height );
    context["brdf_buffer"]->set( brdf_buffer );

    Buffer vis_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                                      RT_FORMAT_FLOAT, width, height );
    context["vis_buffer"]->set( vis_buffer );

    // shadow buffer
    Buffer shadow_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, width, height );
    context["shadow_buffer"]->set( shadow_buffer );

    // shadow buffer
    Buffer test_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, width, height );
    context["test_buffer"]->set( test_buffer );



    // Ray generation program
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "pinhole_camera.cu" );

    Program ray_gen_program = context->createProgramFromPTXString( ptx, "pinhole_camera_first" );
    context->setRayGenerationProgram( 0, ray_gen_program );

    // calculate s1 s2 program
    Program maxmins1s2_program = context->createProgramFromPTXString( ptx, "optimals1s2" );
    context->setRayGenerationProgram( 2, maxmins1s2_program );

    // second ray generation program
    Program ray_gen_second = context->createProgramFromPTXString( ptx, "pinhole_camera_second" );
    context->setRayGenerationProgram( 3, ray_gen_second );

    // gaussian filter program
    Program gaussian_program1 = context->createProgramFromPTXString( ptx, "gaussian_filter1" );
    context->setRayGenerationProgram( 4, gaussian_program1 );

//    // gaussian filter program
//    Program gaussian_program2 = context->createProgramFromPTXString( ptx, "gaussian_filter2" );
//    context->setRayGenerationProgram( 5, gaussian_program2 );

    // Program to show
    Program display_program = context->createProgramFromPTXString( ptx, "show" );
    context->setRayGenerationProgram( 1, display_program );

    // Exception program
    Program exception_program = context->createProgramFromPTXString( ptx, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( 0.8f, 0.8f, 0.8f );

    // Miss programc
    Program miss_program = context->createProgramFromPTXString( ptx, "miss" );
    context->setMissProgram( 0, miss_program );
    context["bg_color"]->setFloat( 0.067f, 0.556f, 0.741f );
}


void loadMesh( )
{
    GeometryGroup geometry_group = context->createGeometryGroup();
    std::vector<Matrix4x4> pos;
//    Transformations
//    float m_floor[16] = {
//            3.707f, 0.0f, 0.707f, 0.0f,
//            0.0f, 3.0f, 0.0f, 1.0f,
//            -0.707f, 0.0f, 3.707f, 2.0f,
//            0.0f, 0.0f, 0.0f, 1.0f
//    };
    float m_floor[16] = {
            150.707f, 0.0f, 0.707f, 0.0f,
            0.0f, 150.0f, 0.0f, 5.0f,
            -0.707f, 0.0f, 150.707f, 2.0f,
            0.0f, 0.0f, 0.0f, 1.0f
    };
    Matrix4x4 floor_xform_m = Matrix<4,4>(m_floor);

    float m_grid1[16] = {
            0.75840f, 0.6232783f, -0.156223f, 4.0f,
            -0.465828f, 0.693876f, 0.549127f, 1.7f,
            0.455878f, -0.343688f, 0.821008f, 2.0f,
            0.0f, 0.0f, 0.0f, 1.0f
    };
    Matrix4x4 grid1_xform_m = Matrix<4,4>(m_grid1);

    float m_grid2[16] = {
            0.893628f, 0.203204f, -0.40017f, 1.5f,
            0.105897f, 0.770988f, 0.627984f, 2.3f,
            0.436135f, -0.603561f, 0.667458f, 2.0f,
            0.0f, 0.0f, 0.0f, 1.0f
    };

    Matrix4x4 grid2_xform_m = Matrix<4,4>(m_grid2);

    float m_grid3[16] = {
            0.109836f, 0.392525f, -0.913159f, -1.6f,
            0.652392f, 0.664651f, 0.364174f, 2.6f,
            0.74988f, -0.635738f, -0.183078f, 2.0f,
            0.0f, 0.0f, 0.0f, 1.0f
    };

    Matrix4x4 grid3_xform_m = Matrix<4,4>(m_grid3);

    pos.push_back(floor_xform_m);
    pos.push_back(grid1_xform_m);
    pos.push_back(grid2_xform_m);
    pos.push_back(grid3_xform_m);

    // set color
    std::vector<float3> Color;
    const float3 white = make_float3( 0.68f, 0.67f, 0.66f );
    const float3 green = make_float3( 0.05f, 0.8f, 0.05f );
    const float3 red   = make_float3( 0.8f, 0.05f, 0.05f );
    const float3 blue = make_float3( 0.05f, 0.05f, 0.8f );
    Color.push_back(red);
    Color.push_back(green);
    Color.push_back(blue);
    Color.push_back(white);

    std::string grid1 = std::string( sutil::samplesDir() ) + "/data/grid1.obj";
    std::string grid2 = std::string( sutil::samplesDir() ) + "/data/grid2.obj";
    std::string grid3 = std::string( sutil::samplesDir() ) + "/data/grid3.obj";
    std::string floor = std::string( sutil::samplesDir() ) + "/data/floor.obj";
    std::vector<std::string> mesh;
    mesh.push_back(floor);
    mesh.push_back(grid1);
    mesh.push_back(grid2);
    mesh.push_back(grid3);

    Material shading = context->createMaterial();
    const char *ptx = sutil::getPtxString( SAMPLE_NAME,  "pinhole_camera.cu" );
    Program diffuse_ch = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
//    Program diffuse_tex = context->createProgramFromPTXString( ptx, "closest_hit_radiance_textured");
    Program diffuse_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );


    // dragon
    float m_tree[16] = {
            1.02f, 0.0f, 0.0f, -50.0f,
            0.0f, 1.02f, 0.0f, 65.0f,
            0.0f, 0.0f, 1.02f, -180.0f,
            0.0f, 0.0f, 0.0f, 1.0f
    };

    // tree
//    float m_tree[16] = {
//            0.02f, 0.0f, 0.0f, 100.0f,
//            0.0f, 0.02f, 0.0f, 10.0f,
//            0.0f, 0.0f, 0.02f, 0.0f,
//            0.0f, 0.0f, 0.0f, 1.0f
//    };

    Matrix4x4 grid_tree = Matrix<4,4>(m_tree);
//    std::string tree = std::string( sutil::samplesDir() ) + "/data/tree1/treeslo-poly.obj";
    std::string tree = std::string( sutil::samplesDir() ) + "/data/Dragon/Dragon.obj";
    OptiXMesh daisy;
    Material daisy_mat = context->createMaterial();
    daisy_mat->setClosestHitProgram( 0, diffuse_ch );
    daisy_mat->setAnyHitProgram( 1, diffuse_ah );
    daisy.context = context;
    daisy.material = daisy_mat;
    loadMesh( tree, daisy, grid_tree);
    std::string tree_mtl = std::string( sutil::samplesDir() ) + "/data/tree1/treeslo-poly.mtl";

//    optix::TextureSampler loadTexture( optix::Context context,
//                                       const std::string& filename,
//                                       const optix::float3& default_color );
//    daisy.geom_instance["diffuse_map"]->setTextureSampler( loadTexture( context, tree_mtl, make_float3(0.2,0.2,0.2) ) );
    daisy.geom_instance["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
    daisy.geom_instance["Kd"]->setFloat( 0.97402f, 0.97402f, 0.97402f );
    daisy.geom_instance["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
    daisy.geom_instance["phong_exp"]->setFloat( 100.0f );
    geometry_group->addChild( daisy.geom_instance );




    // floor
    OptiXMesh omesh_floor;
    Material shading_floor = context->createMaterial();
    shading_floor->setClosestHitProgram( 0, diffuse_ch );
    shading_floor->setAnyHitProgram( 1, diffuse_ah );
    omesh_floor.context = context;
    omesh_floor.material = shading_floor;
    std::string filename = mesh[0];
    loadMesh( filename, omesh_floor, pos[0] );
    omesh_floor.geom_instance["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
    omesh_floor.geom_instance["Kd"]->setFloat( 0.97402f, 0.97402f, 0.97402f );
    omesh_floor.geom_instance["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
    omesh_floor.geom_instance["phong_exp"]->setFloat( 100.0f );
    omesh_floor.geom_instance["obj_id"]->setInt(1);
    geometry_group->addChild( omesh_floor.geom_instance );

    // grid1
    OptiXMesh omesh_grid1;
    Material shading_grid1 = context->createMaterial();
    shading_grid1->setClosestHitProgram( 0, diffuse_ch );
    shading_grid1->setAnyHitProgram( 1, diffuse_ah );
    omesh_grid1.material = shading_grid1;

    omesh_grid1.context = context;
    filename = mesh[1];
    loadMesh( filename, omesh_grid1, pos[1] );
    omesh_grid1.geom_instance["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
    omesh_grid1.geom_instance["Kd"]->setFloat( 0.72f, 0.100741f, 0.09848f );
    omesh_grid1.geom_instance["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
    omesh_grid1.geom_instance["phong_exp"]->setFloat( 100.0f );
    omesh_grid1.geom_instance["obj_id"]->setInt(2);

//    geometry_group->addChild( omesh_grid1.geom_instance );

    // grid2
    OptiXMesh omesh_grid2;
    Material shading_grid2 = context->createMaterial();
    shading_grid2->setClosestHitProgram( 0, diffuse_ch );
    shading_grid2->setAnyHitProgram( 1, diffuse_ah );
    omesh_grid2.material = shading_grid2;
    omesh_grid2.context = context;
    omesh_grid2.material = shading_grid2;

    filename = mesh[2];
    loadMesh( filename, omesh_grid2, pos[2] );
    omesh_grid2.geom_instance["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
    omesh_grid2.geom_instance["Kd"]->setFloat( 0.0885402f, 0.77f, 0.08316f );
    omesh_grid2.geom_instance["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
    omesh_grid2.geom_instance["phong_exp"]->setFloat( 100.0f );
    omesh_grid2.geom_instance["obj_id"]->setInt(3);

//    geometry_group->addChild( omesh_grid2.geom_instance );

    // grid3
    OptiXMesh omesh_grid3;
    Material shading_grid3 = context->createMaterial();
    shading_grid3->setClosestHitProgram( 0, diffuse_ch );
    shading_grid3->setAnyHitProgram( 1, diffuse_ah );
    omesh_grid3.material = shading_grid3;
    omesh_grid3.context = context;
    omesh_grid3.material = shading_grid3;

    filename = mesh[3];
    loadMesh( filename, omesh_grid3, pos[3] );
    omesh_grid3.geom_instance["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
    omesh_grid3.geom_instance["Kd"]->setFloat( 0.123915f, 0.192999f, 0.751f );
    omesh_grid3.geom_instance["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
    omesh_grid3.geom_instance["phong_exp"]->setFloat( 100.0f );
    omesh_grid3.geom_instance["obj_id"]->setInt(4);

//    geometry_group->addChild( omesh_grid3.geom_instance );

    aabb.set( omesh_floor.bbox_min, omesh_floor.bbox_max );
    geometry_group->setAcceleration( context->createAcceleration( "Trbvh" ) );
    context[ "top_object"   ]->set( geometry_group ); 
    context[ "top_shadower" ]->set( geometry_group ); 
}



void setupCamera()
{
    const float max_dim = fmaxf(aabb.extent(0), aabb.extent(1)); // max of x, y components

//    camera_eye    = aabb.center() + make_float3( 0.0f, 0.0f, max_dim*1.5f );
//    camera_lookat = aabb.center();
//    camera_up     = make_float3( 0.0f, 1.0f, 0.0f );

    // grid
//    camera_eye    = make_float3( 10.0f, 8.0f, 2.0f );
    // tree dragon
    camera_eye    = make_float3( 0.0f, 100.0f, -300.0f );


    camera_lookat = make_float3( 0.0f, 1.0f ,2.0f );
    camera_up     = make_float3( 0.0f, 1.0f, 0.0f );

    camera_rotate  = Matrix4x4::identity();
}


void setupLights()
{
    const float max_dim = fmaxf(aabb.extent(0), aabb.extent(1)); // max of x, y components


//    Light buffer
    ParallelogramLight light;
//    // the location of light
//    light.corner   = make_float3( -3.0f, 10.0f, -5.0f);
    light.corner = make_float3( -100.0f, 200.0f, -70.0f);
    light.v1       = make_float3( 5.0f, 0.0f, 0.0f);
    light.v2       = make_float3( 0.0f, 5.0f, 5.0f);
    light.normal   = normalize( cross(light.v1, light.v2) );
    light.color = make_float3( 1.0f, 1.0f, 1.0f );
    float Al = length(cross(light.v1, light.v2));

    Buffer light_buffer = context->createBuffer( RT_BUFFER_INPUT );
    light_buffer->setFormat( RT_FORMAT_USER );
    float light_sigma = sqrt(4.0/Al);
    context["light_sigma"]->setFloat( light_sigma );
    context["light_normal"]->setFloat( light.normal );


    light_buffer->setElementSize( sizeof( ParallelogramLight ) );
    light_buffer->setSize( 1u );
    memcpy( light_buffer->map(), &light, sizeof( light ) );
    light_buffer->unmap();
    context["lights"]->setBuffer( light_buffer );
}


void updateCamera()
{
    const float vfov = 35.0f;
    const float aspect_ratio = static_cast<float>(width) /
                               static_cast<float>(height);
    
    float3 camera_u, camera_v, camera_w;
    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
            camera_u, camera_v, camera_w, true );

    const Matrix4x4 frame = Matrix4x4::fromBasis( 
            normalize( camera_u ),
            normalize( camera_v ),
            normalize( -camera_w ),
            camera_lookat);
    const Matrix4x4 frame_inv = frame.inverse();
    // Apply camera rotation twice to match old SDK behavior
    const Matrix4x4 trans     = frame*camera_rotate*camera_rotate*frame_inv; 

    camera_eye    = make_float3( trans*make_float4( camera_eye,    1.0f ) );
    camera_lookat = make_float3( trans*make_float4( camera_lookat, 1.0f ) );
    camera_up     = make_float3( trans*make_float4( camera_up,     0.0f ) );

    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
            camera_u, camera_v, camera_w, true );

    camera_rotate = Matrix4x4::identity();

    context["eye"]->setFloat( camera_eye );
    context["U"  ]->setFloat( camera_u );
    context["V"  ]->setFloat( camera_v );
    context["W"  ]->setFloat( camera_w );
}


void glutInitialize( int* argc, char** argv )
{
    // openGL window initialize
    glutInit( argc, argv );
    glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE );
    glutInitWindowSize( width, height );
    glutInitWindowPosition( 300, 100 );
    glutCreateWindow( "AAF" );
    glutHideWindow();                                                              
}


void glutRun()
{
    // Initialize GL state                                                            
    glMatrixMode(GL_PROJECTION);                                                   
    glLoadIdentity();                                                              
    glOrtho(0, 1, 0, 1, -1, 1 );                                                   

    glMatrixMode(GL_MODELVIEW);                                                    
    glLoadIdentity();                                                              

    glViewport(0, 0, width, height);                                 

    glutShowWindow();                                                              
    glutReshapeWindow( width, height);

    // register glut callbacks
    glutDisplayFunc( glutDisplay );
    glutIdleFunc( glutDisplay );
    glutReshapeFunc( glutResize );
    glutKeyboardFunc( glutKeyboardPress );
    glutMouseFunc( glutMousePress );
    glutMotionFunc( glutMouseMotion );

    registerExitHandler();

    glutMainLoop();
}


//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void glutDisplay()
{
    updateCamera();
    context->launch( 0, width, height );
    context->launch( 2, width, height );
    context->launch( 3, width, height );
    context->launch( 4, width, height );
//    context->launch( 5, width, height );
    context->launch( 1, width, height );

    sutil::displayBufferGL( getOutputBuffer() );

    {
      static unsigned frame_count = 0;
      sutil::displayFps( frame_count++ );
    }

    glutSwapBuffers();
}


void glutKeyboardPress( unsigned char k, int x, int y )
{

    switch( k )
    {
        case( 'f' ):
        {
            _use_filter = 1 - _use_filter;
            context["use_filter"]->setInt( _use_filter );
            break;
        }

        case( 'q' ):
        case( 27 ): // ESC
        {
            Buffer spp_adapative = context["spp_adaptive_buffer"]->getBuffer();
            Buffer hit_shadow = context["hit_shadow_buffer"]->getBuffer();
            float* spp_ada = reinterpret_cast<float*>(spp_adapative->map());
            int* hit_arr = reinterpret_cast<int*>(hit_shadow->map());
            float min_spp = 100000;
            float max_spp = 0;
            for (int i = 0; i<width; i++){
                for(int j=0; j< height; j++){
                    if(hit_arr[i+j*width] == 1){
                        float cur_spp = spp_ada[i+j*width];
                        min_spp = min(min_spp, spp_ada[i+j*width]);
                        max_spp = max(max_spp, spp_ada[i+j*width]);
                    }
                }
            }
            spp_adapative->unmap();
            hit_shadow->unmap();

            Buffer omega = context["beta_buffer"]->getBuffer();
            float* om_arr = reinterpret_cast<float*>(omega->map());
            float beta_min = 10000;
            float beta_max = 0;
            for (int i = 0; i<width; i++){
                for(int j=0; j< height; j++){
                    if(hit_arr[i+j*width] == 1){
                        float cur_spp = spp_ada[i+j*width];
                        beta_min = min(beta_min, om_arr[i+j*width]);
                        beta_max = max(beta_max, om_arr[i+j*width]);
                    }
                }
            }
            std::cout<<beta_min<<"  "<< beta_max<<std::endl;
            std::cout << "first pass: " << (double)(t2 -t1) / CLOCKS_PER_SEC <<std::endl;
            std::cout << "compute beta: " << t3 << std::endl;
            std::cout << "compuate n and 2 pass: " << t4 - t3 << std::endl;
            std::cout << "adaptive filtering: " << t5 - t4 << std::endl;
            destroyContext();
            exit(0);
        }
        case( 's' ):
        {
            const std::string outputImage = std::string(SAMPLE_NAME) + ".ppm";
            std::cerr << "Saving current frame to '" << outputImage << "'\n";
            sutil::displayBufferPPM( outputImage.c_str(), getOutputBuffer() );
            break;
        }
    }
}


void glutMousePress( int button, int state, int x, int y )
{
    if( state == GLUT_DOWN )
    {
        mouse_button = button;
        mouse_prev_pos = make_int2( x, y );
    }
    else
    {
        // nothing
    }
}


void glutMouseMotion( int x, int y)
{
    if( mouse_button == GLUT_RIGHT_BUTTON )
    {
        const float dx = static_cast<float>( x - mouse_prev_pos.x ) /
                         static_cast<float>( width );
        const float dy = static_cast<float>( y - mouse_prev_pos.y ) /
                         static_cast<float>( height );
        const float dmax = fabsf( dx ) > fabs( dy ) ? dx : dy;
        const float scale = fminf( dmax, 0.9f );
        camera_eye = camera_eye + (camera_lookat - camera_eye)*scale;
    }
    else if( mouse_button == GLUT_LEFT_BUTTON )
    {
        const float2 from = { static_cast<float>(mouse_prev_pos.x),
                              static_cast<float>(mouse_prev_pos.y) };
        const float2 to   = { static_cast<float>(x),
                              static_cast<float>(y) };

        const float2 a = { from.x / width, from.y / height };
        const float2 b = { to.x   / width, to.y   / height };

        camera_rotate = arcball.rotate( b, a );
    }

    mouse_prev_pos = make_int2( x, y );
}


void glutResize( int w, int h )
{
    if ( w == (int)width && h == (int)height ) return;

    width  = w;
    height = h;

    sutil::resizeBuffer( getOutputBuffer(), width, height );
    
    glViewport(0, 0, width, height);                                               

    glutPostRedisplay();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "\nUsage: " << argv0 << " [options]\n";
    std::cerr <<
        "App Options:\n"
        "  -h | --help               Print this usage message and exit.\n"
        "  -f | --file               Save single frame to file and exit.\n"
        "  -n | --nopbo              Disable GL interop for display buffer.\n"
        "  -m | --mesh <mesh_file>   Specify path to mesh to be loaded.\n"
        "  -r | --report <LEVEL>     Enable usage reporting and report level [1-3].\n"
        "App Keystrokes:\n"
        "  q  Quit\n" 
        "  s  Save image to '" << SAMPLE_NAME << ".ppm'\n"
        << std::endl;

    exit(1);
}

int main( int argc, char** argv )
 {
    std::string out_file;


    int usage_report_level = 0;
//    for( int i=1; i<argc; ++i )
//    {
//        const std::string arg( argv[i] );
//
//        if( arg == "-h" || arg == "--help" )
//        {
//            printUsageAndExit( argv[0] );
//        }
//        else if( arg == "-f" || arg == "--file"  )
//        {
//            if( i == argc-1 )
//            {
//                std::cerr << "Option '" << arg << "' requires additional argument.\n";
//                printUsageAndExit( argv[0] );
//            }
//            out_file = argv[++i];
//        }
//        else if( arg == "-n" || arg == "--nopbo"  )
//        {
//            use_pbo = false;
//        }
//        else if( arg == "-m" || arg == "--mesh" )
//        {
//            if( i == argc-1 )
//            {
//                std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
//                printUsageAndExit( argv[0] );
//            }
//            mesh_file = argv[++i];
//        }
//        else if( arg == "-r" || arg == "--report" )
//        {
//            if( i == argc-1 )
//            {
//                std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
//                printUsageAndExit( argv[0] );
//            }
//            usage_report_level = atoi( argv[++i] );
//        }
//        else
//        {
//            std::cerr << "Unknown option '" << arg << "'\n";
//            printUsageAndExit( argv[0] );
//        }
//    }

    try
    {
        glutInitialize( &argc, argv );

#ifndef __APPLE__
        glewInit();
#endif

        UsageReportLogger logger;
        createContext( usage_report_level, &logger );
        loadMesh();

        setupCamera();
        setupLights();

        context->validate();

        if ( out_file.empty() )
        {
            t1 = time(NULL);

            glutRun();
            t2 = time(NULL);

        }
        else
        {
            std::cout << "1" <<std::endl;
            updateCamera();
            context->launch( 0, width, height );
            context->launch( 2, width, height );
            t3 = time(NULL);
            context->launch( 3, width, height );
            t4 = time(NULL);
            context->launch( 4, width, height );
            t5 = time(NULL);
//            context->launch( 5, width, height );
            context->launch( 1, width, height );

            sutil::displayBufferPPM( out_file.c_str(), getOutputBuffer() );
            destroyContext();

        }

        std:: cout << "1" <<std::endl;

        return 0;
    }
    SUTIL_CATCH( context->get() )
}

