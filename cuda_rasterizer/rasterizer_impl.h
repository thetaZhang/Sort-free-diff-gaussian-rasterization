/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once

#include <iostream>
#include <sys/types.h>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{ 
    //for Sort-free
		//size_t scan_size;
		float* depths;
    //for Sort-free
		//char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
    /*for Sort-free
		uint32_t* point_offsets;
		uint32_t* tiles_touched;
    */
    float* depthweights; // for Sort-free

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
    /*for Sort-free
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;
    */
    uint32_t* point_list_idx; // gaussian_point_index
    uint32_t* point_list_tile; // tile_index

		static BinningState fromChunk(char*& chunk, size_t P);
	};

    //for Sort-free     
    struct TileState
    {
        size_t tile_scan_size;
        uint32_t* tile_scanning_space;
        uint32_t* tile_point_touched; // number of gaussian that touch each tile
        uint32_t* tile_offsets; // offset in the point list for each tile
        uint32_t* tile_counters; // counter buffer 

        static TileState fromChunk(char*& chunk, size_t );
    };

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};