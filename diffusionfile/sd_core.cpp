// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// sd_core.cpp — wrapper for stable-diffusion.cpp/stable-diffusion.cpp
//
// The original source #define's STB_IMAGE_IMPLEMENTATION and includes
// stb_image.h, which would compile the implementation inline.  Since
// llamafile already provides stb_image.o (via third_party/stb/stb.a),
// we first include the header WITHOUT the implementation define so the
// include guard fires when the original source does its include.
//
// This lets us share a single stb instance across the whole binary.

#include "stb_image.h"           // declarations only → sets include guard

// Now include the real source — its STB_IMAGE_IMPLEMENTATION/#include
// "stb_image.h" is swallowed by the include guard above.
#include "../llamafile/stable-diffusion.cpp/stable-diffusion.cpp"
