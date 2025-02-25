#pragma once
#ifndef OPERATOR_UTIL_H
#define OPERATOR_UTIL_H

#include "core/operator.h"
#include "core/tensor.h"

#include <numeric>

namespace infini {

// Launch a broadcast shape based on the shape of input A and B
Shape infer_broadcast(const Shape &A, const Shape &B);
// Launch the real axis based on rank and current axis
int get_real_axis(const int &axis, const int &rank);
// Locate the index with size from Shape
Shape locate_index(size_t inputN, const Shape &shape);
// Delocate the ShapeIndex from Shape with broadcast
size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride);
// Convert KernelAttrs to a string representation
std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs);
Shape reorderVector(const Shape &vec,const Shape &permute);
} // namespace infini

#endif
