#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        this->used += size;
        // 遍历寻找是否有空闲内存
        for(auto&block:free_blocks){
            auto free_mem = block.second;
            if (free_mem >= size){
                if(free_mem > size){
                    free_blocks[block.first + size] = block.second - size;
                }
                free_blocks.erase(block.first);
                return block.first;
            }
        }
        this->peak += size;
        return this->peak - size;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        this->used -= size;
        // 判断是否是顶部
        if (addr + size == this->peak) {
            this->peak -= size;
            return;
        }
        for (auto &block:free_blocks){
            // 判断是否有前置空闲内存
            if (block.first + block.second == addr){
                block.second += size;
                return;
            }
            // 判断是否有后置空闲内存
            if (block.first == addr + size){
                free_blocks[addr] = size + block.second;
                free_blocks.erase(block.first);
                return;
            }
        }
        free_blocks[addr] = size;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
