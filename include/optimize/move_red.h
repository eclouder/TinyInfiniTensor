//#pragma once
//
//#include "optimize/optim_base.h"
//#include "operators/transpose.h"
//#include "utils/operator_utils.h"
//#include "core/ref.h"
//#include <memory>
//#include "core/graph.h"
//
//
//namespace infini{
//    class move_red: optim_base{
//    public:
//        static void remove_redundancy_T_op(GraphObj *g,Operator _op){
//            auto op = std::dynamic_pointer_cast<TransposeObj>(_op);
//            // 检查前置算子是否为T
//            auto prev_ops = op->getPredecessors();
//            // 前置算子的类型为T 且只有一个后置算子的一个唯一前置算子
//            if ((prev_ops.size() == 1) &&
//                (prev_ops[0]->getOpType() == OpType::Transpose) &&
//                (prev_ops[0]->getSuccessors().size() == 1)){
//                auto prev_op = as<TransposeObj>(prev_ops[0]);
//                Tensor prev_input = prev_op->getInputs(0);
//                auto prev_permute = prev_op->getPermute();
//                auto cur_permute = op->getPermute();
//                prev_input->removeTarget(prev_op);
//                prev_op->removeSuccessors(op);
//                op->reshape_permute(reorderVector(prev_permute,cur_permute));
//                for(auto next_op:op->getSuccessors()){
//                    next_op->replaceInput(op->getOutputs(0),prev_input);
//                    prev_input->addTarget(next_op);
//                    next_op->removePredecessors(op);
//                }
////                auto new_op = op->getSuccessors()[0];
//                g->removeTensor(op->getOutput());
//                g->removeOperator(op);
//            }
//        }
//    };
//}