#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include "operators/transpose.h"
#include "utils/operator_utils.h"
#include "operators/matmul.h"
namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        for (auto op_i = 0; op_i < ops.size(); ){
            auto _op = ops[op_i];
            if (_op->getOpType() == OpType::Transpose){
//                move_red::remove_redundancy_T_op(this,_op);
                auto op = std::dynamic_pointer_cast<TransposeObj>(_op);
                auto prev_ops = op->getPredecessors();
                // 前置算子的类型为T 且只有一个后置算子的一个唯一前置算子
                if ((prev_ops.size() == 1) &&
                    (prev_ops[0]->getOpType() == OpType::Transpose) &&
                    (prev_ops[0]->getSuccessors().size() == 1)){
                    auto prev_op = as<TransposeObj>(prev_ops[0]);
                    Tensor prev_input = prev_op->getInputs(0);
                    auto prev_permute = prev_op->getPermute();
                    auto cur_permute = op->getPermute();
                    prev_input->removeTarget(prev_op);
                    prev_op->removeSuccessors(op);
                    Shape optimed_permute = reorderVector(prev_permute,cur_permute);
                    if (optimed_permute != Shape {0,1,2,3}){
                    op->reshape_permute(reorderVector(prev_permute,cur_permute));
                    for(auto next_op:op->getSuccessors()){
                        next_op->replaceInput(op->getOutputs(0),prev_input);
                        prev_input->addTarget(next_op);
                        next_op->removePredecessors(op);
                        next_op->addPredecessors(prev_op);
                    }
//                auto new_op = op->getSuccessors()[0];
                    removeTensor(op->getOutput());
                    removeOperator(op);} else{
                        for(auto next_op:op->getSuccessors()){
                            next_op->replaceInput(op->getOutputs(0),prev_input);
                            next_op->removePredecessors(op);
                        }
                        removeTensor(op->getOutput());
                        removeOperator(op);
                        removeTensor(prev_op->getOutput());
                        removeOperator(prev_op);
                    }
                    continue;
                }
                op_i++;
                continue;
            }
            if (_op->getOpType() == OpType::MatMul){
                auto op = as<MatmulObj>(_op);
                auto prev_ops = _op->getPredecessors();
                auto op_input = _op->getInputs();
                for(auto prev_op:prev_ops){
                    if(prev_op->getOpType() == OpType::Transpose && prev_op->getSuccessors().size() == 1){
                        auto _prev_op = as<TransposeObj>(prev_op);
                        auto prev_op_output = prev_op->getOutput(0);
                        auto permute = _prev_op->getPermute();
                        auto rank = permute.size();
                        auto is_same = [&]() {
                            for (int r = 0; r < rank - 2; ++r) {
                                if (permute[r] != r) {
                                    return false;
                                }
                            }
                            return true;
                        }();
                        bool enable_optim = [&]() {
                            return is_same && permute[rank - 1] == rank - 2 && permute[rank - 2] == rank - 1;
                        }();
                        if (enable_optim){
                            if (prev_op_output->getFuid() ==
                                _op->getInputs(0)->getFuid()) {
                                op->setTransA(true);
                            } else {
                                op->setTransB(true);
                            }
                        }
                        auto trans_input = prev_op->getInputs(0);
                        trans_input->removeTarget(op);
                        trans_input->addTarget(op);
                        op->replaceInput(op->getInputs(0),trans_input);
                        op->removePredecessors(prev_op);
                        removeOperator(prev_op);
                        removeTensor(prev_op_output);
                    }
                }
                continue;
            }
            op_i++;
        }
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)(ans.value().size()); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
//        IT_ASSERT(tensor->getRuntime() == runtime,
//                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
//                      tensor->getRuntime()->toString() + " to " +
//                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
//            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini