#pragma once
#include "core/operator.h"
#include "optimize/move_red.h"
namespace infini
{
  /**
   * @brief Transpose the input tensor similar to numpy.transpose.
   *
   */
  class TransposeObj : public OperatorObj
  {
  friend class move_red;
  public:
    /**
     * @brief Construct a new TransposeObj object.
     *
     * @param graph The graph to which this operator belongs.
     * @param input The input tensor.
     * @param output The output tensor.
     * @param permute The permutation of the dimensions.
     */

    TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                 vector<int> permute);
    OP_CLONE(TransposeObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

    std::vector<int> getPermute() const { return transposePermute; }
    void reshape_permute(vector<int> permute){transposePermute = permute;};
  private:
    vector<int> transposePermute;
  };
} // namespace infini
