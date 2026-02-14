namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open TorchSharp

module TrainingFunctional =
  type TensorOp = torch.Tensor -> torch.Tensor
  type TensorPairOp = torch.Tensor -> torch.Tensor * torch.Tensor

  let id : TensorOp = fun x -> x

  let stage (f: TensorOp) : TensorOp = f

  let inline (->>) (lhs: TensorOp) (rhs: TensorOp) : TensorOp =
    fun input ->
      let mid = lhs input
      rhs mid

  let inline (-->) (input: torch.Tensor) (op: TensorOp) : torch.Tensor = op input

  let chain (ops: TensorOp list) : TensorOp =
    ops |> List.fold (fun acc op -> acc ->> op) id

  let residual (block: TensorOp) : TensorOp =
    fun input ->
      let output = block input
      output + input

  let parallel2 (left: TensorOp) (right: TensorOp) : TensorPairOp =
    fun input ->
      let l = left input
      let r = right input
      l, r

  let merge2 (merge: torch.Tensor -> torch.Tensor -> torch.Tensor) (pair: TensorPairOp) : TensorOp =
    fun input ->
      let l, r = pair input
      merge l r

  let linearSte (weight: TorchSharp.Modules.Parameter) (outDtype: torch.ScalarType) : TensorOp =
    fun input -> TorchSharp.Q4.Extension.Nvfp4Training.linearSte input weight outDtype
