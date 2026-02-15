namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open TorchSharp
open TorchSharp.Fun.DGX

module TrainingFunctional =
  type Op<'a, 'b> = 'a -> 'b
  type TensorOp = torch.Tensor -> torch.Tensor
  type TensorPairOp = torch.Tensor -> torch.Tensor * torch.Tensor
  type TensorTripleOp = torch.Tensor -> torch.Tensor * torch.Tensor * torch.Tensor
  type Stage = IModel

  let id : TensorOp = fun x -> x

  let stage (f: TensorOp) : TensorOp = f

  let inline (->>) (lhs: Op<'a, 'b>) (rhs: Op<'b, 'c>) : Op<'a, 'c> =
    fun input -> rhs (lhs input)

  let inline (-->) (input: 'a) (op: Op<'a, 'b>) : 'b = op input

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

  let parallel3 (a: TensorOp) (b: TensorOp) (c: TensorOp) : TensorTripleOp =
    fun input ->
      let av = a input
      let bv = b input
      let cv = c input
      av, bv, cv

  let merge3
    (merge: torch.Tensor -> torch.Tensor -> torch.Tensor -> torch.Tensor)
    (triple: TensorTripleOp)
    : TensorOp =
    fun input ->
      let a, b, c = triple input
      merge a b c

  let linearSte (weight: TorchSharp.Modules.Parameter) (outDtype: torch.ScalarType) : TensorOp =
    fun input -> TorchSharp.Q4.Extension.Nvfp4Training.linearSte input weight outDtype

  let stageM (name: string) (f: TensorOp) : Stage =
    F [ name ] [] f

  let composeM (lhs: Stage) (rhs: Stage) : Stage =
    lhs =>> (rhs.Module.GetName(), rhs)

  let chainM (stages: Stage list) : Stage =
    match stages with
    | [] -> stageM "identity" id
    | head :: tail -> tail |> List.fold (fun acc s -> composeM acc s) head

  let runM (model: Stage) (input: torch.Tensor) : torch.Tensor =
    model.forward input

  let residualM (name: string) (block: Stage) : Stage =
    Fx [ name; $"{name}.block" ] [ box block ] (fun (input, args) ->
      use output = block.forward input
      output + input, args)
