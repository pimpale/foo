import LinearAlgebra.Vector

class IndexType (a : Type u) (n: ℕ) where
  to_fin : a → Fin n


structure Tensor {n : ℕ } (α : Type u) (i: IndexType) where
  data: Vector α
deriving Repr
