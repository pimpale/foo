import LinearAlgebra.Matrix
import LinearAlgebra.Vector

#check 1 + 1


def n := 100*1000*1000

def TRversion (v: Vector UInt8 i): IO (UInt64) := do
  let v1 := Vector.zipWithTR Add.add v v 
  pure v1.data.data.toByteArray.hash

def NonTRversion  (v: Vector UInt8 i): IO (UInt64) := do
  let v1 := Vector.zipWith Add.add v v
  pure v1.data.data.toByteArray.hash

def main : IO Unit := do
  let q ← IO.getRandomBytes n.toUSize
  let v2 := ⟨q.data, rfl⟩
  let x ← timeit "Tail Recursive" (TRversion v2)
  let x ← timeit "Non Tail Recursive" (NonTRversion v2)

  IO.println s!"Done"


