import LinearAlgebra.Matrix
import LinearAlgebra.Vector

#check 1 + 1


def n := 10*1000

def NonTRversion  (v: Vector UInt8 i) (m: Matrix UInt8 i i): IO (UInt64) := do
  let v1 := Matrix.mul (Matrix.row v) m 
  pure (ByteArray.mk v1[0].data).hash

def main : IO Unit := do
  let q ← IO.getRandomBytes n.toUSize
  let v2 := ⟨q.data, rfl⟩
  let m := Matrix.identity UInt8 q.data.size
  let x ← timeit "Non Tail Recursive" (NonTRversion v2 m)

  IO.println s!"Done"


