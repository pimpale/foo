import «Vector»

structure Matrix (α :Type u) (m: Nat) (n: Nat) where
  -- row major order
  rows: Vector (Vector α n) m
deriving Repr

def Matrix.replicate {α : Type u} (m: ℕ) (n: ℕ) (a: α) : Matrix α m n :=
  { rows  := Vector.replicate m (Vector.replicate n a) }

def Matrix.zeros (α : Type u) [Zero α] (m: Nat) (n:Nat) : Matrix α m n :=
  Matrix.replicate m n 0
 
def Matrix.row {α: Type u} (v : Vector α n) : Matrix α 1 n := 
  { rows := Vector.singleton v } 

def Matrix.col {α : Type u} (v: Vector α m) : Matrix α m 1 :=
  { rows := Vector.map Vector.singleton v }