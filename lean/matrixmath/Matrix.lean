import «Vector»

structure Matrix (α :Type u) (m: Nat) (n: Nat) where
  -- row major order
  rows: Vector (Vector α n) m
deriving Repr

namespace Matrix
def replicate (m: ℕ) (n: ℕ) (a: α) : Matrix α m n := {
  rows := Vector.replicate m (Vector.replicate n a)
}

def ofFn (m: ℕ) (n: ℕ) (f: Fin m → Fin n → α) : Matrix α m n := {
  rows := Vector.ofFn (fun i => Vector.ofFn (f i))
}

 /-- Create a row matrix from a vector -/
def row (v : Vector α n) : Matrix α 1 n := 
  { rows := Vector.singleton v } 

/-- Create a column matrix from a vector -/
def col (v: Vector α m) : Matrix α m 1 :=
  { rows := Vector.map Vector.singleton v }

/-- Get a row of the matrix as a vector -/
def getRow (x : Matrix α m n) (i : Fin m): Vector α n :=
  x.rows.get i

/-- Get a column of the matrix as a vector -/
def getCol (x: Matrix α m n) (i : Fin n): Vector α m := 
  x.rows.map (fun v => v.get i)

/-- Update a row in the matrix -/
def setRow (x : Matrix α m n) (i : Fin m) (v: Vector α n): Matrix α m n :=
  { rows := x.rows.set i v }

/-- Update a column in the matrix -/
def setCol (x : Matrix α m n) (i : Fin n) (v: Vector α m): Matrix α m n :=
  { rows := x.rows.zipWith (fun v a => v.set i a) v }

/-- Get an element in the matrix -/
def getElem (x : Matrix α m n) (row : Fin m) (col: Fin n): α :=
  (x.getRow row).get col

-- instance to get a row
instance : GetElem (Matrix α m n) (ℕ) (Vector α n) (fun _ i => i < m) where
  getElem xs i h := xs.getRow ⟨i, h⟩

/-- Get the  i'th column of every row  -/
def indexRows (x : Matrix α m n) (idxs : Vector (Fin n) m) : Vector α m :=
  Vector.zipWith Vector.get x.rows idxs

/-- Get the  i'th row of every column  -/
def indexCols (x : Matrix α m n) (idxs : Vector (Fin m) n) : Vector α n :=
  idxs.mapIdx (fun j i => x[i][j])

/-- Update an element in the matrix -/
def setElem (x : Matrix α m n) (row : Fin m) (col: Fin n) (a:α) : Matrix α m n :=
  x.setRow row (x[row].set col a)

/-- Transpose a matrix -/
def transpose (x : Matrix α m n) : Matrix α n m :=
  Matrix.ofFn n m (fun i j => x[j][i])

@[inherit_doc]
scoped postfix:1024 "ᵀ" => Matrix.transpose

def zeros (α : Type u) [Zero α] (m: Nat) (n:Nat) : Matrix α m n :=
  Matrix.replicate m n 0

def identity (α : Type u) (n :ℕ) [Zero α] [One α] : Matrix α n n :=
  Matrix.ofFn n n (fun i j => if i == j then 1 else 0)

def mul {α : Type u} [Zero α] [Add α] [Mul α] {m₁ : ℕ} {p : ℕ} {n₂ : Nat} (a : Matrix α m₁ p) (b : Matrix α p n₂) : Matrix α m₁ n₂ :=
  let rows := a.rows;
  let cols := (bᵀ).rows;
  Matrix.ofFn m₁ n₂ (fun i j => Vector.dot rows[i] cols[j])


end Matrix