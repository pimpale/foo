import LinearAlgebra.Vector

structure Matrix (α :Type u) (m: Nat) (n: Nat) where
  -- row major order
  rows: Vector (Vector α n) m
deriving Repr

namespace Matrix
def replicate (m: ℕ) (n: ℕ) (a: α) : Matrix α m n := {
  rows := Vector.replicate m (Vector.replicate n a)
}

def ofFn (f: Fin m → Fin n → α) : Matrix α m n :=
  Matrix.mk (Vector.ofFn (fun i => Vector.ofFn (f i)))


def empty {α : Type u} : Matrix α 0 0 :=
  Matrix.mk Vector.empty


example (_:Matrix ℕ 3 3) := Matrix.mk !v[
   !v[1,2,3],
   !v[1,2,3],
   !v[1,2,3]
]

 /-- Create a row matrix from a vector -/
def row (v : Vector α n) : Matrix α 1 n :=
  Matrix.mk (Vector.singleton v)

/-- Create a column matrix from a vector -/
def col (v: Vector α m) : Matrix α m 1 :=
  Matrix.mk (Vector.map Vector.singleton v)

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
  (x.rows.get row).get col

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
  Matrix.ofFn (fun i j => x[j][i])

/-- Map each element of a matrix -/
def map (f: α → β) (x : Matrix α m n) : Matrix β m n :=
  Matrix.mk (x.rows.map (Vector.map f))

/-- Map each element of a matrix with its index -/
def mapIdx (f: Fin m → Fin n → α → β) (x : Matrix α m n) : Matrix β m n :=
  { rows := x.rows.mapIdx (fun i v => v.mapIdx (fun j a => f i j a)) }

/-- Create a matrix by applying a binary function to each element of two matrices -/
def zipWith (f: α → β → γ) (x : Matrix α m n) (y : Matrix β m n) : Matrix γ m n :=
  Matrix.mk (x.rows.zipWith (fun v1 v2 => v1.zipWith f v2) y.rows)

@[inherit_doc]
scoped postfix:1024 "ᵀ" => Matrix.transpose

@[ext]
theorem ext {α: Type u} {m n: ℕ} (m1 m2: Matrix α m n) (h : ∀ (i : Fin m) (j : Fin n), m1[i][j] = m2[i][j])
  : m1 = m2
  :=
    -- prove m1.rows[i] = m2.rows[i] for all i
    have m1_row_i_eq_m2_row_i : ∀ (i : Fin m), m1.rows[i] = m2.rows[i] :=
        fun i => Vector.ext (m1.rows.get i) (m2.rows.get i) (h i)
    -- prove m1.rows = m2.rows
    have hrows : m1.rows = m2.rows :=
        Vector.ext m1.rows m2.rows m1_row_i_eq_m2_row_i;
    -- prove m1 = m2
    congrArg Matrix.mk hrows


theorem get_mk_rows (rows: Vector (Vector α n) m)
  : (Matrix.mk rows).rows = rows
  := rfl

theorem getElem_rows (x: Matrix α m n) (i: Fin m) (j: Fin n)
  : x[i][j] = x.rows[i][j]
  := rfl

@[simp]
theorem get_ofFn (f: Fin m → Fin n → α) (i : Fin m) (j : Fin n)
  : (Matrix.ofFn f)[i][j] = f i j
  := by
      rewrite [getElem_rows (Matrix.ofFn f) i j]
      unfold Matrix.ofFn
      rewrite [get_mk_rows _]
      rewrite [Vector.get_ofFn (fun i => Vector.ofFn (f i)) i]
      rewrite [Vector.get_ofFn (f i) j]
      rfl


@[simp]
theorem get_map (f: α → β) (x: Matrix α m n) (i : Fin m) (j: Fin n)
  : (x.map f)[i][j] = f x[i][j]
  := by
      rewrite [getElem_rows x i j]
      rewrite [getElem_rows (map f x) i j]
      unfold Matrix.map
      rewrite [get_mk_rows _]
      rewrite [Vector.get_map (Vector.map f) x.rows i]
      rewrite [Vector.get_map f x.rows[i] j]
      rfl

@[simp]
theorem get_mapIdx (f: Fin m → Fin n → α → β) (x: Matrix α m n) (i : Fin m) (j: Fin n)
  : (x.mapIdx f)[i][j] = f i j x[i][j]
  := by
      rewrite [getElem_rows x i j]
      rewrite [getElem_rows (mapIdx f x) i j]
      unfold Matrix.mapIdx
      rewrite [get_mk_rows _]
      rewrite [Vector.get_mapIdx (fun i v => Vector.mapIdx (fun j a => f i j a) v) x.rows i]
      rewrite [Vector.get_mapIdx (fun j a => f i j a) x.rows[i] j]
      rfl


@[simp]
theorem get_zipWith (f: α → β → γ) (x: Matrix α m n) (y: Matrix β m n) (i : Fin m) (j: Fin n)
  : (Matrix.zipWith f x y)[i][j] = f x[i][j] y[i][j]
  := by
      rewrite [getElem_rows x i j]
      rewrite [getElem_rows y i j]
      rewrite [getElem_rows (zipWith f x y) i j]
      unfold Matrix.zipWith
      rewrite [get_mk_rows (Vector.zipWith (fun v1 v2 => Vector.zipWith f v1 v2) x.rows y.rows)]
      rewrite [Vector.get_zipWith (fun v1 v2 => Vector.zipWith f v1 v2) x.rows y.rows i]
      rewrite [Vector.get_zipWith f x.rows[i] y.rows[i] j]
      rfl


theorem transpose_elem (a : Matrix α m n ) (i : Fin m) (j : Fin n)
  : aᵀ[j][i] = a[i][j]
  :=
      -- definition of transpose
      Matrix.get_ofFn (fun i j => a[j][i]) j i

@[simp]
theorem transpose_transpose_elem (a : Matrix α m n ) (i : Fin m) (j : Fin n)
  : aᵀᵀ[i][j] = a[i][j]
  :=
      -- prove that aᵀᵀ[i][j] = aᵀ[j][i]
      have att_ij_eq_at_ji := transpose_elem aᵀ j i
      -- prove that aᵀ[j][i] = a[i][j]
      have at_ji_eq_a_ij := transpose_elem a i j
      -- prove that aᵀᵀ[i][j] = a[i][j]
      att_ij_eq_at_ji.trans at_ji_eq_a_ij

@[simp]
theorem transpose_transpose (a : Matrix α m n)
  : aᵀᵀ = a
  := Matrix.ext aᵀᵀ a (transpose_transpose_elem a)


def zero {α : Type u} [Zero α] {m n: ℕ} : Matrix α m n :=
  Matrix.replicate m n 0

def identity {α : Type u} {n :ℕ} [Zero α] [One α] : Matrix α n n :=
  Matrix.ofFn (fun i j => if i == j then 1 else 0)

def mul {α : Type u} [Zero α] [Add α] [Mul α] {m₁ : ℕ} {p : ℕ} {n₂ : Nat} (a : Matrix α m₁ p) (b : Matrix α p n₂) : Matrix α m₁ n₂ :=
  let rows := a.rows;
  let cols := (bᵀ).rows;
  Matrix.ofFn (fun i j => Vector.dot rows[i] cols[j])

def neg {α : Type u} [Neg α] {m n: ℕ} (a : Matrix α m n) : Matrix α m n :=
  Matrix.map (-·) a

def add {α : Type u} [Add α] {m n: ℕ} (a b : Matrix α m n) : Matrix α m n :=
  Matrix.zipWith (·+·) a b

def sub {α : Type u} [Sub α] {m n: ℕ} (a b : Matrix α m n) : Matrix α m n :=
  Matrix.zipWith (·-·) a b

def hadamard {α : Type u} [Mul α] {m n: ℕ} (a b : Matrix α m n) : Matrix α m n :=
  Matrix.zipWith (·*·) a b

instance : Inhabited (Matrix α 0 0) where default := empty
instance [Zero α] : Zero (Matrix α n m) where zero := zero
instance [One α] [Zero α] : One (Matrix α n n) where one := identity
instance [Neg α] : Neg (Matrix α n m) where neg := neg
instance {α : Type u} [Add α] {n m: ℕ} : Add (Matrix α n m) where add := add
instance {α : Type u} [Sub α] {n m: ℕ} : Sub (Matrix α n m) where sub := sub

end Matrix
