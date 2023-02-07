import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Ring.Defs

structure Vector (α : Type u) (n: ℕ) where
  data: List α
  -- a proof that the data.length = n
  isEq: Eq data.length n
deriving Repr

theorem list_replicated_n_is_n {α : Type u}  (n:ℕ) (a:α) :
  (List.replicate n a).length = n :=
    sorry

def Vector.replicate (α : Type u) (n: Nat) (x: α) : Vector α n := {
    data := List.replicate n x,
    isEq := list_replicated_n_is_n n x
  }

def Vector.zeros (α : Type u) [Zero α] (n: Nat) : Vector α n :=
  Vector.replicate α n 0

def Vector.ones (α : Type u) [One α] (n: Nat) : Vector α n :=
  Vector.replicate α n 1

def Vector.get {α : Type u} {n: Nat} (v: Vector α m) (i : Fin n) :=
  List.get v.data i v.length_proof


def Vector.dot {α : Type u} [Semiring α] {m: Nat} (v1: Vector α m) (v2: Vector α m) : α :=
  List.foldl (fun acc x => acc + x) 0 (List.zipWith (fun x y => x*y) v1.data v2.data )

def Vector.add {α : Type u} [x: Add α] {m: Nat} (v1: Vector α m) (v2: Vector α m) : Vector α m :=
  { 
       data  := List.zipWith x.add v1.data v2.data 
       length_proof := sorry
  }

def Vector.sub {α : Type u} [Sub α] {m: Nat} (v1: Vector α m) (v2: Vector α m) : Vector α m :=
  { 
    data  := List.zipWith (fun x y => x-y) v1.data v2.data,
    length_proof := sorry
  }

def Vector.hadamard {α : Type u} [Ring α] {n: ℕ} (v1: Vector α n) (v2: Vector α n) : Vector α n :=
  {
    data  := List.zipWith (fun x y => x*y) v1.data v2.data, 
    length_proof := sorry,
  }


structure Matrix (α :Type u) (m: Nat) (n: Nat) where
  -- row major order
  rows: Vector (Vector α n) m
deriving Repr

def Matrix.zeros (α : Type u) (m: Nat) (n:Nat) : Matrix α m n :=
  {   rows  := Vector.fill (Vector α n) m (Vector.zeros α n) }
 
def Matrix.col 

