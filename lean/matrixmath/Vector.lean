import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Ring.Defs
import Mathlib.Init.Algebra.Functions
import Mathlib.Order.Basic


structure Vector (α : Type u) (n: ℕ) where
  data: List α
  -- a proof that the data.length = n
  isEq: data.length = n
deriving Repr

theorem length_replicate {α : Type u}  (n:ℕ) (a:α) :
  (List.replicate n a).length = n :=
  by
    simp

def Vector.empty (α : Type u) : Vector α 0 := {
  data := List.nil
  isEq := List.length_nil
}

def Vector.replicate {α : Type u} (n: Nat) (x: α) : Vector α n := {
    data := List.replicate n x,
    isEq := length_replicate n x
}

def Vector.singleton {α : Type u} (x:α) : Vector α 1 := 
  Vector.replicate 1 x

def Vector.zeros (α : Type u) [Zero α] (n: Nat) : Vector α n :=
  Vector.replicate n 0

def Vector.ones (α : Type u) [One α] (n: Nat) : Vector α n :=
  Vector.replicate n 1

def Vector.get {α : Type u} {n: Nat} (v: Vector α n) (i : Fin n) :=
  List.get 
    v.data 
    -- we prove that if i < n then i < data.length
    (Fin.mk i.val (lt_of_lt_of_eq i.isLt (Eq.symm v.isEq))) 

def Vector.dot {α : Type u} [Add α] [Mul α] [Zero α] {n: ℕ} (v1: Vector α n) (v2: Vector α n) : α :=
  List.foldl Add.add 0 (List.zipWith Mul.mul v1.data v2.data )

theorem min_of_same {α : Type u} [LinearOrder α] {a : α} {b : α} (h_1 : a = b) :
  min a b = a :=
    min_eq_left (le_of_eq h_1)

def Vector.zipWith {α : Type u} {β : Type u} {γ : Type u} {n: Nat} (f: α → β → γ) (v1: Vector α n) (v2: Vector β n): Vector γ n :=
  -- prove that v1 and v2 are the same length
  let v1_v2_same_len := Eq.trans v1.isEq (Eq.symm v2.isEq);
  -- prove that min (v1.length) (v2.length) = n
  let min_of_v1_v2_is_n := Eq.trans (min_of_same v1_v2_same_len) v1.isEq;
  { 
       data := List.zipWith f v1.data v2.data 
       isEq := Eq.trans (List.length_zipWith f v1.data v2.data) min_of_v1_v2_is_n
  }

def Vector.map {α : Type u} {β : Type u} {n: ℕ} (f: α → β) (v: Vector α n) : Vector β n := {
  data := List.map f v.data,
  isEq := Eq.trans (List.length_map v.data f) v.isEq   
}

def Vector.add {α : Type u} [Add α] {n: ℕ} (v1: Vector α n) (v2: Vector α n) : Vector α n :=
  Vector.zipWith Add.add v1 v2

def Vector.sub {α : Type u} [Sub α] {n: ℕ} (v1: Vector α n) (v2: Vector α n) : Vector α n :=
  Vector.zipWith Sub.sub v1 v2

def Vector.hadamard {α : Type u} [Mul α] {n: ℕ} (v1: Vector α n) (v2: Vector α n) : Vector α n :=
  Vector.zipWith Mul.mul v1 v2

instance {α : Type u} : Inhabited (Vector α 0) where
  default := Vector.empty α
