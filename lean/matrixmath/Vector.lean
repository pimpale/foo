import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Ring.Defs
import Mathlib.Init.Algebra.Functions
import Mathlib.Order.Basic

structure Vector (α : Type u) (n: ℕ) where
  data: List α
  -- a proof that the data.length = n
  isEq: data.length = n
deriving Repr

namespace Vector 

theorem length_replicate (n:ℕ) (a:α) :
  (List.replicate n a).length = n :=
  by
    simp

def empty (α : Type u) : Vector α 0 := {
  data := List.nil
  isEq := List.length_nil
}

def replicate (n: ℕ) (x: α) : Vector α n := {
    data := List.replicate n x,
    isEq := length_replicate n x
}

def singleton (x:α) : Vector α 1 := 
  Vector.replicate 1 x

def get (v: Vector α n) (i : Fin n) : α :=
  List.get 
    v.data 
    -- we prove that if i < n then i < data.length
    (Fin.mk i.val (lt_of_lt_of_eq i.isLt (Eq.symm v.isEq))) 

def set (v: Vector α n) (i : Fin n) (a : α) : Vector α n :=
  -- prove that i ≤ v.data.length
  let i := Fin.mk i.val (lt_of_lt_of_eq i.isLt (Eq.symm v.isEq));
  {
    data := List.set v.data i a,
    isEq := Eq.trans (List.length_set v.data i a) v.isEq 
  }

theorem min_of_same {α : Type u} [LinearOrder α] {a : α} {b : α} (h_1 : a = b) :
  min a b = a :=
    min_eq_left (le_of_eq h_1)

def zipWith {α : Type u} {β : Type u} {γ : Type u} {n: Nat} (f: α → β → γ) (v1: Vector α n) (v2: Vector β n): Vector γ n :=
  -- prove that v1 and v2 are the same length
  let v1_v2_same_len := Eq.trans v1.isEq (Eq.symm v2.isEq);
  -- prove that min (v1.length) (v2.length) = n
  let min_of_v1_v2_is_n := Eq.trans (min_of_same v1_v2_same_len) v1.isEq;
  { 
       data := List.zipWith f v1.data v2.data 
       isEq := Eq.trans (List.length_zipWith f v1.data v2.data) min_of_v1_v2_is_n
  }

def map {α : Type u} {β : Type u} {n: ℕ} (f: α → β) (v: Vector α n) : Vector β n := {
  data := List.map f v.data,
  isEq := Eq.trans (List.length_map v.data f) v.isEq   
}

def cons {α : Type u} {n: ℕ} (a: α) (v: Vector α n) : Vector α (n.succ) := 
{
  data := List.cons a v.data,
  isEq := Eq.trans (List.length_cons a v.data) (congrArg Nat.succ v.isEq)   
}

def ofFn (n: Nat) (f: Fin n -> α) : Vector α n := 
  match n with
  | 0 => Vector.empty α
  | Nat.succ i => Vector.cons (f 0) (Vector.ofFn i (fun i => f (Fin.succ i)))


instance {α : Type u} : Inhabited (Vector α 0) where
  default := Vector.empty α

instance {α : Type u} [Zero α] (n : ℕ) : Zero (Vector α n) where
  zero := Vector.replicate n 0

instance {α : Type u} [One α] (n : ℕ) : One (Vector α n) where
  one := Vector.replicate n 1

instance {α : Type u} [Neg α] {n: ℕ} : Neg (Vector α n) where
  neg (v: Vector α n):= Vector.map Neg.neg v

instance {α : Type u} [Add α] {n: ℕ} : Add (Vector α n) where
  add (v1: Vector α n)  (v2: Vector α n):= Vector.zipWith Add.add v1 v2

instance {α : Type u} [Sub α] {n: ℕ} : Sub (Vector α n) where
  sub (v1: Vector α n)  (v2: Vector α n):= Vector.zipWith Sub.sub v1 v2

def hadamard {α : Type u} [Mul α] {n: ℕ} (v1: Vector α n) (v2: Vector α n) : Vector α n :=
  Vector.zipWith Mul.mul v1 v2  

def dot {α : Type u} [Add α] [Mul α] [Zero α] {n: ℕ} (v1: Vector α n) (v2: Vector α n) : α :=
  List.foldl Add.add 0 (List.zipWith Mul.mul v1.data v2.data )

end Vector

