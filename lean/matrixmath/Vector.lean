import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Ring.Defs
import Mathlib.Init.Algebra.Functions
import Mathlib.Order.Basic

structure Vector (α : Type u) (n: ℕ) where
  data: Array α
  -- a proof that the data.length = n
  isEq: data.size = n
deriving Repr

namespace Vector 

def proveLen {n:ℕ} {n':ℕ} (v:Vector α n) (h: v.data.size = n'): Vector α n' := {
  data := v.data,
  isEq := h
}

@[inline]
def empty (α : Type u) : Vector α 0 := {
  data := Array.empty
  isEq := List.length_nil
}

@[inline]
def replicate (n: ℕ) (x: α) : Vector α n := {
    data := Array.mkArray n x,
    isEq := Array.size_mkArray n x
}

@[inline]
def ofFn {n: Nat} (f: Fin n -> α) : Vector α n := {
  data := Array.ofFn f,
  isEq := Array.size_ofFn f
}

@[inline]
def singleton (x:α) : Vector α 1 := 
  Vector.replicate 1 x

@[inline]
def get (v: Vector α n) (i : Fin n) : α :=
  -- prove that i ≤ v.data.length
  v.data[i]'(lt_of_lt_of_eq i.isLt (Eq.symm v.isEq))

-- instance to get element
instance : GetElem (Vector α n) Nat α (fun _ i => i < n) where
  getElem xs i h := xs.get ⟨i, h⟩

@[inline]
def set (v: Vector α n) (i : Fin n) (a : α) : Vector α n :=
  -- prove that i ≤ v.data.length
  let i := Fin.mk i.val (lt_of_lt_of_eq i.isLt (Eq.symm v.isEq));
  {
    data := Array.set v.data i a,
    isEq := Eq.trans (Array.size_set v.data i a) v.isEq 
  }

@[inline]
def push (v: Vector α n) (a : α) : Vector α (n + 1) :=  {
  data := Array.push v.data a,
  isEq := Eq.trans (Array.size_push v.data a) (congrArg Nat.succ v.isEq) 
}

@[inline]
def pop {α: Type u} {n : ℕ} (v: Vector α n) : Vector α (n - 1) :=  {
  data := Array.pop v.data,
  isEq := Eq.trans (Array.size_pop v.data) (congrArg Nat.pred v.isEq)
}

@[inline]
def truncate {α: Type u} {n : ℕ} (v: Vector α n) (n': ℕ) (h: n' ≤ n): Vector α n' :=  
  if h1: n = n' then
   v.proveLen (v.isEq.trans h1)
  else 
    let n'_ne_n := (Ne.intro h1).symm;
    let n'_lt_n := Nat.lt_of_le_of_ne h (n'_ne_n);
    let n'_succ_le_n := Nat.succ_le_of_lt n'_lt_n;
    v.pop.truncate n' (Nat.pred_le_pred n'_succ_le_n)

@[specialize]
def zipWithAux (f : α → β → γ) (as : Vector α n) (bs : Vector β n) (i : Nat) (cs : Vector γ i) : Vector γ n :=
  if h1: i < n then
    let a := as[i]'h1
    let b := bs[i]'h1
    zipWithAux f as bs i.succ (cs.push (f a b))
  else
    cs.truncate n ((Nat.not_lt).mp h1)
termination_by _ => n - i

def zipWith {α : Type u} {β : Type u} {γ : Type u} {n: Nat} (f: α → β → γ) (v1: Vector α n) (v2: Vector β n): Vector γ n :=
  zipWithAux f v1 v2 0 (Vector.empty γ)

@[inline]
def map {α : Type u} {β : Type u} {n: ℕ} (f: α → β) (v: Vector α n) : Vector β n := {
  data := Array.map f v.data,
  isEq := Eq.trans (Array.size_map f v.data) v.isEq   
}

@[inline]
def mapIdx {α : Type u} {β : Type u} {n: ℕ} (f: Fin n → α → β) (v: Vector α n) : Vector β n := 
let f' := fun (i: Fin v.data.size) => f (Fin.mk i.val (i.isLt.trans_eq v.isEq));
{
  data := Array.mapIdx v.data f',
  isEq := Eq.trans (Array.size_mapIdx v.data f') v.isEq   
}


instance : Inhabited (Vector α 0) where default := empty α

def zeros [Zero α] (n:ℕ): Vector α n := Vector.replicate n 0
instance [Zero α] : Zero (Vector α n) where zero := zeros n

def ones [One α] (n:ℕ): Vector α n := Vector.replicate n 1
instance [One α] : One (Vector α n) where one := ones n

def neg [Neg α] (v: Vector α n) : Vector α n := Vector.map Neg.neg v
instance [Neg α] : Neg (Vector α n) where neg := neg

def add [Add α] (v1: Vector α n) (v2: Vector α n) : Vector α n :=
  Vector.zipWith Add.add v1 v2

instance {α : Type u} [Add α] {n: ℕ} : Add (Vector α n) where add := add

def sub {α : Type u} [Sub α] {n: ℕ} (v1: Vector α n) (v2: Vector α n) : Vector α n :=
  Vector.zipWith Sub.sub v1 v2

instance {α : Type u} [Sub α] {n: ℕ} : Sub (Vector α n) where
  sub := Vector.sub

def hadamard {α : Type u} [Mul α] {n: ℕ} (v1: Vector α n) (v2: Vector α n) : Vector α n :=
  Vector.zipWith Mul.mul v1 v2  

def dot {α : Type u} [Add α] [Mul α] [Zero α] {n: ℕ} (v1: Vector α n) (v2: Vector α n) : α :=
  Array.foldl Add.add 0 (Array.zipWith v1.data v2.data Mul.mul)

end Vector

