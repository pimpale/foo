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

def ofArray (a:Array α) : Vector α (a.size) := {
  data := a,
  isEq := rfl
}

@[inline]
def ofList (l:List α) : Vector α (l.length) := {
  data := Array.mk l,
  isEq := Array.size_mk l
}

syntax "!v[" withoutPosition(sepBy(term, ", ")) "]" : term
macro_rules
  | `(!v[ $elems,* ]) => `(Vector.ofList [ $elems,* ])


@[inline]
def singleton (x:α) : Vector α 1 := 
  Vector.replicate 1 x

/-- prove that i < v.data.size if i < n-/
theorem lt_n_lt_data_size {α : Type u} {n :ℕ} (v: Vector α n) (i : Fin n)
  : (i < v.data.size)
  := lt_of_lt_of_eq i.isLt (Eq.symm v.isEq)

/-- prove that i < n if i < v.array.size-/
theorem lt_data_size_lt_n {α : Type u} {i n :ℕ}  (v: Vector α n) (h: i < v.data.size) 
  : (i < n)
  := v.isEq.symm ▸ h

@[inline]
def get (v: Vector α n) (i : Fin n) : α :=
  v.data[i]'(lt_n_lt_data_size v i)

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
    have n'_ne_n := (Ne.intro h1).symm;
    have n'_lt_n := Nat.lt_of_le_of_ne h (n'_ne_n);
    have n'_succ_le_n := Nat.succ_le_of_lt n'_lt_n;
    v.pop.truncate n' (Nat.pred_le_pred n'_succ_le_n)

@[specialize]
def zipWithAux (f : α → β → γ) (as : Vector α n) (bs : Vector β n) (cs : Vector γ i) : Vector γ n :=
  if h1: i < n then
    let a := as[i]'h1;
    let b := bs[i]'h1;
    zipWithAux f as bs (cs.push (f a b))
  else
    cs.truncate n (Nat.not_lt.mp h1)
termination_by _ => n - i

def zipWith {α : Type u} {β : Type u} {γ : Type u} {n: Nat} (f: α → β → γ) (v1: Vector α n) (v2: Vector β n): Vector γ n :=
  zipWithAux f v1 v2 (Vector.empty γ)

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


-- Some theorems

/-- Object permanence??? 😳 -/
theorem get_set_eq {α: Type u} {n: ℕ} (v: Vector α n) (i: Fin n) (a: α)
  : Vector.get (Vector.set v i a) i = a
  := Array.get_set_eq v.data ⟨i, lt_n_lt_data_size v i⟩ a


theorem ext {α: Type u} {n: ℕ} (v1 v2: Vector α n) (h : ∀ (i : Fin n), v1[i] = v2[i]) :
  v1 = v2
  :=
    -- prove that v1.data.size = v2.data.size
    have v1_data_size_eq_v2_data_size := v1.isEq.trans v2.isEq.symm
    -- prove that for all i < v1.data.size, v1.data.get i = v2.data.get i
    have forall_i_hi_v1_i_v2_i 
      : ∀ (i : ℕ) (h1: i < v1.data.size) (h2: i < v2.data.size), v1.data[i] = v2.data[i] 
      := fun i h1 _ => h ⟨i, lt_data_size_lt_n v1 h1⟩;
    -- prove that v1.data = v2.data
    have v1_data_eq_v2_data :v1.data = v2.data := 
        Array.ext
            v1.data
            v2.data 
            v1_data_size_eq_v2_data_size 
            forall_i_hi_v1_i_v2_i
    -- prove that v1 = v2
    have v1_eq_v2: v1 = v2 := by calc
      v1 = ⟨v1.data, v1.isEq⟩ := by rfl
      _ = ⟨v2.data, v2.isEq⟩ := by simp [v1_data_eq_v2_data]
      _ = v2 := by rfl
    v1_eq_v2

    

end Vector