import LinearAlgebra.Vector
import Aesop

-- source: https://github.com/lecopivo/LeanColls/blob/main/LeanColls/Classes/IndexType.lean
class IndexType (a : Type) where
  cardinality : ℕ
  to_fin : a → Fin cardinality
  from_fin : Fin cardinality → a
  bijection : ∀ (i : a), from_fin (to_fin i) = i

instance : IndexType (Fin n) where
  cardinality := n
  to_fin := id
  from_fin := id
  bijection := by
    intros;
    simp

instance [ait: IndexType α] [bit: IndexType β] : IndexType ( α × β ) where
  cardinality := (IndexType.cardinality α) * (IndexType.cardinality β)
  to_fin := λ (a, b) =>
      let ⟨a,ha⟩ := IndexType.to_fin a;
      let ⟨b,hb⟩ := IndexType.to_fin b;
      ⟨
        a * (IndexType.cardinality β) + b,
        by calc
          _ < a * IndexType.cardinality β + IndexType.cardinality β := by
            apply Nat.add_lt_add_left hb
          _ ≤ IndexType.cardinality α * IndexType.cardinality β := by
            rw [← Nat.succ_mul]
            apply Nat.mul_le_mul_right
            exact ha
      ⟩
  from_fin := λ ⟨i,hi⟩ =>
    have hq : i / IndexType.cardinality β < IndexType.cardinality α := Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hi);
    have b_gt_0 : 0 < IndexType.cardinality β := by
      apply Nat.pos_of_ne_zero;
      intro h;
      rw [h] at hi;
      rw [Nat.mul_zero] at hi;
      contradiction;
    have hr : i % IndexType.cardinality β < IndexType.cardinality β := Nat.mod_lt i (by assumption);
    (
      IndexType.from_fin ⟨i / IndexType.cardinality β, hq⟩,
      IndexType.from_fin ⟨i % IndexType.cardinality β, hr⟩
    )
  bijection := by
    intro p;
    cases p;
    simp;
    rename_i fst snd;
    apply And.intro;
    sorry

structure Tensor (α : Type u) (ι: Type) [i: IndexType ι] where
  data: Array α
  -- a proof that the data.length = n
  isEq: data.size = i.cardinality
deriving Repr

namespace Tensor
def proveLen {ι: Type} {ι': Type} [IndexType ι] [i': IndexType ι'] (t:Tensor α ι) (h: t.data.size = i'.cardinality): Tensor α ι' := {
  data := t.data,
  isEq := h
}

@[inline]
def replicate (ι: Type) [i: IndexType ι] (x: α) : Tensor α ι := {
    data := Array.mkArray (i.cardinality) x,
    isEq := Array.size_mkArray (i.cardinality) x
}

@[inline]
def ofFnMono [it: IndexType ι] (f: Fin it.cardinality -> α) : Tensor α ι := {
  data := Array.ofFn f,
  isEq := Array.size_ofFn f
}

@[inline]
def ofFn [IndexType ι] (f: ι -> α) : Tensor α ι :=
  ofFnMono (λ i => f (IndexType.from_fin i))

def ofArray (a:Array α) : Tensor α (Fin a.size) := {
  data := a,
  isEq := rfl
}

@[inline]
def ofList (l:List α) : Tensor α (Fin l.length) := {
  data := Array.mk l,
  isEq := Array.size_mk l
}

syntax "!t[" withoutPosition(sepBy(term, ", ")) "]" : term
macro_rules
  | `(!t[ $elems,* ]) => `(Tensor.ofList [ $elems,* ])


@[inline]
def singleton (x:α) : Tensor α (Fin 1) :=
  Tensor.replicate (Fin 1) x

/-- prove that i < t.data.size if i < t.cardinality-/
theorem lt_n_lt_data_size [it: IndexType ι] (t: Tensor α ι) (i : Fin it.cardinality)
  : (i < t.data.size)
  := Nat.lt_of_lt_of_eq i.isLt (Eq.symm t.isEq)

/-- prove that i < n if i < v.array.size-/
theorem lt_data_size_lt_n [it: IndexType ι] {i n :Nat}  (t: Tensor α ι) (h: i < t.data.size)
  : (i < it.cardinality)
  := t.isEq.symm ▸ h


@[inline]
def getMono [it: IndexType ι] (t: Tensor α ι) (i : Fin it.cardinality) : α :=
  t.data.get ⟨i.val, (lt_n_lt_data_size t i)⟩

@[inline]
def get [IndexType ι] (t: Tensor α ι) (i : ι) : α :=
  getMono t (IndexType.to_fin i)

-- instance to get element
instance [IndexType ι] : GetElem (Tensor α ι) ι α (fun _ _ => true) where
  getElem xs i _ := xs.get i

@[inline]
def setMono [it: IndexType ι] (t: Tensor α ι) (i : Fin it.cardinality) (a : α) : Tensor α ι :=
  -- prove that i ≤ v.data.length
  let i :=  ⟨i.val, (Nat.lt_of_lt_of_eq i.isLt (Eq.symm t.isEq))⟩;
  {
    data := Array.set t.data i a,
    isEq := Eq.trans (Array.size_set t.data i a) t.isEq
  }

@[inline]
def set [IndexType ι] (t: Tensor α ι) (i : ι) (a : α) : Tensor α ι :=
  setMono t (IndexType.to_fin i) a

@[inline]
def zipWith [it: IndexType ι] {α : Type u} {β : Type u} {γ : Type u} (f: α → β → γ) (t1: Tensor α ι) (t2: Tensor β ι): Tensor γ ι :=
  -- create vector
  let v1: Vector α it.cardinality := ⟨t1.data, t1.isEq⟩;
  let v2: Vector β it.cardinality := ⟨t2.data, t2.isEq⟩;
  -- zipWith
  let v3 := Vector.zipWith f v1 v2;
  -- back to tensor
  {
    data := v3.data,
    isEq := v3.isEq
  }

@[inline]
def map [IndexType ι] {α : Type u} {β : Type u} (f: α → β) (t: Tensor α ι) : Tensor β ι := {
  data := Array.map f t.data,
  isEq := Eq.trans (Array.size_map f t.data) t.isEq
}

@[inline]
def mapIdxMono [it: IndexType ι] {α : Type u} {β : Type u} (f: Fin it.cardinality → α → β) (t: Tensor α ι) : Tensor β ι :=
  letI f' := fun (i: Fin t.data.size) => f (Fin.mk i.val (Nat.lt_of_lt_of_eq i.isLt t.isEq));
  {
    data := Array.mapIdx t.data f',
    isEq := Eq.trans (Array.size_mapIdx t.data f') t.isEq
  }

@[inline]
def mapIdx [IndexType ι] {α : Type u} {β : Type u} (f: ι → α → β) (t: Tensor α ι) : Tensor β ι :=
  mapIdxMono (λ i a => f (IndexType.from_fin i) a) t


def zero [Zero α] [IndexType ι]: Tensor α ι := Tensor.replicate ι 0

def one [One α] [IndexType ι]: Tensor α ι := Tensor.replicate ι 1

def neg [Neg α] [IndexType ι] (t: Tensor α ι) : Tensor α ι := Tensor.map (-·) t

def add [Add α] [IndexType ι] (a b: Tensor α ι) : Tensor α ι :=
  Tensor.zipWith (·+·) a b

def sub [Sub α] [IndexType ι] (a b: Tensor α ι) : Tensor α ι :=
  Tensor.zipWith (·-·) a b

def scale [Mul α] [IndexType ι] (k: α) (t: Tensor α ι) : Tensor α ι :=
  t.map (fun x => k*x)

def hadamard [Mul α] [IndexType ι] (a b: Tensor α ι) : Tensor α ι :=
  Tensor.zipWith (·*·) a b


/-- Object permanence??? 😳 -/
@[simp]
theorem get_set_eq_mono [it: IndexType ι] (t: Tensor α ι) (i: Fin it.cardinality) (a: α)
  : Tensor.getMono (Tensor.setMono t i a) i = a
  := Array.get_set_eq t.data ⟨i, lt_n_lt_data_size t i⟩ a

@[simp]
theorem get_set_eq [IndexType ι] (t: Tensor α ι) (i: ι) (a: α)
  : Tensor.get (Tensor.set t i a) i = a
  := get_set_eq_mono t (IndexType.to_fin i) a

/-- If we construct a vector through ofFn, then each element is the result of the function -/
@[simp]
theorem get_ofFnMono [it: IndexType ι] (f: Fin it.cardinality -> α) (i: Fin it.cardinality)
  : (Tensor.ofFnMono f).getMono i = f i
  :=
    -- prove that the i < Array.size (Array.ofFn f)
    have i_lt_size_ofFn_data : i.val < Array.size (Array.ofFn f) := lt_n_lt_data_size (ofFn f) i
    -- prove that v.data.get i = f i
    Array.getElem_ofFn f i.val i_lt_size_ofFn_data

/-- If we construct a vector through ofFn, then each element is the result of the function -/
@[simp]
theorem get_ofFn [it: IndexType ι] (f: ι -> α) (i: ι)
  : (Tensor.ofFn f).get i = f i
  :=
    by
      unfold Tensor.ofFn;
      unfold Tensor.get;
      rw [get_ofFnMono];
      rw [it.bijection i]

theorem get_mapMono [it: IndexType ι] {α : Type u} {β : Type u}  (f: α → β) (t: Tensor α ι) (i: Fin it.cardinality)
  : (t.map f).getMono i = f (t.getMono i)
  := Array.getElem_map f t.data i (lt_n_lt_data_size (t.map f) i)

theorem get_map [IndexType ι] {α : Type u} {β : Type u}  (f: α → β) (t: Tensor α ι) (i: ι)
  : (t.map f).get i = f (t.get i)
  := get_mapMono f t (IndexType.to_fin i)

theorem get_mapIdxMono {α : Type u} {β : Type u} {n: Nat} (f: Fin n → α → β) (v: Vector α n) (i: Fin n)
  : (v.mapIdx f)[i] = f i v[i]
  :=
    letI f' := fun (i: Fin v.data.size) => f (Fin.mk i.val (Nat.lt_of_lt_of_eq i.isLt v.isEq))
    Array.getElem_mapIdx v.data f' i (lt_n_lt_data_size (v.mapIdx f) i)


end Tensor
