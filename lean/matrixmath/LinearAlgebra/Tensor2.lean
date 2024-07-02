import LinearAlgebra.Vector
import Aesop

inductive IndexType : Type where
  | Mono : Nat -> IndexType
  | Multi : Nat -> IndexType -> IndexType

#check IndexType.Multi 10 (IndexType.Mono 5)

inductive IndexVal : IndexType -> Type where
  | Mono : Fin n -> IndexVal (IndexType.Mono n)
  | Multi : Fin n -> (IndexVal tail_t) -> IndexVal (IndexType.Multi n tail_t)

#check IndexVal (IndexType.Multi 10 (IndexType.Mono 5))

def card : IndexType → Nat
  | (IndexType.Mono n) => n
  | (IndexType.Multi n tail) => n * card tail

def to_fin : IndexVal dims → Fin (card dims)
  | (IndexVal.Mono i) => i
  | (IndexVal.Multi head tail) =>
    by
      -- n is the cardinality of the head
      -- tail_t is the tail
      rename_i n tail_t;
      let ⟨a, ha⟩ := head;
      let ⟨b, hb⟩ := to_fin tail;
      let v := a * card tail_t + b
      let hv : v < card (IndexType.Multi n tail_t) := by calc
            _ < a * card tail_t + card tail_t := by
              apply Nat.add_lt_add_left hb
            _ ≤ n * card tail_t := by
              rw [← Nat.succ_mul]
              apply Nat.mul_le_mul_right
              exact ha
      exact ⟨v, hv⟩

def from_fin {bound: IndexType} (i: Fin (card bound)): IndexVal bound :=
  match bound with
  | IndexType.Mono n =>
    IndexVal.Mono i
  | IndexType.Multi n tail_t =>
  let ⟨i, hi⟩ := i;
    have hq : i / card tail_t < n := Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hi);
    have b_gt_0 : 0 < card tail_t := by
      apply Nat.pos_of_ne_zero;
      intro h;
      unfold card at hi;
      rw [h] at hi;
      rw [Nat.mul_zero] at hi;
      contradiction;

    have hr : i % card tail_t < card tail_t := Nat.mod_lt i (by assumption);
    IndexVal.Multi ⟨i / card tail_t, hq⟩ (from_fin ⟨i % card tail_t, hr⟩)

def bijection (it: IndexType) : ∀ (i : IndexVal it), from_fin (to_fin i) = i
 := by
    intro i;
    induction i with
    | Mono i =>
      unfold from_fin;
      unfold to_fin;
      rfl
    | Multi head tail =>
      rename_i a_ih;
      rename_i tail_t;
      unfold from_fin;
      unfold to_fin;
      simp;
      apply And.intro;
      sorry
      sorry

structure Tensor (α : Type u) (dims: List Nat) where
  data: Array α
  -- a proof that the data.length = n
  isEq: data.size = card dims
deriving Repr


namespace Tensor
def reshape
  {dims: List Nat}
  {dims': List Nat}
  (t:Tensor α dims)
  (h: t.data.size = (card dims'))
: Tensor α dims' := {
  data := t.data,
  isEq := h
}

@[inline]
def replicate (dims: List Nat)  (x: α) : Tensor α dims := {
    data := Array.mkArray (card dims) x,
    isEq := Array.size_mkArray (card dims) x
}

@[inline]
def ofFnMono (f: Fin (card dims) -> α) : Tensor α dims := {
  data := Array.ofFn f,
  isEq := Array.size_ofFn f
}

@[inline]
def ofFn  (f: List Nat -> α) : Tensor α ι :=
  ofFnMono (λ i => f )

def ofArray (a:Array α) : Tensor α [a.size] := {
  data := a,
  isEq := sorry
}

@[inline]
def ofList (l: List α) : Tensor α [l.length] := {
  data := Array.mk l,
  isEq := sorry
}

syntax "!t[" withoutPosition(sepBy(term, ", ")) "]" : term
macro_rules
  | `(!t[ $elems,* ]) => `(Tensor.ofList [ $elems,* ])


@[inline]
def singleton (x:α) : Tensor α [1] :=
  Tensor.replicate [1] x

/-- prove that i < t.data.size if i < t.cardinality-/
theorem lt_n_lt_data_size  (t: Tensor α dims) (i : Fin (card dims))
  : (i < t.data.size)
  := Nat.lt_of_lt_of_eq i.isLt (Eq.symm t.isEq)

/-- prove that i < n if i < v.array.size-/
theorem lt_data_size_lt_n  {i :Nat}  (t: Tensor α dims) (h: i < t.data.size)
  : (i < card dims)
  := t.isEq.symm ▸ h


@[inline]
def getMono  (t: Tensor α dims) (i : Fin (card dims)) : α :=
  t.data.get ⟨i.val, (lt_n_lt_data_size t i)⟩

@[inline]
def get  (t: Tensor α dims) (i : dims.map Fin) : α :=
  getMono t

-- instance to get element
instance  : GetElem (Tensor α ι) ι α (fun _ _ => true) where
  getElem xs i _ := xs.get i

@[inline]
def setMono  (t: Tensor α ι) (i : Fin it.cardinality) (a : α) : Tensor α ι :=
  -- prove that i ≤ v.data.length
  let i :=  ⟨i.val, (Nat.lt_of_lt_of_eq i.isLt (Eq.symm t.isEq))⟩;
  {
    data := Array.set t.data i a,
    isEq := Eq.trans (Array.size_set t.data i a) t.isEq
  }

@[inline]
def set  (t: Tensor α ι) (i : ι) (a : α) : Tensor α ι :=
  setMono t  a

@[inline]
def zipWith  {α : Type u} {β : Type u} {γ : Type u} (f: α → β → γ) (t1: Tensor α ι) (t2: Tensor β ι): Tensor γ ι :=
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
def map  {α : Type u} {β : Type u} (f: α → β) (t: Tensor α ι) : Tensor β ι := {
  data := Array.map f t.data,
  isEq := Eq.trans (Array.size_map f t.data) t.isEq
}

@[inline]
def mapIdxMono  {α : Type u} {β : Type u} (f: Fin it.cardinality → α → β) (t: Tensor α ι) : Tensor β ι :=
  letI f' := fun (i: Fin t.data.size) => f (Fin.mk i.val (Nat.lt_of_lt_of_eq i.isLt t.isEq));
  {
    data := Array.mapIdx t.data f',
    isEq := Eq.trans (Array.size_mapIdx t.data f') t.isEq
  }

@[inline]
def mapIdx  {α : Type u} {β : Type u} (f: ι → α → β) (t: Tensor α ι) : Tensor β ι :=
  mapIdxMono (λ i a => f  a) t


def zero [Zero α] : Tensor α ι := Tensor.replicate ι 0

def one [One α] : Tensor α ι := Tensor.replicate ι 1

def neg [Neg α]  (t: Tensor α ι) : Tensor α ι := Tensor.map (-·) t

def add [Add α]  (a b: Tensor α ι) : Tensor α ι :=
  Tensor.zipWith (·+·) a b

def sub [Sub α]  (a b: Tensor α ι) : Tensor α ι :=
  Tensor.zipWith (·-·) a b

def scale [Mul α]  (k: α) (t: Tensor α ι) : Tensor α ι :=
  t.map (fun x => k*x)

def hadamard [Mul α]  (a b: Tensor α ι) : Tensor α ι :=
  Tensor.zipWith (·*·) a b


/-- Object permanence??? 😳 -/
@[simp]
theorem get_set_eq_mono  (t: Tensor α ι) (i: Fin it.cardinality) (a: α)
  : Tensor.getMono (Tensor.setMono t i a) i = a
  := Array.get_set_eq t.data ⟨i, lt_n_lt_data_size t i⟩ a

@[simp]
theorem get_set_eq  (t: Tensor α ι) (i: ι) (a: α)
  : Tensor.get (Tensor.set t i a) i = a
  := get_set_eq_mono t  a

/-- If we construct a vector through ofFn, then each element is the result of the function -/
@[simp]
theorem get_ofFnMono  (f: Fin it.cardinality -> α) (i: Fin it.cardinality)
  : (Tensor.ofFnMono f).getMono i = f i
  :=
    -- prove that the i < Array.size (Array.ofFn f)
    have i_lt_size_ofFn_data : i.val < Array.size (Array.ofFn f) := lt_n_lt_data_size (ofFn f) i
    -- prove that v.data.get i = f i
    Array.getElem_ofFn f i.val i_lt_size_ofFn_data

/-- If we construct a vector through ofFn, then each element is the result of the function -/
@[simp]
theorem get_ofFn  (f: ι -> α) (i: ι)
  : (Tensor.ofFn f).get i = f i
  :=
    by
      unfold Tensor.ofFn;
      unfold Tensor.get;
      rw [get_ofFnMono];
      rw [it.bijection i]

theorem get_mapMono  {α : Type u} {β : Type u}  (f: α → β) (t: Tensor α ι) (i: Fin it.cardinality)
  : (t.map f).getMono i = f (t.getMono i)
  := Array.getElem_map f t.data i (lt_n_lt_data_size (t.map f) i)

theorem get_map  {α : Type u} {β : Type u}  (f: α → β) (t: Tensor α ι) (i: ι)
  : (t.map f).get i = f (t.get i)
  := get_mapMono f t

theorem get_mapIdxMono {α : Type u} {β : Type u} {n: Nat} (f: Fin n → α → β) (v: Vector α n) (i: Fin n)
  : (v.mapIdx f)[i] = f i v[i]
  :=
    letI f' := fun (i: Fin v.data.size) => f (Fin.mk i.val (Nat.lt_of_lt_of_eq i.isLt v.isEq))
    Array.getElem_mapIdx v.data f' i (lt_n_lt_data_size (v.mapIdx f) i)


end Tensor
