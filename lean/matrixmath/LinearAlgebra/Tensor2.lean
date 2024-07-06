import LinearAlgebra.Vector
import LinearAlgebra.IndexVal
import Mathlib.Data.Nat.Defs
import Batteries.Data.List.Basic


structure Tensor (α : Type u) (dims: List Nat) where
  data: Array α
  -- a proof that the data.length = n
  data_is_eq: data.size = dim_card dims
deriving Repr


namespace Tensor

open IndexVal

def reshape (t:Tensor α dims) (h: t.data.size = (dim_card dims')): Tensor α dims'
  := {
    data := t.data,
    data_is_eq := h,
  }

@[inline]
def replicate (dims: List Nat)  (x: α) : Tensor α dims := {
    data := Array.mkArray (dim_card dims) x,
    data_is_eq := Array.size_mkArray (dim_card dims) x,
}

@[inline]
def ofFnMono (f: Fin (dim_card dims) -> α) : Tensor α dims := {
  data := Array.ofFn f,
  data_is_eq := Array.size_ofFn f,
}

@[inline]
def ofFn  (f: IndexVal dims -> α) : Tensor α dims :=
  ofFnMono (λ i => f (from_fin i))

def ofArray (a:Array α) : Tensor α [a.size] := {
  data := a,
  data_is_eq := by
    unfold dim_card;
    unfold dim_card;
    simp,
}

@[inline]
def ofList (l: List α) : Tensor α [l.length] := {
  data := Array.mk l,
  data_is_eq := by
    unfold dim_card;
    unfold dim_card;
    simp,
}

syntax "!t[" withoutPosition(sepBy(term, ", ")) "]" : term
macro_rules
  | `(!t[ $elems,* ]) => `(Tensor.ofList [ $elems,* ])


@[inline]
def singleton (x:α) : Tensor α [1] :=
  Tensor.replicate [1] x

/-- prove that i < t.data.size if i < t.cardinality-/
theorem lt_n_lt_data_size  (t: Tensor α dims) (i : Fin (dim_card dims))
  : (i < t.data.size)
  := Nat.lt_of_lt_of_eq i.isLt (Eq.symm t.data_is_eq)

/-- prove that i < n if i < v.array.size-/
theorem lt_data_size_lt_n  {i :Nat}  (t: Tensor α dims) (h: i < t.data.size)
  : (i < dim_card dims)
  := t.data_is_eq.symm ▸ h


@[inline]
def getMono  (t: Tensor α dims) (i : Fin (dim_card dims)) : α :=
  t.data.get ⟨i.val, (lt_n_lt_data_size t i)⟩

@[inline]
def get (t: Tensor α dims) (i : IndexVal dims) : α :=
  getMono t (to_fin i)

-- instance to get element
instance : GetElem (Tensor α dims) (IndexVal dims) α (fun _ _ => true) where
  getElem xs i _ := xs.get i

@[inline]
def setMono (t: Tensor α dims) (i : Fin (dim_card dims)) (a : α) : Tensor α dims :=
  -- prove that i ≤ v.data.length
  let i :=  ⟨i.val, (Nat.lt_of_lt_of_eq i.isLt (Eq.symm t.data_is_eq))⟩;
  {
    data := Array.set t.data i a,
    data_is_eq := Eq.trans (Array.size_set t.data i a) t.data_is_eq,
  }

@[inline]
def set  (t: Tensor α dims) (i : IndexVal dims) (a : α) : Tensor α dims :=
  setMono t (to_fin i) a

@[inline]
def zipWith (f: α → β → γ) (t1: Tensor α dims) (t2: Tensor β dims): Tensor γ dims :=
  -- create vector
  let v1: Vector α (dim_card dims) := ⟨t1.data, t1.data_is_eq⟩;
  let v2: Vector β (dim_card dims) := ⟨t2.data, t2.data_is_eq⟩;
  -- zipWith
  let v3 := Vector.zipWith f v1 v2;
  -- back to tensor
  {
    data := v3.data,
    data_is_eq := v3.isEq,
  }

@[inline]
def map (f: α → β) (t: Tensor α dims) : Tensor β dims := {
  data := Array.map f t.data,
  data_is_eq := Eq.trans (Array.size_map f t.data) t.data_is_eq,
}

@[inline]
def mapIdxMono (f: Fin (dim_card dims) → α → β) (t: Tensor α dims) : Tensor β dims :=
  letI f' := fun (i: Fin t.data.size) => f (Fin.mk i.val (Nat.lt_of_lt_of_eq i.isLt t.data_is_eq));
  {
    data := Array.mapIdx t.data f',
    data_is_eq := Eq.trans (Array.size_mapIdx t.data f') t.data_is_eq,
  }

@[inline]
def mapIdx (f: IndexVal dims → α → β) (t: Tensor α dims) : Tensor β dims :=
  mapIdxMono (λ i a => f (from_fin i) a) t

set_option diagnostics true

def transposeDims
  (dims: List Nat)
  (permutation: List Nat)
  (h: List.Perm permutation (List.range dims.length))
: List Nat :=

  -- convert to array for O(1) access
  let perm_arr := Array.mk permutation;
  let dims_arr := Array.mk dims;

  -- prove each element of permutation is less than dims.length
  let permutation_lt_dims_arr_size : ∀ (i: Nat), i ∈ perm_arr -> i < dims_arr.size := by
    intro i;
    intro h;
    rename_i h2;
    apply List.mem_range.mp;
    apply (List.Perm.mem_iff h2).mp;
    apply (Array.mem_def i perm_arr).mp;
    exact h;

  List.ofFn (fun i:Fin perm_arr.size =>
    let p := perm_arr[i.val]'i.isLt;
    let hp := by
      apply permutation_lt_dims_arr_size;
      apply (Array.mem_def p perm_arr).mpr;
      apply Array.getElem_mem_data perm_arr i.isLt;
    dims_arr[p]'hp
  )


def transpose
  (t: Tensor α dims)
  (permutation: List Nat)
  (h: List.Perm permutation (List.range dims.length))
: Tensor α (transposeDims dims permutation h) :=
  sorry


def zero [Zero α] : Tensor α dims := Tensor.replicate dims 0

def one [One α] : Tensor α dims := Tensor.replicate dims 1

def neg [Neg α] (t: Tensor α dims) : Tensor α dims := Tensor.map (-·) t

def add [Add α] (a b: Tensor α dims) : Tensor α dims :=
  Tensor.zipWith (·+·) a b

def sub [Sub α] (a b: Tensor α dims) : Tensor α dims :=
  Tensor.zipWith (·-·) a b

def scale [Mul α] (k: α) (t: Tensor α dims) : Tensor α dims :=
  t.map (fun x => k*x)

def hadamard [Mul α] (a b: Tensor α dims) : Tensor α dims :=
  Tensor.zipWith (·*·) a b

def sum [Zero α] [Add α] (t: Tensor α dims) : α :=
  t.data.foldl (·+·) 0

def mul [Mul α] (a: Tensor α [m₁, p]) (b: Tensor α [p, n₂]) : Tensor α [m₁, n₂] :=
  let rows := a;
  let cols := b.transpose [1, 0] (by
    unfold List.length List.length List.length;
    unfold List.range;
    unfold List.range.loop List.range.loop List.range.loop;
    simp;
    apply List.Perm.swap'
    apply List.Perm.nil
  );
  sorry

/-- Object permanence??? 😳 -/
@[simp]
theorem get_set_eq_mono  (t: Tensor α dims) (i: Fin (dim_card dims)) (a: α)
  : Tensor.getMono (Tensor.setMono t i a) i = a
  := Array.get_set_eq t.data ⟨i, lt_n_lt_data_size t i⟩ a

@[simp]
theorem get_set_eq  (t: Tensor α dims) (i: IndexVal dims) (a: α)
  : Tensor.get (Tensor.set t i a) i = a
  := get_set_eq_mono t (to_fin i) a

/-- If we construct a vector through ofFn, then each element is the result of the function -/
@[simp]
theorem get_ofFnMono  (f: Fin (dim_card dims) -> α) (i: Fin (dim_card dims))
  : (Tensor.ofFnMono f).getMono i = f i
  :=
    -- prove that the i < Array.size (Array.ofFn f)
    have i_lt_size_ofFn_data : i.val < Array.size (Array.ofFn f) := lt_n_lt_data_size (ofFnMono f) i
    -- prove that v.data.get i = f i
    Array.getElem_ofFn f i.val i_lt_size_ofFn_data

/-- If we construct a vector through ofFn, then each element is the result of the function -/
@[simp]
theorem get_ofFn (f: IndexVal dims -> α) (i: IndexVal dims)
  : (Tensor.ofFn f).get i = f i
  :=
    by
      unfold Tensor.ofFn;
      unfold Tensor.get;
      rw [get_ofFnMono];
      rw [bijection]

theorem get_mapMono (f: α → β) (t: Tensor α dims) (i: Fin (dim_card dims))
  : (t.map f).getMono i = f (t.getMono i)
  := Array.getElem_map f t.data i (lt_n_lt_data_size (t.map f) i)

theorem get_map (f: α → β) (t: Tensor α dims) (i: IndexVal dims)
  : (t.map f).get i = f (t.get i)
  :=
    by
      unfold Tensor.get;
      rw [get_mapMono];

theorem get_mapIdxMono (f: Fin (dim_card dims) → α → β) (t: Tensor α dims) (i: Fin (dim_card dims))
  : (t.mapIdxMono f).getMono i = f i (t.getMono i)
  :=
    let f' := fun (i: Fin t.data.size) => f (Fin.mk i.val (Nat.lt_of_lt_of_eq i.isLt t.data_is_eq))
    Array.getElem_mapIdx t.data f' i (lt_n_lt_data_size (t.mapIdxMono f) i)


theorem get_mapIdx (f: IndexVal dims → α → β) (t: Tensor α dims) (i: IndexVal dims)
  : (t.mapIdx f).get i = f i (t.get i)
  :=
    by
      unfold Tensor.mapIdx;
      unfold Tensor.get;
      rw [get_mapIdxMono];
      rw [bijection]


@[ext]
theorem extMono (t1 t2: Tensor α dims) (h : ∀ (i : Fin (dim_card dims)), t1.getMono i = t2.getMono i) :
  t1 = t2
  :=
    -- prove that t1.data.size = t2.data.size
    have t1_data_size_eq_t2_data_size := t1.data_is_eq.trans t2.data_is_eq.symm
    -- prove that for all i < t1.data.size, t1.data.get i = t2.data.get i
    have forall_i_hi_t1_i_t2_i
      : ∀ (i : Nat) (h1: i < t1.data.size) (h2: i < t2.data.size), t1.data[i] = t2.data[i]
      := fun i h1 _ => h ⟨i, lt_data_size_lt_n t1 h1⟩;
    -- prove that t1.data = t2.data
    have t1_data_eq_t2_data :t1.data = t2.data :=
        Array.ext
            t1.data
            t2.data
            t1_data_size_eq_t2_data_size
            forall_i_hi_t1_i_t2_i

    -- prove that t1 = t2
    have t1_eq_t2: t1 = t2 := by calc
      t1 = ⟨t1.data, t1.data_is_eq⟩ := by rfl
      _ = ⟨t2.data, t2.data_is_eq⟩ := by simp [t1_data_eq_t2_data]
      t2 = t2 := by rfl
    t1_eq_t2

@[ext]
theorem ext (t1 t2: Tensor α dims) (h : ∀ (i : IndexVal dims), t1.get i = t2.get i) :
  t1 = t2
  := by
      apply extMono;
      intro i;
      have z := h (from_fin i);
      unfold Tensor.get at z;
      simp_all [bijection_inv]

end Tensor
