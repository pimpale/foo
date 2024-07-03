import LinearAlgebra.Vector
import Aesop
import Mathlib.Data.Nat.Defs

inductive IndexVal : List ℕ -> Type where
  | Nil : IndexVal []
  | Cons : Fin n -> (IndexVal tail_t) -> IndexVal (n :: tail_t)

#check IndexVal.Cons 0 (IndexVal.Nil)
#check IndexVal [1, 2, 3]

-- def example_index : IndexVal [1, 2, 3] := IndexVal.Cons 0 (IndexVal.Cons 1 (IndexVal.Cons 2 IndexVal.Nil))

def card : List ℕ -> ℕ
  | [] => 1
  | (n :: tail) => n * card tail

def to_fin : IndexVal dims → Fin (card dims)
  | IndexVal.Nil =>
    let v := 0;
    let hv := by
      unfold card;
      simp;
    ⟨v,hv⟩
  | (IndexVal.Cons head tail) =>
    by
      -- n is the cardinality of the head
      -- tail_t is the tail
      rename_i n tail_t;
      let ⟨a, ha⟩ := head;
      let ⟨b, hb⟩ := to_fin tail;
      let v := a * card tail_t + b
      let hv : v < card (n :: tail_t) := by calc
            _ < a * card tail_t + card tail_t := by
              apply Nat.add_lt_add_left hb
            _ ≤ n * card tail_t := by
              rw [← Nat.succ_mul]
              apply Nat.mul_le_mul_right
              exact ha
      exact ⟨v, hv⟩

def from_fin {bound: List ℕ} (i: Fin (card bound)): IndexVal bound :=
  match bound with
  | [] => IndexVal.Nil
  | n :: tail_t =>
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
    IndexVal.Cons ⟨i / card tail_t, hq⟩ (from_fin ⟨i % card tail_t, hr⟩)

theorem div_add {m n k : ℕ} (h: k < n) : (m * n + k) / n = m
  := by
      -- prove 0 < n
      have zero_lt_n : 0 < n := by
        have k_ge_0 : k ≥ 0 := by
          apply Nat.zero_le;
        apply Nat.lt_of_le_of_lt k_ge_0 h;
      -- convert to (k + m * n) / n = m
      rw [Nat.add_comm];
      -- convert to  k / n + m = m
      rw [Nat.add_mul_div_right k m zero_lt_n];
      -- convert to k / n = 0
      simp;
      -- convert to k < n
      rw [Nat.div_eq_of_lt h]

def bijection (it: List Nat) : ∀ (i : IndexVal it), from_fin (to_fin i) = i
 := by
    intro i;
    induction i with
    | Nil =>
      unfold from_fin;
      rfl
    | Cons head tail =>
      rename_i a_ih;
      rename_i tail_t;
      unfold from_fin;
      unfold to_fin;
      simp;
      apply And.intro;
      -- only care about the fin val
      case Cons.left =>
        -- we have an expression of (m * n + k) / n = m and a hypothesis k < n
        -- need to reduce to m
        have to_fin_tail_lt_card_tail : ↑(to_fin tail) < card tail_t := by
          sorry;
        simp [div_add];
      case Cons.right =>
        sorry;

def bijection_inv (it: List Nat) : ∀ (i : Fin (card it)), to_fin (from_fin i) = i
 := by
      intro i;
      induction it with
      | nil =>
        simp [card, Fin.fin_one_eq_zero]
      | cons n tail_t ih =>
        unfold to_fin;
        unfold from_fin;
        simp [ih, Nat.div_add_mod'];

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
def ofFn  (f: IndexVal dims -> α) : Tensor α dims :=
  ofFnMono (λ i => f (from_fin i))

def ofArray (a:Array α) : Tensor α [a.size] := {
  data := a,
  isEq := by
    unfold card;
    unfold card;
    simp
}

@[inline]
def ofList (l: List α) : Tensor α [l.length] := {
  data := Array.mk l,
  isEq := by
    unfold card;
    unfold card;
    simp
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
def get  (t: Tensor α dims) (i : IndexVal dims) : α :=
  getMono t (to_fin i)

-- instance to get element
instance : GetElem (Tensor α dims) (IndexVal dims) α (fun _ _ => true) where
  getElem xs i _ := xs.get i

@[inline]
def setMono (t: Tensor α dims) (i : Fin (card dims)) (a : α) : Tensor α dims :=
  -- prove that i ≤ v.data.length
  let i :=  ⟨i.val, (Nat.lt_of_lt_of_eq i.isLt (Eq.symm t.isEq))⟩;
  {
    data := Array.set t.data i a,
    isEq := Eq.trans (Array.size_set t.data i a) t.isEq
  }

@[inline]
def set  (t: Tensor α dims) (i : IndexVal dims) (a : α) : Tensor α dims :=
  setMono t (to_fin i) a

@[inline]
def zipWith (f: α → β → γ) (t1: Tensor α dims) (t2: Tensor β dims): Tensor γ dims :=
  -- create vector
  let v1: Vector α (card dims) := ⟨t1.data, t1.isEq⟩;
  let v2: Vector β (card dims) := ⟨t2.data, t2.isEq⟩;
  -- zipWith
  let v3 := Vector.zipWith f v1 v2;
  -- back to tensor
  {
    data := v3.data,
    isEq := v3.isEq
  }

@[inline]
def map (f: α → β) (t: Tensor α dims) : Tensor β dims := {
  data := Array.map f t.data,
  isEq := Eq.trans (Array.size_map f t.data) t.isEq
}

@[inline]
def mapIdxMono (f: Fin (card dims) → α → β) (t: Tensor α dims) : Tensor β dims :=
  letI f' := fun (i: Fin t.data.size) => f (Fin.mk i.val (Nat.lt_of_lt_of_eq i.isLt t.isEq));
  {
    data := Array.mapIdx t.data f',
    isEq := Eq.trans (Array.size_mapIdx t.data f') t.isEq
  }

@[inline]
def mapIdx (f: IndexVal dims → α → β) (t: Tensor α dims) : Tensor β dims :=
  mapIdxMono (λ i a => f (from_fin i) a) t

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

def mul [Mul α] (a: Tensor α [m, n]) (b: Tensor α [n, p]) : Tensor α [m, p] :=
  sorry

/-- Object permanence??? 😳 -/
@[simp]
theorem get_set_eq_mono  (t: Tensor α dims) (i: Fin (card dims)) (a: α)
  : Tensor.getMono (Tensor.setMono t i a) i = a
  := Array.get_set_eq t.data ⟨i, lt_n_lt_data_size t i⟩ a

@[simp]
theorem get_set_eq  (t: Tensor α dims) (i: IndexVal dims) (a: α)
  : Tensor.get (Tensor.set t i a) i = a
  := get_set_eq_mono t (to_fin i) a

/-- If we construct a vector through ofFn, then each element is the result of the function -/
@[simp]
theorem get_ofFnMono  (f: Fin (card dims) -> α) (i: Fin (card dims))
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

theorem get_mapMono (f: α → β) (t: Tensor α dims) (i: Fin (card dims))
  : (t.map f).getMono i = f (t.getMono i)
  := Array.getElem_map f t.data i (lt_n_lt_data_size (t.map f) i)

theorem get_map (f: α → β) (t: Tensor α dims) (i: IndexVal dims)
  : (t.map f).get i = f (t.get i)
  :=
    by
      unfold Tensor.get;
      rw [get_mapMono];

theorem get_mapIdxMono (f: Fin (card dims) → α → β) (t: Tensor α dims) (i: Fin (card dims))
  : (t.mapIdxMono f).getMono i = f i (t.getMono i)
  :=
    letI f' := fun (i: Fin t.data.size) => f (Fin.mk i.val (Nat.lt_of_lt_of_eq i.isLt t.isEq))
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
theorem extMono (t1 t2: Tensor α dims) (h : ∀ (i : Fin (card dims)), t1.getMono i = t2.getMono i) :
  t1 = t2
  :=
    -- prove that t1.data.size = t2.data.size
    have t1_data_size_eq_t2_data_size := t1.isEq.trans t2.isEq.symm
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
      t1 = ⟨t1.data, t1.isEq⟩ := by rfl
      _ = ⟨t2.data, t2.isEq⟩ := by simp [t1_data_eq_t2_data]
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
