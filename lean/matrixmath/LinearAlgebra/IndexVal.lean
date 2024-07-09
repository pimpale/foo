import Mathlib.Data.Nat.Defs

def dim_card : List ℕ -> ℕ
  | [] => 1
  | (n :: tail) => n * dim_card tail

inductive IndexVal : List ℕ -> Type where
  | Nil : IndexVal []
  | Cons : Fin n -> (IndexVal tail_t) -> IndexVal (n :: tail_t)

syntax "!ti[" withoutPosition(term,*,?) "]"  : term
macro_rules
  | `(!ti[ $elems,* ]) => do
    -- NOTE: we do not have `TSepArray.getElems` yet at this point
    let rec expandListLit (i : Nat) (skip : Bool) (result : Lean.TSyntax `term) : Lean.MacroM Lean.Syntax := do
      match i, skip with
      | 0,   _     => pure result
      | i+1, true  => expandListLit i false result
      | i+1, false => expandListLit i true  (← ``(IndexVal.Cons $(⟨elems.elemsAndSeps.get! i⟩) $result))
    let size := elems.elemsAndSeps.size
    expandListLit size (size % 2 == 0) (← ``(IndexVal.Nil))

namespace IndexVal

#check IndexVal.Cons 0 (IndexVal.Nil)
#check IndexVal [1, 2, 3]

-- def example_index : IndexVal [1, 2, 3] := IndexVal.Cons 0 (IndexVal.Cons 1 (IndexVal.Cons 2 IndexVal.Nil))

-- t : Tensor α [2, 2]
-- t = [[1, 2], [3, 4]]
-- t.get i![0, 1] = t.data[0*2 + 1]

def to_fin : IndexVal dims → Fin (dim_card dims)
  | IndexVal.Nil =>
    let v := 0;
    let hv := by
      unfold dim_card;
      simp;
    ⟨v,hv⟩
  | (IndexVal.Cons head tail) =>
    by
      -- n is the cardinality of the head
      -- tail_t is the tail
      rename_i n tail_t;
      let ⟨a, ha⟩ := head;
      let ⟨b, hb⟩ := to_fin tail;
      let v := a * dim_card tail_t + b
      let hv : v < dim_card (n :: tail_t) := by calc
            v < a * dim_card tail_t + dim_card tail_t := by
              apply Nat.add_lt_add_left hb
            _ ≤ n * dim_card tail_t := by
              rw [← Nat.succ_mul]
              apply Nat.mul_le_mul_right
              exact ha
      exact ⟨v, hv⟩

def from_fin {bound: List ℕ} (i: Fin (dim_card bound)): IndexVal bound :=
  match bound with
  | [] => IndexVal.Nil
  | n :: tail_t =>
  let ⟨i, hi⟩ := i;
    let q : ℕ := i / dim_card tail_t;
    let hq : q < n := Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hi);
    let b_gt_0 : 0 < dim_card tail_t := by
      apply Nat.pos_of_ne_zero;
      intro h;
      unfold dim_card at hi;
      rw [h] at hi;
      rw [Nat.mul_zero] at hi;
      contradiction;
    let r := i % dim_card tail_t;
    let hr : r < dim_card tail_t := Nat.mod_lt i (by assumption);
    IndexVal.Cons ⟨q, hq⟩ (from_fin ⟨r, hr⟩)

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

theorem mod_add {m n k : ℕ } (h: k < n) : (m * n + k) % n = k
  := by
      -- convert to (k + m * n) % n = k
      rw [Nat.add_comm];
      -- convert to k % n = k
      simp [Nat.add_mul_mod_self_left k m n];
      -- use k < n
      simp [Nat.mod_eq_of_lt h]

theorem bijection (it: List Nat) : ∀ (i : IndexVal it), from_fin (to_fin i) = i
 := by
    intro i;
    induction i with
    | Nil =>
      unfold from_fin;
      rfl
    | Cons head tail =>
      rename_i n tail_t a_ih;
      unfold from_fin to_fin;
      simp;
      apply And.intro;
      case Cons.left =>
        simp_all [div_add];
      case Cons.right =>
        simp_all [mod_add];

theorem bijection_inv (it: List Nat) : ∀ (i : Fin (dim_card it)), to_fin (from_fin i) = i
 := by
      intro i;
      induction it with
      | nil =>
        simp [dim_card, Fin.fin_one_eq_zero]
      | cons n tail_t ih =>
        unfold to_fin;
        unfold from_fin;
        simp [ih, Nat.div_add_mod'];

def splitAt (dims dims': List Nat) (v: IndexVal (dims ++ dims') ): IndexVal dims × IndexVal dims' :=
  sorry

instance : HAppend (IndexVal dims) (IndexVal dims') (IndexVal (dims ++ dims')) where
  hAppend := fun x y => sorry

end IndexVal

inductive IndexValR : List ℕ -> Type where
  | Nil : IndexValR []
  | ConsI : Fin n -> (IndexValR tail_t) -> IndexValR (n :: tail_t)
  | ConsR : (IndexValR tail_t) -> IndexValR (n :: tail_t)

syntax "!tr[" withoutPosition(term,*,?) "]" : term
macro_rules
  | `(!tr[ $elems,* ]) => do
    -- NOTE: we do not have `TSepArray.getElems` yet at this point
    let rec expandListLit (i : Nat) (skip : Bool) (result : Lean.TSyntax `term) : Lean.MacroM Lean.Syntax := do
      match i, skip with
      | 0,   _     => pure result
      | i+1, true  => expandListLit i false result
      | i+1, false => expandListLit i true (
        ← match (elems.elemsAndSeps.get! i)  with
          | Lean.Syntax.ident _ _ `all _ =>
            -- let z := dbgTrace "z"
            ``(IndexValR.ConsR $result)
          | _ => ``(IndexValR.ConsI $(⟨elems.elemsAndSeps.get! i⟩) $result)
      )
    let size := elems.elemsAndSeps.size
    expandListLit size (size % 2 == 0) (← ``(IndexValR.Nil))


namespace IndexValR

def ofIdx: (v: IndexVal dims) → IndexValR dims
  | IndexVal.Nil => IndexValR.Nil
  | (IndexVal.Cons head tail) => IndexValR.ConsI head (ofIdx tail)

/-- Dimensions of the resulting tensor -/
def result_dims: IndexValR dims -> List Nat
  | IndexValR.Nil => []
  | IndexValR.ConsI _ tail => result_dims tail
  | IndexValR.ConsR tail => by
     rename_i tail_t n
     exact n :: result_dims tail

@[simp]
theorem result_dims_ofIdx (v: IndexVal dims) : result_dims (ofIdx v) = [] :=
  by
    induction v with
    | Nil =>
      rfl
    | Cons head tail =>
      simp [ofIdx, result_dims]
      rename_i a_ih
      exact a_ih

/-- Gets a list of all indexes accessed by this query -/
def to_src_fin (v: IndexValR dims) (i: Fin (dim_card (result_dims v))): Fin (dim_card dims) :=
  sorry

instance : HAppend (IndexValR dims) (IndexValR dims') (IndexValR (dims ++ dims')) where
  hAppend := fun x y => sorry

instance : HAppend (IndexValR dims) (IndexVal dims') (IndexValR (dims ++ dims')) where
  hAppend := fun x y => sorry

instance : HAppend (IndexVal dims) (IndexValR dims') (IndexValR (dims ++ dims')) where
  hAppend := fun x y => sorry

end IndexValR
