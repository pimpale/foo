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
def empty {α : Type u} : Vector α 0 := {
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
  v.data.get ⟨i, (lt_n_lt_data_size v i)⟩

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
def truncateTR {α: Type u} {n : ℕ} (v: Vector α n) (n': ℕ) (h: n' ≤ n): Vector α n' :=  
  if h1: n = n' then
   v.proveLen (v.isEq.trans h1)
  else 
    have n'_ne_n := (Ne.intro h1).symm;
    have n'_lt_n := Nat.lt_of_le_of_ne h (n'_ne_n);
    have n'_succ_le_n := Nat.succ_le_of_lt n'_lt_n;
    v.pop.truncateTR n' (Nat.pred_le_pred n'_succ_le_n)

@[inline]
def truncate {α: Type u} {n : ℕ} (v: Vector α n) (n': ℕ) (h: n' ≤ n): Vector α n' :=
  Vector.ofFn (fun i => v[i])

@[specialize]
def zipWithAux {α β γ:Type u} {i n:ℕ} (f : α → β → γ) (as : Vector α n) (bs : Vector β n) (acc : Vector γ i) (h : i ≤ n) : Vector γ n :=
  if h1: i = n then
    acc.proveLen (acc.isEq.trans h1)
  else
    -- we have to use letI in order to not have to deal with let fns in the proof when we unfold
    letI h2: i < n := Nat.lt_of_le_of_ne h h1
    letI a := as[i]'h2;
    letI b := bs[i]'h2;
    zipWithAux f as bs (acc.push (f a b)) h2
termination_by _ => n-i
decreasing_by 
  simp_wf
  -- current goal: n - (i + 1) < n - i
  apply Nat.sub_add_lt_sub
  -- h₂ is 0 < 1
  case h₂ => exact Nat.one_pos
  have h2 := Nat.lt_of_le_of_ne h h1
  -- h₁ is 1 + 1 ≤ n
  case h₁ => exact Nat.succ_le_of_lt h2

@[inline]
def zipWith {α : Type u} {β : Type u} {γ : Type u} {n: Nat} (f: α → β → γ) (v1: Vector α n) (v2: Vector β n): Vector γ n :=
  zipWithAux f v1 v2 ⟨Array.mkEmpty n, rfl⟩ (by simp)


@[inline]
def map {α : Type u} {β : Type u} {n: ℕ} (f: α → β) (v: Vector α n) : Vector β n := {
  data := Array.map f v.data,
  isEq := Eq.trans (Array.size_map f v.data) v.isEq   
}

@[inline]
def mapIdx {α : Type u} {β : Type u} {n: ℕ} (f: Fin n → α → β) (v: Vector α n) : Vector β n := 
  letI f' := fun (i: Fin v.data.size) => f (Fin.mk i.val (i.isLt.trans_eq v.isEq));
  {
    data := Array.mapIdx v.data f',
    isEq := Eq.trans (Array.size_mapIdx v.data f') v.isEq   
  }


def zero [Zero α] {n:ℕ}: Vector α n := Vector.replicate n 0

def one [One α] {n:ℕ}: Vector α n := Vector.replicate n 1

def neg [Neg α] (v: Vector α n) : Vector α n := Vector.map (-·) v

def add [Add α] (v1: Vector α n) (v2: Vector α n) : Vector α n :=
  Vector.zipWith (·+·) v1 v2

def sub {α : Type u} [Sub α] {n: ℕ} (a b: Vector α n) : Vector α n :=
  Vector.zipWith (·-·) a b

def scale {α : Type u} [Mul α] {n: ℕ} (k: α) (v: Vector α n) : Vector α n := 
  v.map (fun x => k*x)

def hadamard {α : Type u} [Mul α] {n: ℕ} (a b: Vector α n) : Vector α n :=
  Vector.zipWith (·*·) a b  

def dot {α : Type u} [Add α] [Mul α] [Zero α] {n: ℕ} (a b: Vector α n) : α :=
  Array.foldl (·+·) 0 (Vector.zipWith (·*·) a b).data



-- Some theorems


/-- Object permanence??? 😳 -/
@[simp]
theorem get_set_eq {α: Type u} {n: ℕ} (v: Vector α n) (i: Fin n) (a: α)
  : Vector.get (Vector.set v i a) i = a
  := Array.get_set_eq v.data ⟨i, lt_n_lt_data_size v i⟩ a

/-- If we construct a vector through ofFn, then each element is the result of the function -/
@[simp]
theorem get_ofFn {n: Nat} (f: Fin n -> α) (i: Fin n) 
  : (ofFn f)[i] = f i
  :=
    -- prove that the i < Array.size (Array.ofFn f)
    have i_lt_size_ofFn_data : i.val < Array.size (Array.ofFn f) := lt_n_lt_data_size (ofFn f) i
    -- prove that v.data.get i = f i
    Array.getElem_ofFn f i.val i_lt_size_ofFn_data

/-- If we construct an array through mkArray then each element is the provided value -/
@[simp]
theorem Array_getElem_mk {n: Nat} (a:α) (i: ℕ) (h: i < Array.size (Array.mkArray n a)) 
  : (Array.mkArray n a)[i] = a
  := by
      -- here's a neat trick: in order to avoid "motive type correctness" issues, we can use get? instead of get
      -- let's execute this strategy:
      -- first, wrap both sides in some
      apply Option.some.inj
      -- move from (some (mkArray n a)[i]) to (mkArray n a).get? i
      have some_mkArray_i_eq_mkArray_i? := (Array.getElem?_eq_getElem (Array.mkArray n a) i h).symm
      rw [some_mkArray_i_eq_mkArray_i?]
      -- now we can use the fact that mkArray n a uses List.replicate n a
      rw [Array.getElem?_eq_data_get?]
      rw [mkArray_data]
      have mkArray_eq_n := Array.size_mkArray n a
      have replicate_eq_n := List.length_replicate n a
      have i_lt_replicate_n_length : i < (List.replicate n a).length := lt_of_lt_of_eq h (mkArray_eq_n.trans replicate_eq_n.symm)
      have get?_eq_get := List.get?_eq_get i_lt_replicate_n_length
      -- move back from get? to get
      rw [get?_eq_get]
      have get_replicate := List.get_replicate a ⟨i, i_lt_replicate_n_length⟩
      rw [get_replicate]


theorem get_eq_data_get {α : Type u} {n: Nat} (v : Vector α n) (i: Fin n)
  : v[i] = v.data.get ⟨i, lt_n_lt_data_size v i⟩
  := rfl

theorem get_eq_data_data_get {α : Type u} {n: Nat} (v : Vector α n) (i: Fin n)
  : v[i] = v.data.data.get ⟨i, lt_n_lt_data_size v i⟩
  := rfl

/-- If we construct a vector through replicate, then each element is the provided function -/
@[simp]
theorem get_replicate {n: Nat} (a:α) (i: Fin n) 
  : (replicate n a)[i] = a
  :=
    -- prove that the i < Array.size (Array.mkArray n a)
    have i_lt_size_mkArray_data : i.val < Array.size (Array.mkArray n a) := lt_n_lt_data_size (replicate n a) i
    -- prove that v.data.get i = f i
    Array_getElem_mk a i.val i_lt_size_mkArray_data


theorem get_truncate {α: Type u} {n : ℕ} (v: Vector α n) (n': ℕ) (h: n' ≤ n) (i : Fin n')
  : (v.truncate n' h)[i] = v[i]
  := get_ofFn (fun i => v[i]) i

theorem get_map {α : Type u} {β : Type u} {n: ℕ} (f: α → β) (v: Vector α n) (i: Fin n)
  : (v.map f)[i] = f v[i]
  := Array.getElem_map f v.data i (lt_n_lt_data_size (v.map f) i)


theorem get_mapIdx {α : Type u} {β : Type u} {n: ℕ} (f: Fin n → α → β) (v: Vector α n) (i: Fin n)
  : (v.mapIdx f)[i] = f i v[i]
  := 
    letI f' := fun (i: Fin v.data.size) => f (Fin.mk i.val (i.isLt.trans_eq v.isEq))
    Array.getElem_mapIdx v.data f' i (lt_n_lt_data_size (v.mapIdx f) i)

/-- After push, the last element of the array is what we pushed -/
@[simp]
theorem get_push_eq {α : Type u} {n: Nat} (v: Vector α n) (a: α)
  : (v.push a)[n] = a
  := 
    -- prove that n < n + 1
    have n_lt_n_plus_1
        : n < n + 1
        := Nat.lt_succ_self n
    -- prove that n < v.push.data.size
    have n_lt_push_data_size
        : n < (v.push a).data.size
        := lt_of_lt_of_eq n_lt_n_plus_1 (v.push a).isEq.symm
    -- prove that (Array.push v.data a)[Array.size v.data] = a
    have array_push_v_data_size_eq_a 
        : (Array.push v.data a)[v.data.size] = a
        := Array.get_push_eq v.data a
    -- prove that (Vector.push v a)[n] = a
    have array_push_v_data_n_eq_a : (Array.push v.data a)[n] = a := by
      convert array_push_v_data_size_eq_a
      rw [v.isEq]
    array_push_v_data_n_eq_a

/-- After push, the previous elements are the same -/
@[simp]
theorem get_push_lt {α : Type u} {n: Nat} (v: Vector α n) (a: α) (i: Fin n)
  : (v.push a)[i] = v[i]
  := 
    have i_lt_size_data : i.val < v.data.size := lt_n_lt_data_size v i
    Array.get_push_lt v.data a i.val i_lt_size_data

@[simp]
theorem get_push {α : Type u} {n: Nat} (v: Vector α n) (a: α) (i: Fin (n+1))
  : (v.push a)[i] = if h:i < n then v[i]'h else a
  := by
    split
    case inl =>
      rename _ => h1
      exact get_push_lt v a ⟨i, h1⟩
    case inr =>
      rename _ => h1
      have h2: i = n := Nat.le_antisymm (Nat.le_of_lt_succ i.isLt) (Nat.ge_of_not_lt h1)
      simp [get_push_lt, h2]

theorem get_zipWithAux
    (f : α → β → γ) (as : Vector α n) (bs : Vector β n) (acc : Vector γ i) (hin : i ≤ n)
    (hacc : ∀ (j:Fin i), acc[j] = f as[j] bs[j])
    (k: Fin n)
  : (zipWithAux f as bs acc hin)[k] = f as[k] bs[k]
  := by
      unfold zipWithAux
      split
      case inl =>
       rename _ => h1
       exact hacc ⟨k.val, (Nat.lt_of_lt_of_eq k.isLt h1.symm)⟩ 
      case inr =>
        rename _ => h1
        have hin_next: i + 1 ≤ n := Nat.succ_le_of_lt (Nat.lt_of_le_of_ne hin h1)
        exact get_zipWithAux 
          -- input elements
          f as bs 
          -- accumulator
          (acc.push (f as[i] bs[i]))
          -- proof that accumulator length is valid
          (hin_next)
          -- proof that accumulator is correct
          (by
            intro j
            -- we want to prove that (acc.push (f as[i] bs[i]))[j] = f as[j] bs[j]
            -- split into the case where j < i and j = i
            rw [get_push acc (f as[i] bs[i]) j]
            split
            -- case j < i
            case inl => 
              rename _ => h2
              -- prove that acc[j] = f as[j] bs[j]
              exact hacc ⟨j.val, h2⟩
            -- case j = i
            case inr =>
              rename _ => h2
              -- prove that f as[i] bs[i] = f as[j] bs[j]
              have h3 : j.val = i := Nat.le_antisymm (Nat.le_of_lt_succ j.isLt) (Nat.ge_of_not_lt h2)
              have h3' : j = ⟨i, Nat.lt.base i⟩ := Fin.eq_of_val_eq h3
              rw [h3']
              rfl
          )
          -- index we want to get
          k
termination_by _ => n - i

/-- proves the absurd if we have an instance of Fin 0-/
theorem Fin_0_absurd (i: Fin 0) : False
  := by
    have i_lt_0 : i.val < 0 := i.isLt
    exact Nat.not_lt_zero i.val i_lt_0

/-- If we construct a vector through zipWith, then the i'th element is f a[i] b[i] -/
@[simp]
theorem get_zipWith {α : Type u} {β : Type u} {γ : Type u} {n: Nat} (f: α → β → γ) (v1: Vector α n) (v2: Vector β n) (i: Fin n)
  : (Vector.zipWith f v1 v2)[i] = f v1[i] v2[i]
  := by unfold zipWith
        exact get_zipWithAux 
          -- input elements
          f v1 v2 
          -- accumulator
          ⟨Array.mkEmpty n, rfl⟩
          -- a proof that i ≤ n
          (by simp)
          -- a proof that for all j < i, acc[j] = f as[j] bs[j]
          -- note that in this case, i = 0, so we don't have to prove anything
          (by intro z; exact False.elim (Fin_0_absurd z))
          -- the index we want to get
          i

@[ext]
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
      v2 = v2 := by rfl
    v1_eq_v2


instance : Inhabited (Vector α 0) where default := empty
instance [Zero α] : Zero (Vector α n) where zero := zero
instance [One α] : One (Vector α n) where one := one
instance [Neg α] : Neg (Vector α n) where neg := neg
instance {α : Type u} [Add α] {n: ℕ} : Add (Vector α n) where add := add
instance {α : Type u} [Sub α] {n: ℕ} : Sub (Vector α n) where sub := sub
instance {α : Type u} [Mul α] {n: ℕ} : Mul (Vector α n) where mul := hadamard

end Vector
