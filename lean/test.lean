axiom f : Nat → Nat

def thing : (Nat → (Nat → Nat)) :=
  λ x y => x + y


#check 1
#check thing 1 2
#check (thing 9) 1


#check Nat
#check Type
#check Type 1

#check @List.cons -- {α : Type u_1} → α → List α → List α

-- ∀ T:Type (T →  List T → List T)

-- normalize : ℝ × ℝ → ℝ × ℝ

-- 

inductive nat1 : Type
| zero : nat1
| succ : nat1 → nat1

open nat1
#check nat1
#check succ (succ zero)

