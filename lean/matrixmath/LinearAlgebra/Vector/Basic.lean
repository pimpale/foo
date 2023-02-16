import Mathlib.Algebra.Group.Defs
import LinearAlgebra.Vector

namespace Vector
theorem add_assoc {α : Type u} [q:AddSemigroup α] {n: ℕ} (a b c:Vector α n) 
  : (a + b) + c = a + (b + c)
  :=
    have ab_c_eq_a_bc : (add (add a b) c) = (add a (add b c)) := by
      unfold add
      apply ext
      intro i
      rw [get_zipWith, get_zipWith, get_zipWith, get_zipWith]
      -- current goal is a[i] + b[i] + c[i] = a[i] + (b[i] + c[i])
      rw [q.add_assoc]

    ab_c_eq_a_bc

theorem add_comm {α : Type u} [q:AddCommSemigroup α] {n: ℕ} (a b: Vector α n) 
  : a + b = b + a
  :=
    have ab_eq_ba : (add a b) = (add b a) := by
      unfold add
      apply ext
      intro i
      rw [get_zipWith, get_zipWith]
      -- current goal is a[i] + b[i]  = b[i] + a[i]
      rw [q.add_comm]
    ab_eq_ba

theorem mul_assoc {α : Type u} [q:Semigroup α] {n: ℕ} (a b c:Vector α n) 
  : (a * b) * c = a * (b * c)
  :=
    have ab_c_eq_a_bc : (hadamard (hadamard a b) c) = (hadamard a (hadamard b c)) := by
      unfold hadamard
      apply ext
      intro i
      rw [get_zipWith, get_zipWith, get_zipWith, get_zipWith]
      -- current goal is a[i] * b[i] * c[i] = a[i] * (b[i] * c[i])
      rw [q.mul_assoc]

    ab_c_eq_a_bc

instance {α : Type u} [AddSemigroup α] {n:ℕ} : AddSemigroup (Vector α n) where
  add_assoc := add_assoc

instance {α : Type u} [AddCommSemigroup α] {n:ℕ} : AddCommSemigroup (Vector α n) where
  add_comm := add_comm

instance {α : Type u} [Semigroup α] {n:ℕ} : Semigroup (Vector α n) where
  mul_assoc := mul_assoc

theorem add_zero {α : Type u} [q:AddMonoid α] {n: ℕ} (a:Vector α n)
  : a + 0 = a
  :=
    have add_a_0_eq_a : (add a zero) = a := by
      unfold add zero
      apply ext
      intro i
      rw [get_zipWith, get_replicate]
      -- current goal is a[i] + 0 = a[i]
      rw [q.add_zero]
    add_a_0_eq_a

theorem zero_add {α : Type u} [q:AddMonoid α] {n: ℕ} (a:Vector α n)
  : 0 + a = a
  :=
    have add_0_a_eq_a : (add zero a) = a := by
      unfold add zero
      apply ext
      intro i
      rw [get_zipWith, get_replicate]
      -- current goal is 0 + a[i] = a[i]
      rw [q.zero_add]
    add_0_a_eq_a

instance {α : Type u} [AddMonoid α] {n:ℕ} : AddMonoid (Vector α n) where
  add_zero := add_zero
  zero_add := zero_add

theorem mul_one {α : Type u} [q:Monoid α] {n: ℕ} (a:Vector α n)
  : a * 1 = a
  :=
    have mul_a_1_eq_a : (hadamard a one) = a := by
      unfold hadamard one
      apply ext
      intro i
      rw [get_zipWith, get_replicate]
      -- current goal is a[i] * 1 = a[i]
      rw [q.mul_one]
    mul_a_1_eq_a

theorem one_mul {α : Type u} [q:Monoid α] {n: ℕ} (a:Vector α n)
  : 1 * a = a
  :=
    have mul_1_a_eq_a : (hadamard one a) = a := by
      unfold hadamard one
      apply ext
      intro i
      rw [get_zipWith, get_replicate]
      -- current goal is 1 * a[i] = a[i]
      rw [q.one_mul]
    mul_1_a_eq_a

instance {α : Type u} [Monoid α] {n:ℕ} : Monoid (Vector α n) where
  mul_one := mul_one
  one_mul := one_mul

instance {α : Type u} [AddCommMonoid α] {n:ℕ} : AddCommMonoid (Vector α n) where
  add_comm := add_comm

theorem mul_comm {α : Type u} [q:CommSemigroup α] {n: ℕ} (a b:Vector α n)
  : a * b = b * a
  :=
    have mul_a_b_eq_mul_b_a : (hadamard a b) = (hadamard b a) := by
      unfold hadamard
      apply ext
      intro i
      rw [get_zipWith, get_zipWith]
      -- current goal is a[i] * b[i] = b[i] * a[i]
      rw [q.mul_comm]
    mul_a_b_eq_mul_b_a

instance {α : Type u} [CommMonoid α] {n:ℕ} : CommMonoid (Vector α n) where
  mul_comm := mul_comm

theorem add_left_neg {α : Type u} [q:AddCommGroup α] {n: ℕ} (a:Vector α n)
  : -a + a = 0
  :=
    have neg_a_add_a_eq_0 : (add (neg a) a) = zero := by
      unfold add neg zero
      apply ext
      intro i
      rw [get_zipWith, get_map, get_replicate]
      -- current goal is -a[i] + a[i] = 0
      rw [q.add_left_neg]
    neg_a_add_a_eq_0

end Vector