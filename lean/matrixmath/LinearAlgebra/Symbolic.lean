import LinearAlgebra.Vector
import LinearAlgebra.Matrix


/-- Represents valid data types that can be represented on the backend -/
inductive DType where
| f32
| f64
| u64
| u32
| u16
| u8
| i64
| i32
| i16
| i8
| b
deriving Repr

namespace DType 

def isFloat
| f64 => true
| f32 => true
| _ => false

def isSInt
| i64 => true
| i32 => true
| i16 => true
| i8 => true
| _ => false

def isUInt
| u64 => true
| u32 => true
| u16 => true
| u8 => true
| _ => false

def isBool
| b => true
| _ => false

def isInt (d:DType) : Bool :=
  d.isSInt ∨ d.isUInt

end DType

mutual
/-- Symbolic vector data -/
inductive SVector where
-- From constant data
| of_float : (data : Vector Float n) -> (d: DType) -> SVector
| of_int : (data : Vector Int n) -> (d: DType) -> SVector
| of_nat : (data : Vector Nat n) -> (d: DType)-> SVector
| of_bool : (data : Vector Bool n) -> (d: DType) -> SVector
-- From Symbolic Matrix
| of_row : (data : SMatrix) -> (i : Fin m) -> SVector
| of_col : (data : SMatrix) -> (i : Fin m) -> SVector
-- Symbolic
| input : (n : Nat) -> (d: DType) -> SVector
-- Operations
| unaryOp : (a: SVector) -> (b: SVector) -> SVector

inductive SMatrix where
-- From constant data
| of_float : (data : Matrix Float m n) -> (d: DType) -> SMatrix
| of_int : (data : Matrix Int m n) -> (d: DType) -> SMatrix
| of_nat : (data : Matrix Nat m n) -> (d: DType)-> SMatrix
| of_bool : (data : Matrix Bool m n) -> (d: DType) -> SMatrix
-- From Symbolic Vector
| row : SVector -> SMatrix
| col : SVector -> SMatrix
-- Symbolic
| input : (n m : Nat) -> (d: DType) -> SMatrix
end


mutual
def vectorSize (v: SVector) : Bool :=
  match v with
  | SVector.of_float _ _ => true
  | SVector.of_int _ _ => true
  | SVector.of_nat _ _ => true
  | SVector.of_bool _ _ => true
  | SVector.of_row _ _ => true
  | SVector.of_col _ _ => true
  | SVector.input _ _ (n : Nat) -> (d: DType) -> SVector
  | SVector.pointwiseOp : (a: SVector) -> (b: SVector) -> SVector

def isMatrixValid (v: SVector) : Bool :=
  sorry


end

structure LVector (n:ℕ) where
  data: SVector
  size: data.isValid == true


#eval (!v[0.5, 0.2])[0]

#check (SVector.of_float !v[0.5, 0.2] DType.f32)