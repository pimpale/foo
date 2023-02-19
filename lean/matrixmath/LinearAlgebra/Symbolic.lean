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
-- SMatrixElem and SVectorElem are the symbolic representation of the data
-- They form the graph. However they 
mutual
/-- Symbolic vector data -/
inductive SVectorElem where
-- From constant data
| of_float : (data : Vector Float n) -> (d: DType) -> SVectorElem
| of_int : (data : Vector Int n) -> (d: DType) -> SVectorElem
| of_nat : (data : Vector Nat n) -> (d: DType)-> SVectorElem
| of_bool : (data : Vector Bool n) -> (d: DType) -> SVectorElem
-- From Symbolic Matrix
| of_row : (data : SMatrixElem) -> (i : Fin m) -> SVectorElem
| of_col : (data : SMatrixElem) -> (i : Fin m) -> SVectorElem
-- Operations
| unary_op : (a: SVectorElem)  -> SVectorElem
| binary_op : (a b: SVectorElem) -> SVectorElem


inductive SMatrixElem where
-- From constant data
| of_float : (data : Matrix Float m n) -> (d: DType) -> SMatrixElem
| of_int : (data : Matrix Int m n) -> (d: DType) -> SMatrixElem
| of_nat : (data : Matrix Nat m n) -> (d: DType)-> SMatrixElem
| of_bool : (data : Matrix Bool m n) -> (d: DType) -> SMatrixElem
-- From Symbolic Vector
| row : SVectorElem -> SMatrixElem
| col : SVectorElem -> SMatrixElem
-- Symbolic
| input : (n m : Nat) -> (d: DType) -> SMatrixElem
end