import Lake
open Lake DSL

package «matrixmath»

lean_lib LinearAlgebra

@[default_target]
lean_exe «matrixmath» {
  root := `Main
}

require mathlib from git "https://github.com/leanprover-community/mathlib4.git"
