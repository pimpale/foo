import Lake
open Lake DSL

package «matrixmath» {
  -- add package configuration options here
}

lean_lib «MyMatrixImpl» {
  -- add library configuration options here
}

require mathlib from git "https://github.com/leanprover-community/mathlib4.git"


@[default_target]
lean_exe «matrixmath» {
  root := `Main
}
