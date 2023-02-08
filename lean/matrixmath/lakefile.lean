import Lake
open Lake DSL

package «matrixmath» {
  -- add package configuration options here
}

lean_lib «Matrix» {
  -- add library configuration options here
}

lean_lib «Vector» {
  -- add library configuration options here
}

require mathlib from git "https://github.com/leanprover-community/mathlib4.git"


@[default_target]
lean_exe «matrixmath» {
  root := `Main
}
