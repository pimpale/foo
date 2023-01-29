def main : IO Unit := do
  let stdout ← IO.getStdout
  let handle ← IO.FS.Handle.mk "hello_world.lean4" IO.FS.Mode.read
  let data ← IO.FS.Handle.readToEnd handle
  stdout.putStr data