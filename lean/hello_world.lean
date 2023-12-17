def main : IO Unit := do
  let stdout ← IO.getStdout
  stdout.putStrLn "Hello World!"
