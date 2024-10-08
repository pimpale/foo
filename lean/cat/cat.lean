def bufSize : USize := 20 * 1024

-- create file stream from filename
def createFileStream (filename: String) : IO (Option IO.FS.Stream) := do
  tryCatch
    ( do
      let handle ← IO.FS.Handle.mk filename IO.FS.Mode.read
      let stream := IO.FS.Stream.ofHandle handle
      pure (some stream)
    )
    ( fun e => do 
        println! s!"Error: {e}"
        pure none
    )

partial def dumpStream (stream: IO.FS.Stream) : IO Unit := do
  let stdout ← IO.getStdout
  let buf ← stream.read bufSize
  if buf.size > 0 then
    stdout.write buf
    dumpStream stream
  else
    pure ()

def catFilename (filename: String) : IO Unit := do
  let stream ← match filename with
    | "-" => some <$> IO.getStdout
    | f => createFileStream f
  
  match stream with
    | some s => dumpStream s
    | none => println! s!"Error: didn't print {filename}"

def main (args: List String) : IO Unit := do
  match args with
    | [] => catFilename "-"
    | _ => args.forM catFilename


#check λ t: Type => λ x: t => (x, x)