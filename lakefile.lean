import Lake
open Lake DSL
open System (FilePath)

def tyrLeanSharedLibRPath : String := run_io do
  let out ← IO.Process.output {
    cmd := "lean"
    args := #["--print-prefix"]
  }
  let leanPrefix := out.stdout.trimAscii.toString
  if out.exitCode == 0 && !leanPrefix.isEmpty then
    let leanSharedDir : FilePath := leanPrefix / "lib" / "lean"
    if ← leanSharedDir.pathExists then
      pure leanSharedDir.toString
    else
      let leanLibDir : FilePath := leanPrefix / "lib"
      if ← leanLibDir.pathExists then
        pure leanLibDir.toString
      else
        pure "@loader_path"
  else
    pure "@loader_path"

def linuxSystemLinkDirs : Array String :=
  #[
    "-L/usr/lib/x86_64-linux-gnu",
    "-L/lib/x86_64-linux-gnu",
    "-L/usr/lib/gcc/x86_64-linux-gnu/13",
    "-L/usr/lib/gcc/x86_64-linux-gnu/14",
    "-L/usr/lib/aarch64-linux-gnu",
    "-L/lib/aarch64-linux-gnu",
    "-L/usr/lib/gcc/aarch64-linux-gnu/13",
    "-L/usr/lib/gcc/aarch64-linux-gnu/14",
    "-L/usr/local/cuda/lib64",
    "-L/usr/local/cuda/targets/aarch64-linux/lib",
    "-L/usr/lib"
  ]

/-- CUDA driver stub link flags. Hosted CI usually has CUDA-enabled LibTorch
    but no CUDA driver stub, so only link `-lcuda` when the stub is present. -/
def linuxCudaDriverStubLinkArgs : Array String := run_io do
  if System.Platform.isOSX then return #[] else
  let envCandidates ←
    match (← IO.getEnv "CUDA_HOME") with
    | some home => pure #[s!"{home}/lib64/stubs"]
    | none => pure #[]
  let fallbackCandidates := #[
    "/usr/local/cuda/lib64/stubs",
    "/usr/local/cuda-13.0/lib64/stubs",
    "/usr/local/cuda-12.6/lib64/stubs",
    "/grid/it/easybuild/easybuild5/software/CUDA/12.9.1/stubs/lib64"
  ]
  let candidates := envCandidates ++ fallbackCandidates
  for stubsDir in candidates do
    let libcuda : FilePath := ⟨stubsDir⟩ / "libcuda.so"
    if ← libcuda.pathExists then
      return #[s!"-L{stubsDir}", "-lcuda"]
  return #[]

/-- CUDA link flags for Linux: vendored `libtorch_cuda` / `libc10_cuda` and the
    system `libcudart` are needed because `cc/build/libTyrC.a` whole-archives
    CUDA-using objects (TK kernels). Returns `#[]` if the vendored libtorch
    doesn't ship CUDA support — this keeps a CPU-only checkout linking. -/
def linuxCudaLinkArgs : Array String := run_io do
  let torchCudaCandidates : Array System.FilePath := #[
    ⟨(__dir__ / "external" / "libtorch" / "lib" / "libtorch_cuda.so").toString⟩,
    ⟨(__dir__ / "external" / "libtorch" / "lib" / "libtorch_cuda.dylib").toString⟩
  ]
  let hasTorchCuda ← torchCudaCandidates.anyM (·.pathExists)
  if hasTorchCuda then
    let vendorCuDir : FilePath :=
      __dir__ / "external" / "libtorch" / "lib" / ".." / ".." /
        "nvidia" / "cu13" / "lib"
    let vendorCublasLt := vendorCuDir / "libcublasLt.so.13"
    let cublasLtArgs :=
      if ← vendorCublasLt.pathExists then
        #[s!"-L{vendorCuDir}", "-l:libcublasLt.so.13",
          s!"-Wl,-rpath,{vendorCuDir}"]
      else
        #["-lcublasLt"]
    pure (#["-ltorch_cuda", "-lc10_cuda", "-lcudart"] ++ cublasLtArgs ++
      linuxCudaDriverStubLinkArgs)
  else
    pure #[]

/-- Lean v4.29's bundled `Scrt1.o` references `__libc_csu_init` / `__libc_csu_fini`,
    which were removed from glibc 2.34 (Ubuntu 24.04 ships glibc 2.39). Define
    both as zero so the link succeeds; glibc 2.34+ `__libc_start_main` ignores
    the legacy init/fini function-pointer arguments, so the symbols are never
    actually dereferenced. Scoped to Linux. -/
def linuxGlibc234CompatLinkArgs : Array String :=
  if System.Platform.isOSX then #[] else
    #["-Wl,--defsym=__libc_csu_init=0", "-Wl,--defsym=__libc_csu_fini=0"]

def linuxArrowLinkArgs : Array String := run_io do
  let candidates : Array System.FilePath := #[
    ⟨"/usr/lib/aarch64-linux-gnu/libarrow.so"⟩,
    ⟨"/usr/lib/x86_64-linux-gnu/libarrow.so"⟩,
    ⟨"/usr/lib/libarrow.so"⟩,
    ⟨"/usr/local/lib/libarrow.so"⟩
  ]
  let hasArrow ← candidates.anyM (·.pathExists)
  let parquetCandidates : Array System.FilePath := #[
      ⟨"/usr/lib/aarch64-linux-gnu/libparquet.so"⟩,
      ⟨"/usr/lib/x86_64-linux-gnu/libparquet.so"⟩,
      ⟨"/usr/lib/libparquet.so"⟩,
      ⟨"/usr/local/lib/libparquet.so"⟩
    ]
  let hasParquet ← parquetCandidates.anyM (·.pathExists)
  if hasArrow && hasParquet then
    pure #["-larrow", "-lparquet"]
  else
    pure #[]

/-- Return `none` for blank strings after trimming whitespace. -/
def nonEmptyTrimmed? (s : String) : Option String :=
  let trimmed := s.trimAscii.toString
  if trimmed.isEmpty then none else some trimmed

/-- Resolve the macOS SDK root from env or `xcrun` without hard-coded Xcode/CLT paths. -/
def normalizeMacOSSDKRoot (sdk : String) : IO String := do
  let sdkPath : FilePath := ⟨sdk⟩
  match sdkPath.parent with
  | some parent =>
      let stablePath := parent / "MacOSX.sdk"
      if ← stablePath.pathExists then
        pure stablePath.toString
      else
        pure sdk
  | none =>
      pure sdk

/-- Resolve the macOS SDK root from env or `xcrun` without hard-coded Xcode/CLT paths. -/
def macOSSDKRoot? : Option String := run_io do
  let envSdk? ← do
    match (← IO.getEnv "TYR_MACOS_SDKROOT") with
    | some p => pure (some p)
    | none => IO.getEnv "SDKROOT"
  match envSdk?.bind nonEmptyTrimmed? with
  | some p =>
      let normalized ← normalizeMacOSSDKRoot p
      if ← (⟨normalized⟩ : FilePath).pathExists then
        pure (some normalized)
      else
        pure none
  | none =>
    try
      let out ← IO.Process.output {
        cmd := "xcrun"
        args := #["--sdk", "macosx", "--show-sdk-path"]
      }
      if out.exitCode == 0 then
        match nonEmptyTrimmed? out.stdout with
        | some sdk =>
            pure (some (← normalizeMacOSSDKRoot sdk))
        | none =>
            pure none
      else
        pure none
    catch _ =>
      pure none

/-- Optional macOS SDK search flags when an SDK root can be discovered. -/
def macOSSDKLinkArgs : Array String :=
  match macOSSDKRoot? with
  | some sdk =>
    #[
      s!"-F{sdk}/System/Library/Frameworks",
      s!"-Wl,-syslibroot,{sdk}"
    ]
  | none => #[]

/-- Resolve macOS deployment target:
    `TYR_MACOS_DEPLOYMENT_TARGET` > `MACOSX_DEPLOYMENT_TARGET` > `14.0`.

Using the active SDK version here can overshoot the locally supported deployment
target when Xcode ships a newer SDK than the installed linker/runtime stack. -/
def macOSDeploymentTarget : String := run_io do
  let envTarget? ← do
    match (← IO.getEnv "TYR_MACOS_DEPLOYMENT_TARGET") with
    | some t => pure (some t)
    | none => IO.getEnv "MACOSX_DEPLOYMENT_TARGET"
  match envTarget?.bind nonEmptyTrimmed? with
  | some t => pure t
  | none => pure "14.0"

/-- macOS deployment-target link args to keep linker target aligned with local SDK/libs. -/
def macOSDeploymentLinkArgs : Array String :=
  if System.Platform.isOSX then
    #[s!"-mmacosx-version-min={macOSDeploymentTarget}"]
  else
    #[]

/-- Apple system frameworks used by the C++ bridge/runtime on macOS. -/
def macOSFrameworkArgs : Array String :=
  #[
    "-framework", "Foundation",
    "-framework", "CoreFoundation",
    "-framework", "Metal",
    "-framework", "CoreGraphics",
    "-framework", "ImageIO",
    "-framework", "AVFoundation",
    "-framework", "CoreMedia",
    "-framework", "CoreVideo",
    "-framework", "VideoToolbox",
    "-framework", "Accelerate",
    "-framework", "AudioToolbox"
  ]

/-- Prefer the locally built libsoxr from submodule source. -/
def soxrLinkArgs : Array String :=
  #[s!"-L{__dir__ / "cc" / "build" / "soxr" / "src"}", "-lsoxr"]

/-- Vendored LibTorch directory used by both Lean dynlibs and `cc/build/libTyrC.so`. -/
def linuxTorchLibDir : String :=
  (__dir__ / "external" / "libtorch" / "lib").toString

/-- Common Linux link tail shared by `packageLinkArgs` and `commonLinkArgs`:
    libtorch + CUDA (if vendored) + arrow/soxr + glibc-2.34 compat + rpath. -/
def linuxLinkTail : Array String :=
  #[
    s!"-L{__dir__ / "external" / "libtorch" / "lib"}",
    "-ltorch", "-ltorch_cpu", "-lc10"
  ] ++ linuxCudaLinkArgs ++ linuxSystemLinkDirs ++ soxrLinkArgs ++ linuxArrowLinkArgs
    ++ linuxGlibc234CompatLinkArgs ++ #[
    "-l:libgomp.so.1", "-l:libstdc++.so.6",
    s!"-Wl,-rpath,{linuxTorchLibDir}",
    "-Wl,-rpath,$ORIGIN/../../../external/libtorch/lib"
  ]

def packageLinkArgs : Array String :=
  if System.Platform.isOSX then
    #[
      s!"-L{__dir__ / "external" / "libtorch" / "lib"}",
      "-ltorch", "-ltorch_cpu", "-lc10",
      "-L/opt/homebrew/opt/libomp/lib", "-lomp",
      "-L/opt/homebrew/lib", "-larrow", "-lparquet"
    ] ++ soxrLinkArgs ++ macOSSDKLinkArgs ++ macOSDeploymentLinkArgs ++ macOSFrameworkArgs ++ #[
      "-Wl,-rpath,@loader_path/../../external/libtorch/lib",
      "-Wl,-rpath,@loader_path/../../../external/libtorch/lib",
      "-Wl,-rpath,@executable_path/../../../external/libtorch/lib",
      "-Wl,-rpath,/opt/homebrew/opt/libomp/lib",
      "-Wl,-rpath,/opt/homebrew/lib",
      s!"-Wl,-rpath,{tyrLeanSharedLibRPath}"
    ]
  else
    linuxLinkTail

def commonLinkArgs : Array String :=
  if System.Platform.isOSX then
    #[
      s!"{__dir__ / "cc" / "build" / "libTyrC.a"}",
      s!"-L{__dir__ / "external" / "libtorch" / "lib"}",
      "-ltorch", "-ltorch_cpu", "-lc10",
      "-L/opt/homebrew/opt/libomp/lib", "-lomp",
      "-L/opt/homebrew/lib", "-larrow", "-lparquet"
    ] ++ soxrLinkArgs ++ macOSSDKLinkArgs ++ macOSDeploymentLinkArgs ++ macOSFrameworkArgs ++ #[
      "-Wl,-rpath,@executable_path/../../../external/libtorch/lib",
      "-Wl,-rpath,/opt/homebrew/opt/libomp/lib",
      "-Wl,-rpath,/opt/homebrew/lib"
    ]
  else
    #[s!"{__dir__ / "cc" / "build" / "libTyrC.a"}"] ++ linuxLinkTail

package tyr where
  srcDir := "."
  buildDir := ".lake/build"
  moreServerArgs := #["-Dpp.unicode.fun=true"]
  moreLinkArgs := packageLinkArgs

require LeanTest from git "https://github.com/cpehle/lean_test.git" @ "b42cd3d78716e5a2de5b640ac82d7fe3f05f2a4c"
require LeanBenchmark from git "https://github.com/cpehle/lean-benchmark.git" @
  "9ab68a2e976aef3791b5b5630be8f5f1e8f79fe9"
require LeanUrdfTypeProvider from "../lean-urdf-typeprovider"

/-! ## Platform Detection

Use `System.Platform` for compile-time platform-specific link arguments and
runtime environment setup in scripts.
-/

/-- Check if we're on macOS. -/
def isMacOS : Bool :=
  System.Platform.isOSX

/-- OpenMP library path - macOS uses Homebrew, Linux uses system path -/
def getOmpLibPath : IO FilePath := do
  if isMacOS then
    let armPath : FilePath := "/opt/homebrew/opt/libomp/lib"
    if ← armPath.pathExists then
      return armPath
    let intelPath : FilePath := "/usr/local/opt/libomp/lib"
    if ← intelPath.pathExists then
      return intelPath
    return armPath
  else
    match (← IO.getEnv "EBROOTGCCCORE") with
    | some root =>
      let p : FilePath := root / "lib64"
      if (← p.pathExists) then
        return p
      else
        return "/usr/lib"
    | none =>
      return "/usr/lib"

def builtExecutablePath (rootPath : FilePath) (exeName : String) : FilePath :=
  rootPath / ".lake" / "build" / "bin" / exeName

def builtExecutableTracePath (rootPath : FilePath) (exeName : String) : FilePath :=
  rootPath / ".lake" / "build" / "bin" / s!"{exeName}.trace"

def ensureExecutablePath (path : FilePath) : IO Unit := do
  let chmod := if System.Platform.isWindows then "cmd" else "chmod"
  let chmodArgs :=
    if System.Platform.isWindows then
      #["/c", "exit", "0"]
    else
      #["+x", path.toString]
  let out ← IO.Process.output {
    cmd := chmod
    args := chmodArgs
  }
  if out.exitCode != 0 then
    throw <| IO.userError s!"Failed to mark {path} executable: {out.stderr}"

def builtExecutableLooksValid (path : FilePath) : IO Bool := do
  if !(← path.pathExists) then
    pure false
  else
    let out ← IO.Process.output {
      cmd := "file"
      args := #[path.toString]
    }
    if out.exitCode != 0 then
      pure false
    else
      let desc := out.stdout
      pure <|
        desc.contains "ELF " ||
        desc.contains "Mach-O " ||
        desc.contains "PE32" ||
        desc.contains "script text executable"

def builtExecutableLooksStale (rootPath : FilePath) (exeName : String) (exe : FilePath) : IO Bool := do
  let irRoot := rootPath / ".lake" / "build" / "ir"
  if !(← irRoot.pathExists) then
    pure false
  else
    let out ← IO.Process.output {
      cmd := "find"
      args := #[
        irRoot.toString,
        "-name", s!"{exeName}.c.o.export",
        "-newer", exe.toString,
        "-print",
        "-quit"
      ]
    }
    pure (out.exitCode == 0 && !out.stdout.trimAscii.isEmpty)

def extractTraceLinkCommand? (tracePath : FilePath) : IO (Option String) := do
  if !(← tracePath.pathExists) then
    pure none
  else
    let traceText ← IO.FS.readFile tracePath
    match traceText.splitOn ".> " with
    | _prefix :: after :: _rest =>
        match after.splitOn "\",\n" with
        | cmd :: _ => pure <| some cmd
        | [] =>
            match after.splitOn "\",\r\n" with
            | cmd :: _ => pure <| some cmd
            | [] => pure none
    | _ => pure none

def relinkBuiltExecutableToTmp (rootPath : FilePath) (exeName : String) : IO FilePath := do
  let originalExe := builtExecutablePath rootPath exeName
  let tracePath := builtExecutableTracePath rootPath exeName
  let some linkCmd ← extractTraceLinkCommand? tracePath
    | throw <| IO.userError s!"Missing relink trace for {exeName}: {tracePath}"
  let repairDir : FilePath := "/tmp/tyr_relinked"
  IO.FS.createDirAll repairDir
  let repairedExe := repairDir / exeName
  let patchedCmd := linkCmd.replace originalExe.toString repairedExe.toString
  let out ← IO.Process.output {
    cmd := "bash"
    args := #["-lc", patchedCmd]
    cwd := rootPath
  }
  if out.exitCode != 0 then
    throw <| IO.userError s!"Failed to relink {exeName} to {repairedExe}:\n{out.stderr}"
  ensureExecutablePath repairedExe
  if !(← builtExecutableLooksValid repairedExe) then
    throw <| IO.userError s!"Relinked executable is still invalid: {repairedExe}"
  pure repairedExe

/-! ## C++ Library Build -/

/-- Forward GPU build overrides from the outer environment into the C++ runtime
build so Lake and shell scripts compile `libTyrC` for the same target/family. -/
def gpuMakeEnv : IO (Array (String × Option String)) := do
  let gpuTarget? ← do
    match (← IO.getEnv "TYR_GPU_TARGET") with
    | some v => pure (some v)
    | none => IO.getEnv "GPU"
  let gpuFamily? ← do
    match (← IO.getEnv "TYR_GPU_FAMILY") with
    | some v => pure (some v)
    | none => IO.getEnv "GPU_FAMILY"
  let gpuCompute? ← do
    match (← IO.getEnv "TYR_GPU_COMPUTE") with
    | some v => pure (some v)
    | none => IO.getEnv "GPU_COMPUTE"
  let gpuCode? ← do
    match (← IO.getEnv "TYR_GPU_CODE") with
    | some v => pure (some v)
    | none => IO.getEnv "GPU_CODE"
  pure <| #[
    ("GPU", gpuTarget?.bind nonEmptyTrimmed?),
    ("GPU_FAMILY", gpuFamily?.bind nonEmptyTrimmed?),
    ("GPU_COMPUTE", gpuCompute?.bind nonEmptyTrimmed?),
    ("GPU_CODE", gpuCode?.bind nonEmptyTrimmed?)
  ]

/-- External library target for the C++ bindings.
    This wraps the Makefile build for now - a future enhancement could
    use Lake's native C++ compilation. -/
extern_lib libtyr pkg := do
  let tyrCLib := pkg.dir / "cc" / "build" / "libTyrC.a"
  let gpuIrRoot := pkg.buildDir / "ir" / "Tyr" / "GPU"
  let gpuKernelSrcRoot := pkg.dir / "Tyr" / "GPU" / "Kernels"
  let generatedCudaDir := pkg.dir / "cc" / "src" / "generated"
  let gpuCodegenConfigPath := pkg.buildDir / "libtyr_gpu_codegen.env"
  -- TYR_GPU_CODEGEN_MODULE may be a single module name OR a space-separated
  -- list of module names. The latter is convenient when a build needs more
  -- than one kernel module's CUDA emitted, e.g.
  -- `TYR_GPU_CODEGEN_MODULE="Tyr.GPU.Kernels.MhaH100 Tyr.GPU.Kernels.MhaH100Decode"`.
  let gpuCodegenModule :=
    match (← IO.getEnv "TYR_GPU_CODEGEN_MODULE") with
    | some moduleName => (nonEmptyTrimmed? moduleName).getD "Tyr.GPU.Kernels.MhaH100"
    | none => "Tyr.GPU.Kernels.MhaH100"
  let gpuCodegenModules : Array String :=
    (gpuCodegenModule.splitOn " ").toArray.filterMap (fun s => nonEmptyTrimmed? s)
  let skipGpuCodegenValue := (← IO.getEnv "TYR_SKIP_GPU_CODEGEN").getD ""
  let buildTyrCDylibValue := (← IO.getEnv "TYR_BUILD_TYRC_DYLIB").getD ""
  let gpuCodegenConfig :=
    s!"TYR_GPU_CODEGEN_MODULE={gpuCodegenModule}\nTYR_SKIP_GPU_CODEGEN={skipGpuCodegenValue}\nTYR_BUILD_TYRC_DYLIB={buildTyrCDylibValue}\n"
  let shouldWriteConfig ← do
    if ← gpuCodegenConfigPath.pathExists then
      pure ((← IO.FS.readFile gpuCodegenConfigPath) != gpuCodegenConfig)
    else
      pure true
  if shouldWriteConfig then
    -- On a fresh checkout (e.g. CI runner) `pkg.buildDir` may not yet
    -- exist; `writeFile` won't create it.
    IO.FS.createDirAll pkg.buildDir
    IO.FS.writeFile gpuCodegenConfigPath gpuCodegenConfig

  -- Track Makefile plus C/CUDA sources/headers so Lake reruns `make` when FFI changes.
  let makefileJob ← inputTextFile <| pkg.dir / "cc" / "Makefile"
  let gpuCodegenConfigJob ← inputTextFile gpuCodegenConfigPath
  let srcJob ← inputDir (pkg.dir / "cc" / "src") (text := true) fun p =>
    p.toString.endsWith ".cpp" || p.toString.endsWith ".mm" ||
      p.toString.endsWith ".cu" || p.toString.endsWith ".h"
  let toolJob ← inputDir (pkg.dir / "cc" / "tools") (text := true) fun p =>
    p.toString.endsWith ".py"
  -- Note: we deliberately do NOT watch `gpuKernelSrcRoot` (the kernel `.lean`
  -- source tree). Changes to a kernel `.lean` file flow through Lean
  -- compilation to its `.c.o.export`, which `gpuIrJob` already watches with
  -- the right scope (only the active codegen module's IR triggers a rebuild).
  -- Watching the whole `Tyr/GPU/Kernels/` directory caused every kernel-edit
  -- in the workspace to invalidate the libtyr build cascade.
  let _ := gpuKernelSrcRoot
  -- Fresh checkouts do not have the generated GPU IR tree yet.
  -- Create it so the optional IR scan can track later `.c.o.export` files instead of failing early.
  IO.FS.createDirAll gpuIrRoot
  let gpuIrJob ←
    if gpuCodegenModule == "Tyr.GPU.Kernels.MhaH100" then
      let mhaH100IrSuffixes : Array String := #[
        "Kernels/MhaH100.c.o.export",
        "Kernels/Prelude.c.o.export",
        "Types.c.o.export",
        "Codegen/Macros.c.o.export",
        "Codegen/Var.c.o.export",
        "Codegen/TileTypes.c.o.export",
        "Codegen/IR.c.o.export",
        "Codegen/Monad.c.o.export",
        "Codegen/AST.c.o.export",
        "Codegen/Primitives.c.o.export",
        "Codegen/Loop.c.o.export",
        "Codegen/GlobalLayout.c.o.export",
        "Codegen/EmitNew.c.o.export",
        "Codegen/Attribute.c.o.export",
        "Codegen/FFI.c.o.export",
        "Codegen/GenerateMain.c.o.export",
        "Codegen/Arch/Level.c.o.export"
      ]
      inputDir gpuIrRoot (text := false) fun p =>
        mhaH100IrSuffixes.any fun suffix => p.toString.endsWith suffix
    else
      inputDir gpuIrRoot (text := false) fun p =>
        p.toString.endsWith ".c.o.export"
  let depJob := makefileJob.mix gpuCodegenConfigJob |>.mix srcJob |>.mix toolJob |>.mix gpuIrJob

  buildFileAfterDep tyrCLib depJob fun _ => do
    let sysroot ← getLeanSysroot
    let gpuEnv ← gpuMakeEnv
    let extraEnv :=
      if System.Platform.isOSX then
        #[("MACOSX_DEPLOYMENT_TARGET", some macOSDeploymentTarget)]
      else
        #[]
    let skipGpuCodegen? ← IO.getEnv "TYR_SKIP_GPU_CODEGEN"
    if skipGpuCodegen?.getD "" != "1" then
      let generatorExe := pkg.dir / ".lake" / "build" / "bin" / "GenerateGpuKernels"
      proc {
        cmd := "lake"
        args := #["-R", "build", "GenerateGpuKernels"]
        cwd := pkg.dir
        env := #[
          ("LEAN_HOME", some sysroot.toString),
          ("TYR_SKIP_GPU_CODEGEN", some "1"),
          ("TYR_GPU_CODEGEN_MODULE", some gpuCodegenModule)
        ] ++ gpuEnv ++ extraEnv
      }
      let chmod := if System.Platform.isWindows then "cmd" else "chmod"
      let chmodArgs :=
        if System.Platform.isWindows then
          #["/c", "exit", "0"]
        else
          #["+x", generatorExe.toString]
      let chmodOut ← IO.Process.output { cmd := chmod, args := chmodArgs }
      if chmodOut.exitCode != 0 then
        IO.eprintln s!"warning: failed to mark {generatorExe} executable: {chmodOut.stderr}"
      let runnableGeneratorExe ←
        if ← builtExecutableLooksValid generatorExe then
          pure generatorExe
        else
          relinkBuiltExecutableToTmp pkg.dir "GenerateGpuKernels"
      proc {
        cmd := "lake"
        args := #["-R", "env", runnableGeneratorExe.toString]
                  ++ gpuCodegenModules
                  ++ #["--out-dir", generatedCudaDir.toString]
        cwd := pkg.dir
        env := #[
          ("LEAN_HOME", some sysroot.toString),
          ("TYR_SKIP_GPU_CODEGEN", some "1"),
          ("TYR_GPU_CODEGEN_MODULE", some gpuCodegenModule)
        ] ++ gpuEnv ++ extraEnv
      }
    let buildTyrCDylib :=
      match (← IO.getEnv "TYR_BUILD_TYRC_DYLIB") with
      | some "0" => false
      | _ => true
    let makeArgs :=
      if buildTyrCDylib then
        #["-C", (pkg.dir / "cc").toString, "lib", "dylib"]
      else
        #["-C", (pkg.dir / "cc").toString, "lib"]
    proc {
      cmd := "make"
      args := makeArgs
      env := #[
        ("LEAN_HOME", some sysroot.toString),
        ("TYR_GPU_CODEGEN_MODULE", some gpuCodegenModule)
      ] ++ gpuEnv ++ extraEnv
    }

/-! ## Lean Library -/

/-- Codegen-only sub-library.

    Owns `Tyr.GPU.Codegen.*` plus the small set of `Tyr.GPU.*` modules they
    depend on (Types, Capabilities, Tile) and `Tyr.Basic`. These modules are
    pure Lean — no FFI, no libtorch — so we explicitly disable
    `precompileModules`. That way `lean_exe GenerateGpuKernels` (whose root
    is `Tyr.GPU.Codegen.GenerateMain`) doesn't trigger the per-module `.so`
    cascade across all of `Tyr.*` whenever a kernel `.lean` file is touched. -/
lean_lib TyrCodegen where
  roots := #[
    `Tyr.GPU.Codegen,
    `Tyr.GPU.Types,
    `Tyr.GPU.Capabilities,
    `Tyr.GPU.Tile,
    `Tyr.Basic
  ]
  precompileModules := false

/-- Main Lean library containing all Tyr modules.

    `roots := #[\`Tyr]` claims everything under `Tyr.*` that isn't already
    owned by `TyrCodegen` (a more specific match for `Tyr.GPU.Codegen.*`
    etc. wins per Lake's lib resolution). -/
@[default_target]
lean_lib Tyr where
  roots := #[`Tyr]
  precompileModules := true

/-- Test library containing all tests -/
lean_lib Tests where
  roots := #[`Tests]
  precompileModules := false

/-- Experimental tests that track in-progress modules. -/
lean_lib TestsExperimental where
  roots := #[`TestsExperimental]
  precompileModules := false

/-- Examples library -/
lean_lib Examples where
  roots := #[`Examples]
  precompileModules := false

/-! ## Executables -/

/-- Main test runner using LeanTest -/
@[test_driver]
lean_exe test_runner where
  root := `Tests.RunTests
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Extra link args for `GenerateGpuKernels`. The auto-attached
    `libtyr.static` references CUDA driver API symbols
    (`cuTensorMapEncodeTiled`, `cuGetErrorString`) via `tk_vendor_mha_h100.o`,
    which the package-default flags don't pick up. CUDA toolchains ship a
    link-time stub for `libcuda.so` at `$CUDA_HOME/lib64/stubs`. Return no
    CUDA driver flags when the stub is absent, which is the normal hosted-CI
    CPU-stub path. -/
def codegenExeLinkArgs : Array String :=
  linuxCudaDriverStubLinkArgs ++ linuxGlibc234CompatLinkArgs

/-- Generate CUDA translation units from registered @[gpu_kernel] declarations.

    Its root `Tyr.GPU.Codegen.GenerateMain` lives in the codegen-only sub-lib
    `TyrCodegen` (above) so the per-module `.so` cascade across `Tyr.*` is
    skipped. `moreLinkArgs := codegenExeLinkArgs` overrides the package
    default — drops the heavy libtorch/arrow/parquet flags (we don't need
    them in a pure-Lean codegen tool) but keeps `-lcuda` when a CUDA driver
    stub is available for auto-attached `libtyr.static` CUDA references. -/
lean_exe GenerateGpuKernels where
  root := `Tyr.GPU.Codegen.GenerateMain
  supportInterpreter := true
  moreLinkArgs := codegenExeLinkArgs

/-- Compile registered @[tileir_kernel] declarations through NVIDIA TileIR tooling. -/
lean_exe GenerateTileIRKernels where
  root := `Tyr.GPU.Codegen.TileIR.GenerateMain
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Experimental test runner for unstable/in-progress modules. -/
lean_exe test_runner_experimental where
  root := `Tests.RunTestsExperimental
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- BranchingFlows continuous overfit training check. -/
lean_exe BranchingFlowsContinuousTrain where
  root := `Examples.BranchingFlows.ContinuousTrainDemo
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- BranchingFlows molecule overfit training check. -/
lean_exe BranchingFlowsMoleculeTrain where
  root := `Examples.BranchingFlows.MoleculeTrainDemo
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Molecule-shaped oracle generation with branch-event trajectory export. -/
lean_exe BranchingFlowsMoleculeGenerate where
  root := `Examples.BranchingFlows.MoleculeGenerationDemo
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- BranchingFlows molecule transformer overfit training check. -/
lean_exe BranchingFlowsMoleculeTransformerTrain where
  root := `Examples.BranchingFlows.MoleculeTransformerTrainDemo
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Dataset-backed BranchingFlows molecule transformer training and generation. -/
lean_exe BranchingFlowsMoleculeTrainGenerate where
  root := `Examples.BranchingFlows.MoleculeTrainGenerate
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Manual FFI failure-mode probe (intentionally crashes; not in the suite).
    Run it to check that an uncaught libtorch exception terminates with an
    intelligible message instead of a bare SIGABRT/SIGSEGV. -/
lean_exe ffi_crash_probe where
  root := `Tests.FfiCrashProbe
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Focused LeanTest runner for the Riemannian nanoGPT tests. -/
lean_exe RunRiemannianNanoGPTTests where
  root := `Tests.RunRiemannianNanoGPTTests
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- GPT training executable -/
lean_exe TrainGPT where
  root := `Examples.TrainGPT
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Exact-VJP Riemannian nanoGPT prototype runner. -/
lean_exe RunRiemannianNanoGPT where
  root := `Examples.GPT.RunRiemannianNanoGPT
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Diffusion training executable -/
lean_exe TrainDiffusion where
  root := `Examples.TrainDiffusion
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- URDF-backed hybrid contact event-skeleton simulation demo. -/
lean_exe RunUrdfContactExample where
  root := `Examples.EventSkeleton.RunUrdfContactExample
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- AlphaGrad-style RoeFlux_1d elimination planning port demo. -/
lean_exe AlphaGradRoeFlux1dA0 where
  root := `Examples.AlphaGradPort.RoeFlux1dA0
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- AlphaGrad port task sweep runner (targets tasks one-by-one). -/
lean_exe AlphaGradPortSweep where
  root := `Examples.AlphaGradPort.TaskSweep
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- AlphaGrad policy-training runner with real parameter updates. -/
lean_exe AlphaGradPolicyTrain where
  root := `Examples.AlphaGradPort.PolicyTrainMain
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- AlphaGrad policy-training sweep runner across tasks and training modes. -/
lean_exe AlphaGradPolicySweep where
  root := `Examples.AlphaGradPort.PolicySweepMain
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- NanoChat training executable (modded GPT + distributed) -/
lean_exe TrainNanoChat where
  root := `Examples.NanoChat.TrainNanoChat
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- NanoChat multi-stage pipeline executable. -/
lean_exe NanoChatPipeline where
  root := `Examples.NanoChat.Pipeline
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- NanoChat checkpoint-backed chat/inference executable. -/
lean_exe NanoChatChat where
  root := `Examples.NanoChat.RunChat
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Live microphone streaming Qwen3-ASR demo (macOS AudioToolbox input). -/
lean_exe Qwen3ASRLiveMic where
  root := `Examples.Qwen3ASR.LiveMic
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Separate streaming-native ASR session executable (parallel path). -/
lean_exe Qwen3ASRLiveMicTrueStream where
  root := `Examples.Qwen3ASR.LiveMicTrueStream
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Diffusion tests executable -/
lean_exe TestDiffusion where
  root := `Tests.TestDiffusion
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- DataLoader test executable -/
lean_exe TestDataLoader where
  root := `Tests.RunTestDataLoader
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Differential equation baseline test executable. -/
lean_exe TestDiffEq where
  root := `Tests.RunTestDiffEq
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Adjoint differential equation test executable. -/
lean_exe TestDiffEqAdjoint where
  root := `Tests.RunTestDiffEqAdjoint
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Core adjoint differential equation test executable. -/
lean_exe TestDiffEqAdjointCore where
  root := `Tests.RunTestDiffEqAdjointCore
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- GPU DSL regression test executable. -/
lean_exe TestGPUDSL where
  root := `Tests.RunTestGPUDSL
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- GPU kernel fixture test executable. -/
lean_exe TestGPUKernels where
  root := `Tests.RunTestGPUKernels
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end GPU parity tests (Tyr vs PyTorch, with optional vendored references). -/
lean_exe TestGPUE2E where
  root := `Tests.RunGPUE2E
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- GB10/Blackwell-specific end-to-end GPU parity tests. -/
lean_exe TestGPUGB10E2E where
  root := `Tests.RunGPUGB10E2E
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Laguna config.json parsing tests. -/
lean_exe LagunaConfigTest where
  root := `Tests.RunLagunaConfig
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Laguna tokenizer encode/decode tests. -/
lean_exe LagunaTokenizerTest where
  root := `Tests.RunLagunaTokenizer
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Laguna NVFP4 dequantization tests. -/
lean_exe LagunaNvFp4Test where
  root := `Tests.RunLagunaNvFp4
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Laguna MoE block (router + packed experts) tests. -/
lean_exe LagunaMoeTest where
  root := `Tests.RunLagunaMoe
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Laguna attention/model forward tests. -/
lean_exe LagunaModelTest where
  root := `Tests.RunLagunaModel
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Laguna end-to-end parity test vs the HF reference (tiny fixture). -/
lean_exe LagunaParityTest where
  root := `Tests.RunLagunaParity
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Laguna fused NVFP4 MoE kernel tests. -/
lean_exe LagunaFusedTest where
  root := `Tests.RunLagunaFused
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Laguna rotary (YaRN + plain) table tests. -/
lean_exe LagunaRopeTest where
  root := `Tests.RunLagunaRope
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Laguna-S-2.1 model loader/generation demo with HF repo-id resolution. -/
lean_exe LagunaRunHF where
  root := `Examples.Laguna.RunHF
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- NVIDIA TileIR rendering and toolchain driver tests. -/
lean_exe TestGPUTileIR where
  root := `Tests.RunTestGPUTileIR
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- TileIR export driver regression tests. -/
lean_exe TestTileIRGenerateMain where
  root := `Tests.RunTestTileIRGenerateMain
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Flux image generation demo -/
lean_exe FluxDemo where
  root := `Examples.Flux.FluxDemo
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end Qwen3-TTS demo (Lean talker + Python speech-tokenizer decode). -/
lean_exe Qwen3TTSEndToEnd where
  root := `Examples.Qwen3TTS.EndToEnd
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Offline KittenTTS / Kokoro synthesis demo using converted safetensors checkpoints. -/
lean_exe KittenTTSPretrained where
  root := `Examples.KittenTTSPretrained
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

lean_exe KittenTTSDurations where
  root := `Examples.KittenTTSDurations
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

lean_exe KittenTTSDebug where
  root := `Examples.KittenTTSDebug
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

lean_exe KittenTTSCompare where
  root := `Examples.KittenTTSCompare
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Offline Qwen3-ASR transcription demo (fully Lean pipeline). -/
lean_exe Qwen3ASRTranscribe where
  root := `Examples.Qwen3ASR.Transcribe
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Offline Whisper transcription demo (native Tyr encoder-decoder implementation). -/
lean_exe WhisperTranscribe where
  root := `Examples.Whisper.Transcribe
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Interactive Whisper voice mode with microphone input and silence detection. -/
lean_exe WhisperVoiceMode where
  root := `Examples.Whisper.VoiceMode
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Isolated test: in-memory Whisper transcription (no WAV round-trip). -/
lean_exe WhisperTranscribeInMem where
  root := `Examples.Whisper.TranscribeInMem
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Qwen3.5 model loader/generation demo with HF repo-id resolution. -/
lean_exe Qwen35RunHF where
  root := `Examples.Qwen35.RunHF
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Qwen2.5-Omni thinker text loader/generation demo (3B/7B). -/
lean_exe Qwen25OmniRunHF where
  root := `Examples.Qwen25Omni.RunHF
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Gemma 4 text loader/generation demo with HF repo-id resolution. -/
lean_exe Gemma4RunHF where
  root := `Examples.Gemma4.RunHF
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Flux debug harness (saves intermediate tensors) -/
lean_exe FluxDebug where
  root := `Examples.Flux.FluxDebug
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end demo for a minimal ThunderKittens-style copy kernel. -/
lean_exe RunCopy where
  root := `Examples.GPU.RunCopy
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end rotary fixture validation using a ThunderKittens-style kernel. -/
lean_exe RunRotary where
  root := `Examples.GPU.RunRotary
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end ThunderKittens layernorm fixture validation. -/
lean_exe RunLayerNorm where
  root := `Examples.GPU.RunLayerNorm
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end fused residual + RMSNorm fixture validation. -/
lean_exe RunRMSNorm where
  root := `Examples.GPU.RunRMSNormExe
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Fused BF16 cross-entropy training benchmark. -/
lean_exe RunLoss where
  root := `Examples.GPU.RunLoss
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Fused mixed-precision AdamW training benchmark. -/
lean_exe RunOptimizer where
  root := `Examples.GPU.RunOptimizer
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end ThunderKittens flash attention fixture validation. -/
lean_exe RunFlashAttn where
  root := `Examples.GPU.RunFlashAttn
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end FlashAttention3 validation. -/
lean_exe RunFlashAttn3 where
  root := `Examples.GPU.RunFlashAttn3
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Runtime validation for the high-level `tyr::flash_attn` bridge. -/
lean_exe RunFlashAttnOp where
  root := `Examples.GPU.RunFlashAttnOp
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- One-H100 benchmark scaffold for the `tyr::flash_attn` bring-up. -/
lean_exe RunFlashAttnBench where
  root := `Examples.GPU.RunFlashAttnBench
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Numerical correctness harness for the TK-style decode kernel.

    Generates random Q/K/V plus a torch SDPA reference for several decode
    shapes (Llama-3 head_dim=128, Qwen3-4B head_dim=64, tail-mask case,
    single-block case, batch>1) and asserts kernel parity within BF16
    tolerance. Also runs a cache-vs-no-cache parity test against
    `Cache.attendLayer`.

    Usage: `lake exe RunMhaH100Decode [--regen|--gen-only]`. -/
lean_exe RunMhaH100Decode where
  root := `Examples.GPU.RunMhaH100Decode
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Decode-specific perf benchmark: timed forward over the same shape
    matrix as `RunMhaH100Decode`, reporting p50 latency vs PyTorch SDPA
    plus a couple of long-context (kv_seq=8k) rows. Forward-only
    (decode is inference, no backward).

    Usage: `lake exe RunDecodeBench [--case all|<id>] [--backend all|tyr|sdpa]
                                    [--warmup N] [--iters N] [--repeats N]
                                    [--jsonl-out path]`. -/
lean_exe RunDecodeBench where
  root := `Examples.GPU.RunDecodeBench
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end ThunderKittens `mha_h100` forward/backward fixture validation. -/
lean_exe RunMhaH100 where
  root := `Examples.GPU.RunMhaH100
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end `mha_h100` training/benchmark demo (kernel + optional torch baseline). -/
lean_exe RunMhaH100Train where
  root := `Examples.GPU.RunMhaH100Train
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end GB10 MHA validation and synchronized benchmark. -/
lean_exe RunMhaGB10 where
  root := `Examples.GPU.RunMhaGB10
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end multi-block `mha_h100` validation (`seq=768`, `d=64`). -/
lean_exe RunMhaH100Seq768 where
  root := `Examples.GPU.RunMhaH100Seq768
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end Blackwell/B200 BF16 GEMM validation. -/
lean_exe RunB200Bf16Gemm where
  root := `Examples.GPU.RunB200Bf16Gemm
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-! ## Scripts -/

def gccCoreRuntimeLibPath? : IO (Option FilePath) := do
  match (← IO.getEnv "EBROOTGCCCORE") with
  | none => pure none
  | some root =>
    let p : FilePath := root / "lib64"
    if (← p.pathExists) then pure (some p) else pure none

def arrowRuntimeLibPath? : IO (Option FilePath) := do
  match (← IO.getEnv "EBROOTARROW") with
  | none => pure none
  | some root =>
    let p : FilePath := root / "lib"
    if (← p.pathExists) then
      pure (some p)
    else
      let p64 : FilePath := root / "lib64"
      if (← p64.pathExists) then pure (some p64) else pure none

def runtimeLibEnvVar : String :=
  if isMacOS then "DYLD_LIBRARY_PATH" else "LD_LIBRARY_PATH"

def leanRuntimeLibDir : IO FilePath := do
  let out ← IO.Process.output {
    cmd := "lean"
    args := #["--print-prefix"]
  }
  if out.exitCode != 0 then
    throw <| IO.userError s!"Failed to resolve Lean sysroot: {out.stderr}"
  pure <| (FilePath.mk out.stdout.trimAscii.toString) / "lib" / "lean"

def runtimeLibPath (rootPath : FilePath) : IO String := do
  let tyrCLib := rootPath / "cc" / "build"
  let lakeLib := rootPath / ".lake" / "build" / "lib"
  let libtorchPath := rootPath / "external" / "libtorch" / "lib"
  let leanLib ← leanRuntimeLibDir
  let ompPath ← getOmpLibPath
  let gccCoreLibPath? ← gccCoreRuntimeLibPath?
  let arrowLibPath? ← arrowRuntimeLibPath?
  let inheritedLibPath := (← IO.getEnv runtimeLibEnvVar)
  let baseLibPath := s!"{tyrCLib}:{lakeLib}:{libtorchPath}:{ompPath}:{leanLib}"
  let baseLibPath :=
    match arrowLibPath? with
    | some p => s!"{baseLibPath}:{p}"
    | none => baseLibPath
  let libPathPrefix :=
    match gccCoreLibPath? with
    | some p => s!"{baseLibPath}:{p}"
    | none => baseLibPath
  pure <|
    match inheritedLibPath with
    | some v => s!"{libPathPrefix}:{v}"
    | none => libPathPrefix

def ensureExecutable (path : FilePath) : IO Unit := do
  let chmod := if System.Platform.isWindows then "cmd" else "chmod"
  let chmodArgs :=
    if System.Platform.isWindows then
      #["/c", "exit", "0"]
    else
      #["+x", path.toString]
  let out ← IO.Process.output {
    cmd := chmod
    args := chmodArgs
  }
  if out.exitCode != 0 then
    throw <| IO.userError s!"Failed to mark {path} executable: {out.stderr}"

def runBuiltExecutable (rootPath : FilePath) (exeName : String) (args : Array String) : IO UInt32 := do
  let exe := builtExecutablePath rootPath exeName
  if !(← exe.pathExists) then
    throw <| IO.userError s!"Missing compiled executable {exe}. Build it first with `lake -R build {exeName}`."
  ensureExecutable exe
  let runnableExe ←
    if (← builtExecutableLooksValid exe) && !(← builtExecutableLooksStale rootPath exeName exe) then
      pure exe
    else
      relinkBuiltExecutableToTmp rootPath exeName
  let libPath ← runtimeLibPath rootPath
  let child ← IO.Process.spawn {
    cmd := runnableExe.toString
    args := args
    env := #[(runtimeLibEnvVar, some libPath)]
    stdin := .inherit
    stdout := .inherit
    stderr := .inherit
  }
  child.wait

private def lakeBuildArgs (targets : Array String) (reconfigure : Bool) : Array String :=
  (if reconfigure then #["-R", "build"] else #["build"]) ++ targets

private def lakeFailureLooksLikeReconfigure (stdout stderr : String) : Bool :=
  let text := (stdout ++ "\n" ++ stderr).toLower
  (text.contains "compiled configuration") ||
    (text.contains "package configuration") ||
    (text.contains "reconfigure") ||
    (text.contains "run again with -r")

private def runLakeBuildCapture (rootPath : FilePath) (targets : Array String)
    (reconfigure : Bool) : IO (UInt32 × String × String) := do
  let out ← IO.Process.output {
    cmd := "lake"
    args := lakeBuildArgs targets reconfigure
    cwd := rootPath
    env := #[("TYR_BUILD_TYRC_DYLIB", some "0")]
  }
  if !out.stdout.isEmpty then
    IO.print out.stdout
  if !out.stderr.isEmpty then
    IO.eprint out.stderr
  pure (out.exitCode, out.stdout, out.stderr)

def buildNamedExecutables (rootPath : FilePath) (targets : Array String) : IO UInt32 := do
  let (firstExitCode, stdout, stderr) ← runLakeBuildCapture rootPath targets false
  let exitCode ←
    if firstExitCode == 0 then
      pure firstExitCode
    else if lakeFailureLooksLikeReconfigure stdout stderr then
      IO.eprintln "lake build requested reconfigure; retrying with `lake -R build`."
      let child ← IO.Process.spawn {
        cmd := "lake"
        args := lakeBuildArgs targets true
        cwd := rootPath
        env := #[("TYR_BUILD_TYRC_DYLIB", some "0")]
        stdin := .inherit
        stdout := .inherit
        stderr := .inherit
      }
      child.wait
    else
      pure firstExitCode
  if exitCode == 0 then
    targets.forM fun exeName => do
      let exe := rootPath / ".lake" / "build" / "bin" / FilePath.mk exeName
      if ← exe.pathExists then
        ensureExecutable exe
  else
    pure ()
  pure exitCode

def buildGpuBackedTargets (rootPath : FilePath) (kernelModule : String) (targets : Array String) : IO UInt32 := do
  let child ← IO.Process.spawn {
    cmd := "lake"
    args := #["-R", "build"] ++ targets
    cwd := rootPath
    env := #[
      ("TYR_GPU_CODEGEN_MODULE", some kernelModule),
      ("TYR_BUILD_TYRC_DYLIB", some "0")
    ]
    stdin := .inherit
    stdout := .inherit
    stderr := .inherit
  }
  child.wait

/-- Script to run the test executable with proper environment.
    Usage: lake run -/
script run (args) do
  let rootPath := (← getWorkspace).root.dir
  return ← runBuiltExecutable rootPath "test_runner" args.toArray

/-- Script to run TrainGPT with proper environment.
    Usage: lake run train -/
script train (args) do
  let rootPath := (← getWorkspace).root.dir
  return ← runBuiltExecutable rootPath "TrainGPT" args.toArray

/-- Reconfigure once and build the raw H100 MHA example binaries together.
    This avoids paying the Lake replay twice and skips `libTyrC.so` because the
    compiled examples link `cc/build/libTyrC.a` directly. -/
script buildMhaH100Examples (_args) do
  let rootPath := (← getWorkspace).root.dir
  buildNamedExecutables rootPath #["RunMhaH100", "RunMhaH100Seq768"]

/-- Build GPU-backed Lake targets by requesting one kernel module through the normal
    `extern_lib libtyr` build flow instead of manually invoking `GenerateGpuKernels`
    and `make`.
    Usage:
      `lake -R run buildGpuTarget <KernelModule> <BuildTarget> [ExtraBuildTarget ...]`

    Lake versions differ on whether a script-level `--` separator is consumed or
    forwarded. Accept it in either position so the documented helper cannot
    accidentally request a kernel module literally named `--`. -/
script buildGpuTarget (args) do
  let args := if args.head? == some "--" then args.drop 1 else args
  if args.length < 2 then
    IO.eprintln "Usage: lake -R run buildGpuTarget <KernelModule> <BuildTarget> [ExtraBuildTarget ...]"
    pure 2
  else
    let rootPath := (← getWorkspace).root.dir
    let kernelModule := args[0]!
    let targets := args.drop 1 |>.toArray
    buildGpuBackedTargets rootPath kernelModule targets

/-- Run a compiled Lake executable from `.lake/build/bin` with the required runtime
    library path.
    Usage:
      `lake run runBuiltTarget -- <ExeName> [ExeArg ...]` -/
script runBuiltTarget (args) do
  if args.isEmpty then
    IO.eprintln "Usage: lake run runBuiltTarget -- <ExeName> [ExeArg ...]"
    pure 2
  else
    let rootPath := (← getWorkspace).root.dir
    let exeName := args[0]!
    runBuiltExecutable rootPath exeName (args.drop 1 |>.toArray)

/-- Run the compiled `RunMhaH100` executable with the correct runtime library path. -/
script runMhaH100Exe (args) do
  let rootPath := (← getWorkspace).root.dir
  return ← runBuiltExecutable rootPath "RunMhaH100" args.toArray

/-- Run the compiled `RunMhaH100Seq768` executable with the correct runtime library path. -/
script runMhaH100Seq768Exe (args) do
  let rootPath := (← getWorkspace).root.dir
  return ← runBuiltExecutable rootPath "RunMhaH100Seq768" args.toArray

/-- Build both raw H100 MHA example binaries, then run them back-to-back. -/
script validateMhaH100Examples (args) do
  let rootPath := (← getWorkspace).root.dir
  let buildExitCode ← buildNamedExecutables rootPath #["RunMhaH100", "RunMhaH100Seq768"]
  if buildExitCode != 0 then
    pure buildExitCode
  else
    let firstExitCode ← runBuiltExecutable rootPath "RunMhaH100" args.toArray
    if firstExitCode != 0 then
      pure firstExitCode
    else
      runBuiltExecutable rootPath "RunMhaH100Seq768" args.toArray
